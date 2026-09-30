"""CPU decoding core: autoregressive baseline and hand-written speculative loop.

The speculative loop implements proposal / verification / accept-reject
mechanics directly instead of delegating to a built-in assisted generation
API. Each model keeps its own DynamicCache across rounds; after every round
both caches are cropped to the same confirmed prefix length, while position
ids and attention masks follow the cache length, so prompt history is never
recomputed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import torch
from transformers import DynamicCache

from .sampling import (
    Sampler,
    acceptance_probability,
    greedy_token,
    normalized_probs,
    residual_distribution,
)


@dataclass
class GenerateStats:
    mode: str
    candidates: int = 0
    accepted: int = 0
    target_forwards: int = 0
    elapsed_seconds: float = 0.0
    stopped: bool = False

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "candidates": self.candidates,
            "accepted": self.accepted,
            "target_forwards": self.target_forwards,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "stopped": self.stopped,
        }


@dataclass
class _ModelState:
    """Per-request model state; never shared between requests."""

    model: object
    cache: DynamicCache
    position: int = 0

    def forward(self, token_ids: list[int], stats: GenerateStats | None = None) -> torch.Tensor:
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        position_ids = torch.arange(
            self.position, self.position + len(token_ids), dtype=torch.long
        ).unsqueeze(0)
        attention_mask = torch.ones(
            1, self.position + len(token_ids), dtype=torch.long
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=self.cache,
            use_cache=True,
        )
        self.position += len(token_ids)
        if stats is not None:
            stats.target_forwards += 1
        return outputs.logits[0].float()

    def crop(self, length: int) -> None:
        if length < self.position:
            self.cache.crop(length)
            self.position = length


CancelCheck = Callable[[], bool]


@dataclass
class Decoder:
    target: object
    draft: object
    stats: GenerateStats
    sampler: Sampler
    temperature: float
    max_new_tokens: int
    is_cancelled: CancelCheck = field(default=lambda: False)

    def _sample(self, logits: torch.Tensor) -> int:
        if self.temperature == 0.0:
            return greedy_token(logits)
        return self.sampler.sample(normalized_probs(logits, self.temperature))

    def autoregressive(self, prompt_ids: list[int]) -> Iterator[int]:
        state = _ModelState(self.target.model, DynamicCache())
        logits = state.forward(prompt_ids, self.stats)[-1]
        emitted = 0
        while emitted < self.max_new_tokens:
            token_id = self._sample(logits)
            emitted += 1
            yield token_id
            if token_id == self.target.eos_token_id or emitted >= self.max_new_tokens:
                return
            if self.is_cancelled():
                self.stats.stopped = True
                return
            logits = state.forward([token_id], self.stats)[-1]

    def speculative(self, prompt_ids: list[int], gamma: int) -> Iterator[int]:
        """Standard speculative decoding with exact target marginals.

        Round token bookkeeping (P = confirmed prefix length, b = pending
        emitted token carried from the previous round, 0 or 1):

        * Draft advances from P with an incremental forward for b (if any) and
          for all gamma proposed tokens. It therefore performs gamma + b
          cheap forwards and its cache ends at P + b + gamma.
        * Target performs a SINGLE forward over b + gamma tokens; when b == 0
          the prompt-prefill target logits evaluate the first candidate.
        * Greedy: accept the longest prefix matching target argmax; the first
          divergence is replaced by the target token. Sampling: accept with
          min(1, p/q); the correction comes from normalized max(0, p - q).
        * If every candidate is accepted the target's own bonus token is also
          emitted, preserving the target sampling distribution.
        * Both caches crop to P + b + accepted, i.e. exactly the emitted
          confirmed prefix. The correction/bonus becomes the next b.
        """
        target_state = _ModelState(self.target.model, DynamicCache())
        draft_state = _ModelState(self.draft.model, DynamicCache())

        first_target_logits: torch.Tensor | None = target_state.forward(
            prompt_ids, self.stats
        )[-1]
        next_draft_logits: torch.Tensor | None = draft_state.forward(prompt_ids)[-1]

        emitted = 0
        pending: int | None = None

        while emitted < self.max_new_tokens:
            has_pending = pending is not None

            # ---- Draft proposes gamma tokens one at a time ----
            draft_q: list[torch.Tensor] = []
            candidates: list[int] = []
            budget = min(gamma, self.max_new_tokens - emitted)
            if budget <= 0:
                return
            if has_pending:
                next_draft_logits = draft_state.forward([pending])[-1]
            for _ in range(budget):
                token_id = self._sample(next_draft_logits)
                draft_q.append(next_draft_logits)
                candidates.append(token_id)
                self.stats.candidates += 1
                next_draft_logits = draft_state.forward([token_id])[-1]

            # ---- Target verifies pending + candidates in one forward ----
            verifier_input = ([pending] if has_pending else []) + candidates
            verifier_logits = target_state.forward(verifier_input, self.stats)

            def eval_logits(index: int) -> torch.Tensor:
                if has_pending:
                    return verifier_logits[index]
                if index == 0:
                    assert first_target_logits is not None
                    return first_target_logits
                return verifier_logits[index - 1]

            rejected_at: int | None = None
            correction: int | None = None
            for i, candidate in enumerate(candidates):
                p_logits = eval_logits(i)
                if self.temperature == 0.0:
                    target_token = greedy_token(p_logits)
                    if candidate == target_token:
                        continue
                    rejected_at, correction = i, target_token
                    break
                if self.sampler.uniform() < acceptance_probability(
                    p_logits, draft_q[i], candidate, self.temperature
                ):
                    continue
                rejected_at = i
                correction = self.sampler.sample(
                    residual_distribution(p_logits, draft_q[i], self.temperature)
                )
                break

            accepted_count = len(candidates) if rejected_at is None else rejected_at

            # Confirm accepted candidate prefix.
            for token_id in candidates[:accepted_count]:
                emitted += 1
                self.stats.accepted += 1
                yield token_id
                if token_id == self.target.eos_token_id or emitted >= self.max_new_tokens:
                    return

            if rejected_at is None:
                follow_up = self._sample(verifier_logits[-1])
            else:
                follow_up = correction
            emitted += 1
            self.stats.accepted += 1
            yield follow_up
            if follow_up == self.target.eos_token_id or emitted >= self.max_new_tokens:
                return

            # Both caches currently end at P + offset + gamma. Crop to the
            # confirmed prefix P + offset + accepted_count; they stay equal
            # in length and positions/masks resynchronize automatically.
            confirmed_length = target_state.position - (
                len(candidates) - accepted_count
            )
            assert draft_state.position == target_state.position
            target_state.crop(confirmed_length)
            draft_state.crop(confirmed_length)

            # The logits at the confirmed tail predict `follow_up`; the draft
            # recomputes from it next round via the pending forward. For the
            # target, when no pending existed the prefill logits are consumed;
            # subsequent first-candidate evaluation always comes from the
            # verifier forward over the pending token.
            pending = follow_up
            first_target_logits = None
            next_draft_logits = None

            if self.is_cancelled():
                self.stats.stopped = True
                return
