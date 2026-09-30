"""Temperature-normalized probability handling and seeded RNG."""
from __future__ import annotations

import torch


def greedy_token(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1).item())


def log_probs_at(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Gather log-probabilities of token_ids from logits with a matching last dim."""
    log_softmax = torch.log_softmax(logits.float(), dim=-1)
    return log_softmax.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def normalized_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature-scaled probabilities over the vocabulary (1-D tensor)."""
    return torch.softmax(logits.float() / temperature, dim=-1)


def residual_distribution(
    target_logits: torch.Tensor, draft_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Normalized positive part of (target - draft) distributions.


    Used for the correction token after a speculative-sampling rejection.
    Guaranteed to sum to one for finite model outputs.
    """
    p = normalized_probs(target_logits, temperature)
    q = normalized_probs(draft_logits, temperature)
    residual = torch.clamp(p - q, min=0.0)
    total = residual.sum()
    if total <= 0.0:
        # Degenerate numerical edge case: fall back to the target distribution.
        return p
    return residual / total


class Sampler:
    """Seeded CPU sampler. Every random draw funnels through one generator so
    same mode + same seed produces identical streams."""

    def __init__(self, seed: int):
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

    def sample(self, probs: torch.Tensor) -> int:
        return int(torch.multinomial(probs, num_samples=1, generator=self.generator).item())

    def uniform(self) -> float:
        return float(torch.rand(1, generator=self.generator, dtype=torch.float64).item())


def acceptance_probability(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    token_id: int,
    temperature: float,
) -> float:
    """min(1, p(x)/q(x)) with both models using the same temperature."""
    p = normalized_probs(target_logits, temperature)
    q = normalized_probs(draft_logits, temperature)
    qx = float(q[token_id].item())
    if qx <= 0.0:
        return 0.0 if float(p[token_id].item()) > 0.0 else 1.0
    return min(1.0, float(p[token_id].item()) / qx)
