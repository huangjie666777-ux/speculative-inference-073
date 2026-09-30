"""Request validation, single active-request slot and generation orchestration."""
from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event, Lock
from typing import Iterator

from pydantic import BaseModel, Field

from .decoding import Decoder, GenerateStats
from .models import Engine, ModelBundle
from .sampling import Sampler


MAX_NEW_TOKENS_LIMIT = 512
MAX_GAMMA = 16


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    mode: str = Field(..., pattern="^(autoregressive|speculative)$")
    max_new_tokens: int = Field(..., ge=1, le=MAX_NEW_TOKENS_LIMIT)
    draft_steps: int = Field(..., ge=1, le=MAX_GAMMA)
    temperature: float = Field(..., ge=0.0, le=2.0)
    seed: int = Field(..., ge=0, le=2**31 - 1)


@dataclass
class Generation:
    token_ids: Iterator[int]
    stats: GenerateStats
    stop_event: Event
    tokenizer: object
    start: float


class BusyError(RuntimeError):
    pass


class GenerationService:
    """Allows exactly one active generation; validation runs before acquiring."""

    def __init__(self, engine: Engine | None = None):
        self._engine = engine or Engine()
        self._slot = Lock()

    @property
    def busy(self) -> bool:
        return self._slot.locked()

    def start(self, payload: GenerateRequest) -> Generation:
        # Acquire only after Pydantic validation succeeded, so illegal
        # parameters never occupy the running slot.
        if not self._slot.acquire(blocking=False):
            raise BusyError("another generation is active")
        try:
            target: ModelBundle = self._engine.target()
            draft: ModelBundle = self._engine.draft()
            prompt_ids = target.tokenizer(
                payload.text, return_tensors=None, add_special_tokens=True
            )["input_ids"]
            if not prompt_ids:
                raise ValueError("empty prompt after tokenization")
            stats = GenerateStats(mode=payload.mode)
            decoder = Decoder(
                target=target,
                draft=draft,
                stats=stats,
                sampler=Sampler(payload.seed),
                temperature=payload.temperature,
                max_new_tokens=payload.max_new_tokens,
            )
            stop_event = Event()
            decoder.is_cancelled = stop_event.is_set
            if payload.mode == "speculative":
                token_ids = decoder.speculative(prompt_ids, payload.draft_steps)
            else:
                token_ids = decoder.autoregressive(prompt_ids)
            return Generation(
                token_ids=token_ids,
                stats=stats,
                stop_event=stop_event,
                tokenizer=target.tokenizer,
                start=time.perf_counter(),
            )
        except BaseException:
            self._slot.release()
            raise

    def stop(self, generation: Generation) -> None:
        generation.stop_event.set()

    def finish(self, generation: Generation) -> None:
        generation.stats.elapsed_seconds = time.perf_counter() - generation.start
        if self._slot.locked():
            self._slot.release()
