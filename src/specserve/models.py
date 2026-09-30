"""Lazy CPU loading of the target/draft SmolLM2 models and shared tokenizer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DRAFT_DIR = ROOT / "models" / "draft"
TARGET_DIR = ROOT / "models" / "target"


@dataclass(frozen=True)
class ModelBundle:
    model: object
    tokenizer: object
    eos_token_id: int


class Engine:
    """Process-wide singleton holder. Loading is guarded so concurrent requests
    trigger only one load; inference itself runs under the request slot lock."""

    def __init__(self, draft_dir: Path = DRAFT_DIR, target_dir: Path = TARGET_DIR):
        self.draft_dir = Path(draft_dir)
        self.target_dir = Path(target_dir)
        self._lock = Lock()
        self._draft: ModelBundle | None = None
        self._target: ModelBundle | None = None

    @staticmethod
    def _load_model(model_dir: Path) -> object:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float32,
            local_files_only=True,
            use_cache=True,
        )
        model.to("cpu")
        model.eval()
        return model

    def _load(self, role: str) -> ModelBundle:
        with self._lock:
            cached = getattr(self, f"_{role}")
            if cached is not None:
                return cached
            model_dir = self.draft_dir if role == "draft" else self.target_dir
            tokenizer = AutoTokenizer.from_pretrained(str(self.target_dir), local_files_only=True)
            model = self._load_model(model_dir)
            eos = tokenizer.eos_token_id
            if eos is None:
                eos = int(model.generation_config.eos_token_id)
            bundle = ModelBundle(model=model, tokenizer=tokenizer, eos_token_id=int(eos))
            setattr(self, f"_{role}", bundle)
            return bundle

    def draft(self) -> ModelBundle:
        return self._load("draft")

    def target(self) -> ModelBundle:
        return self._load("target")


engine = Engine()
