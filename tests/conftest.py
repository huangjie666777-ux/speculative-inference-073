"""Shared fixtures: tiny CPU Llama bundles stand in for the real models."""
from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from specserve.models import ModelBundle


def _make_config(vocab_size: int = 64, seed: int = 0) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=True,
    )


def make_model(seed: int) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    model = LlamaForCausalLM(_make_config())
    model.eval()
    return model


class FakeTokenizer:
    eos_token_id = 1

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        # Deterministic encoding of short test strings.
        ids = [(ord(ch) % 62) + 2 for ch in text[:12]]
        if not ids:
            ids = [2]
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr((i - 2) % 26 + 97) for i in ids if i != 1)


@pytest.fixture
def bundles():
    tokenizer = FakeTokenizer()
    target_model = make_model(7)
    draft_model = make_model(7)  # identical weights: every greedy proposal matches
    target = ModelBundle(target_model, tokenizer, 1)
    draft = ModelBundle(draft_model, tokenizer, 1)
    return target, draft


@pytest.fixture
def divergent_bundles():
    tokenizer = FakeTokenizer()
    target = ModelBundle(make_model(7), tokenizer, 1)
    draft = ModelBundle(make_model(99), tokenizer, 1)
    return target, draft
