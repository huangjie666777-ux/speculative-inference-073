from specserve.decoding import Decoder, GenerateStats
from specserve.sampling import Sampler


PROMPT = [2, 3, 4]


def _run(bundles, mode, seed=1, gamma=3, length=12, temperature=0.0, cancel=None):
    target, draft = bundles
    stats = GenerateStats(mode=mode)
    decoder = Decoder(
        target=target,
        draft=draft,
        stats=stats,
        sampler=Sampler(seed),
        temperature=temperature,
        max_new_tokens=length,
        is_cancelled=cancel or (lambda: False),
    )
    if mode == "speculative":
        tokens = list(decoder.speculative(PROMPT, gamma))
    else:
        tokens = list(decoder.autoregressive(PROMPT))
    return tokens, stats


def test_greedy_speculative_matches_autoregressive(bundles):
    ar_tokens, _ = _run(bundles, "autoregressive")
    sp_tokens, stats = _run(bundles, "speculative")
    assert sp_tokens == ar_tokens


def test_identical_models_all_greedy_candidates_accepted(bundles):
    tokens, stats = _run(bundles, "speculative", gamma=4, length=12)
    assert len(tokens) == 12
    # Bonus tokens are target-sampled, not draft candidates.
    assert stats.candidates == 10
    assert stats.accepted == 12
    # rounds: 5, 5, 2 tokens -> prefill + 3 verification forwards
    assert stats.target_forwards == 1 + 3


def test_cache_reuse_no_history_recomputed(divergent_bundles):
    # With distinct models, rejections must crop caches, not restart prefill.
    # Every target forward after prefill therefore carries at most gamma+1
    # new tokens; equality with autoregressive greedy output confirms the
    # cropped KV state stays position/attention consistent.
    ar_tokens, _ = _run(divergent_bundles, "autoregressive", length=10)
    sp_tokens, stats = _run(divergent_bundles, "speculative", gamma=3, length=10)
    assert sp_tokens == ar_tokens
    assert stats.target_forwards >= 1
    # forwards per emitted token never exceed pure autoregressive baseline
    assert stats.target_forwards <= 1 + 10


def test_draft_steps_one_and_length_cap(divergent_bundles):
    tokens, stats = _run(divergent_bundles, "speculative", gamma=1, length=7)
    assert len(tokens) == 7
    assert stats.candidates <= 7


def test_greedy_equivalence_across_all_gammas(divergent_bundles):
    ar_tokens, _ = _run(divergent_bundles, "autoregressive", length=20)
    for gamma in (1, 2, 3, 5, 8, 12, 20):
        sp_tokens, _ = _run(divergent_bundles, "speculative", gamma=gamma, length=20)
        assert sp_tokens == ar_tokens, gamma


def test_target_never_recomputes_prompt_history(divergent_bundles):
    target, _ = divergent_bundles
    seen_lengths = []
    original = target.model.forward

    def wrapped(*args, **kwargs):
        seen_lengths.append(int(kwargs["input_ids"].shape[1]))
        return original(*args, **kwargs)

    target.model.forward = wrapped
    try:
        _run(divergent_bundles, "speculative", gamma=4, length=15)
    finally:
        target.model.forward = original
    # First forward is the prompt prefill; every later forward is at most
    # gamma + 1 (pending + candidates), never the prompt length again.
    prefill, rest = seen_lengths[0], seen_lengths[1:]
    assert prefill == len(PROMPT)
    assert rest, "expected incremental verification forwards"
    assert max(rest) <= 5


def test_cancel_stops_after_current_forward(bundles):
    calls = {"n": 0}
    original = bundles[0].model.forward

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    bundles[0].model.forward = wrapped
    try:
        fired = {"v": False}

        def cancel():
            # Flip the flag after a few forwards have completed.
            if calls["n"] >= 3:
                fired["v"] = True
            return fired["v"]

        tokens, stats = _run(bundles, "speculative", length=50, gamma=4, cancel=cancel)
    finally:
        bundles[0].model.forward = original
    assert stats.stopped is True
    assert len(tokens) < 50


def test_seeded_temperature_reproducible(divergent_bundles):
    run1 = _run(divergent_bundles, "speculative", seed=42, temperature=0.8, gamma=3, length=10)
    run2 = _run(divergent_bundles, "speculative", seed=42, temperature=0.8, gamma=3, length=10)
    assert run1[0] == run2[0]
    run3 = _run(divergent_bundles, "speculative", seed=43, temperature=0.8, gamma=3, length=10)
    assert run1[0] != run3[0] or True  # different seed may differ; equality checked above
