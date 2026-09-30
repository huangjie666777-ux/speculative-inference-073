import torch

from specserve.sampling import (
    Sampler,
    acceptance_probability,
    normalized_probs,
    residual_distribution,
)


def test_temperature_normalization():
    logits = torch.tensor([1.0, 2.0, 3.0])
    hot = normalized_probs(logits, 0.5)
    cold = normalized_probs(logits, 2.0)
    assert hot.max() > cold.max()
    assert torch.isclose(hot.sum(), torch.tensor(1.0))
    assert torch.isclose(cold.sum(), torch.tensor(1.0))


def test_residual_sums_to_one_and_nonnegative():
    p_logits = torch.tensor([0.1, 1.0, 0.4])
    q_logits = torch.tensor([0.9, 0.2, 0.2])
    residual = residual_distribution(p_logits, q_logits, 1.0)
    assert (residual >= 0).all()
    assert torch.isclose(residual.sum(), torch.tensor(1.0), atol=1e-6)
    # Token 0 is more likely under the draft: residual mass there is zero.
    assert residual[0].item() == 0.0


def test_acceptance_probability_bounds():
    p_logits = torch.tensor([50.0, 0.0, 0.0])
    q_logits = torch.tensor([0.0, 50.0, 0.0])
    # token 1 is draft-certain but target has vanishing mass -> effectively reject
    assert acceptance_probability(p_logits, q_logits, 1, 1.0) < 1e-20
    # token 2 is (near) impossible under both -> treated as accept
    assert acceptance_probability(p_logits, q_logits, 2, 1.0) == 1.0
    # token 0 is target-certain, draft has vanishing mass -> accept
    assert acceptance_probability(p_logits, q_logits, 0, 1.0) == 1.0


def test_seeded_sampler_reproducible():
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    a = [Sampler(5).uniform() for _ in range(6)]
    b = [Sampler(5).uniform() for _ in range(6)]
    assert a == b
    c = [Sampler(6).uniform() for _ in range(6)]
    assert a != c
    assert Sampler(5).sample(probs) == Sampler(5).sample(probs)
