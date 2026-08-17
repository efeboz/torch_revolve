import torch

from torchrevolve.model import TinyGPT, TinyGPTConfig
from torchrevolve.revolve import make_schedule as make_revolve_schedule
from torchrevolve.training import train_baseline, train_scheduled


def test_model_forward_and_loss() -> None:
    torch.manual_seed(7)
    config = TinyGPTConfig(
        vocab_size=32,
        max_sequence_length=8,
        depth=2,
        width=16,
        heads=4,
    )
    model = TinyGPT(config)
    tokens = torch.randint(config.vocab_size, (2, 8))
    logits = model(tokens)
    loss = model(tokens, tokens.roll(-1, dims=1))
    assert logits.shape == (2, 8, config.vocab_size)
    assert loss.ndim == 0
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_baseline_training_is_bitwise_reproducible() -> None:
    config = TinyGPTConfig(
        vocab_size=32,
        max_sequence_length=8,
        depth=2,
        width=16,
        heads=4,
        dropout=0.2,
    )
    first = train_baseline(config, steps=3, batch_size=2, sequence_length=8, seed=11)
    second = train_baseline(config, steps=3, batch_size=2, sequence_length=8, seed=11)
    assert first.losses == second.losses
    assert first.state.keys() == second.state.keys()
    assert all(
        torch.equal(first.state[name], second.state[name]) for name in first.state
    )


def test_200_step_revolve_loss_curve_matches_baseline() -> None:
    config = TinyGPTConfig(
        vocab_size=16,
        max_sequence_length=4,
        depth=2,
        width=8,
        heads=2,
        dropout=0.2,
    )
    options = {"steps": 200, "batch_size": 1, "sequence_length": 4, "seed": 41}
    expected = train_baseline(config, **options)
    actual = train_scheduled(
        config,
        lambda profile: make_revolve_schedule(profile, budget=1),
        **options,
    )
    assert actual.losses == expected.losses
    assert actual.state.keys() == expected.state.keys()
    assert all(
        torch.equal(actual.state[name], expected.state[name]) for name in actual.state
    )
