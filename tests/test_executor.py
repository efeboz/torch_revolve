import copy

import pytest
import torch
from torch.nn import functional as F

from torchrevolve.chain import BlockChain
from torchrevolve.dp import make_schedule as make_dp_schedule
from torchrevolve.executor import run_scheduled_backward
from torchrevolve.heuristics import (
    make_all_schedule,
    make_none_schedule,
    make_selective_schedule,
    make_uniform_schedule,
)
from torchrevolve.model import TinyGPT, TinyGPTConfig
from torchrevolve.revolve import make_schedule as make_revolve_schedule


def language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


def gradients(model: TinyGPT) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def baseline(
    model: TinyGPT,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    model.zero_grad(set_to_none=True)
    loss = model(tokens, targets)
    loss.backward()
    return loss.detach(), gradients(model)


@pytest.mark.parametrize("granularity", ["coarse", "fine"])
@pytest.mark.parametrize("dropout", [0.0, 0.25])
def test_revolve_gradients_are_bitwise_equal(granularity: str, dropout: float) -> None:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(5)
    config = TinyGPTConfig(
        vocab_size=32,
        max_sequence_length=8,
        depth=3,
        width=16,
        heads=4,
        dropout=dropout,
    )
    reference = TinyGPT(config)
    scheduled = copy.deepcopy(reference)
    tokens = torch.randint(config.vocab_size, (2, 8))
    targets = torch.randint(config.vocab_size, (2, 8))
    expected_loss, expected_gradients = baseline(reference, tokens, targets, seed=19)

    chain = BlockChain.from_model(scheduled, granularity=granularity)
    profile = chain.profile(tokens, n_reps=1)
    schedule = make_revolve_schedule(profile, budget=2)
    torch.manual_seed(19)
    result = run_scheduled_backward(
        chain,
        schedule,
        (tokens, targets),
        language_model_loss,
    )
    assert torch.equal(result.loss, expected_loss)
    assert result.gradients.keys() == expected_gradients.keys()
    assert all(
        torch.equal(result.gradients[name], expected_gradients[name])
        for name in result.gradients
    )


def test_dp_gradients_are_bitwise_equal() -> None:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(7)
    config = TinyGPTConfig(
        vocab_size=32,
        max_sequence_length=8,
        depth=2,
        width=16,
        heads=4,
        dropout=0.2,
    )
    reference = TinyGPT(config)
    scheduled = copy.deepcopy(reference)
    tokens = torch.randint(config.vocab_size, (2, 8))
    targets = torch.randint(config.vocab_size, (2, 8))
    expected_loss, expected_gradients = baseline(reference, tokens, targets, seed=23)

    chain = BlockChain.from_model(scheduled, granularity="fine")
    profile = chain.profile(tokens, n_reps=1)
    budget = profile.units[0].effective_state_bytes + max(
        unit.activation_bytes for unit in profile.units
    )
    schedule = make_dp_schedule(profile, budget=budget)
    torch.manual_seed(23)
    result = run_scheduled_backward(
        chain,
        schedule,
        (tokens, targets),
        language_model_loss,
    )
    assert torch.equal(result.loss, expected_loss)
    assert all(
        torch.equal(result.gradients[name], expected_gradients[name])
        for name in result.gradients
    )


def test_disabling_rng_replay_changes_dropout_gradients() -> None:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(11)
    config = TinyGPTConfig(
        vocab_size=32,
        max_sequence_length=8,
        depth=3,
        width=16,
        heads=4,
        dropout=0.4,
    )
    reference = TinyGPT(config)
    scheduled = copy.deepcopy(reference)
    tokens = torch.randint(config.vocab_size, (2, 8))
    targets = torch.randint(config.vocab_size, (2, 8))
    _, expected_gradients = baseline(reference, tokens, targets, seed=29)
    chain = BlockChain.from_model(scheduled)
    schedule = make_revolve_schedule(chain.profile(tokens, n_reps=1), budget=1)
    torch.manual_seed(29)
    result = run_scheduled_backward(
        chain,
        schedule,
        (tokens, targets),
        language_model_loss,
        replay_rng=False,
    )
    assert any(
        not torch.equal(result.gradients[name], expected_gradients[name])
        for name in result.gradients
    )


@pytest.mark.parametrize("scheduler", ["none", "all", "uniform", "selective"])
@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_heuristic_gradients_are_bitwise_equal(scheduler: str, dropout: float) -> None:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(31)
    config = TinyGPTConfig(
        vocab_size=32,
        max_sequence_length=8,
        depth=3,
        width=16,
        heads=4,
        dropout=dropout,
    )
    reference = TinyGPT(config)
    scheduled = copy.deepcopy(reference)
    tokens = torch.randint(config.vocab_size, (2, 8))
    targets = torch.randint(config.vocab_size, (2, 8))
    expected_loss, expected_gradients = baseline(reference, tokens, targets, seed=37)
    granularity = "fine" if scheduler == "selective" else "coarse"
    chain = BlockChain.from_model(scheduled, granularity=granularity)
    chain_profile = chain.profile(tokens, n_reps=1)
    makers = {
        "none": lambda: make_none_schedule(chain_profile),
        "all": lambda: make_all_schedule(chain_profile),
        "uniform": lambda: make_uniform_schedule(chain_profile, k=2),
        "selective": lambda: make_selective_schedule(chain_profile),
    }
    torch.manual_seed(37)
    result = run_scheduled_backward(
        chain,
        makers[scheduler](),
        (tokens, targets),
        language_model_loss,
    )
    assert torch.equal(result.loss, expected_loss)
    assert result.gradients.keys() == expected_gradients.keys()
    assert all(
        torch.equal(result.gradients[name], expected_gradients[name])
        for name in result.gradients
    )


def test_action_observer_sees_every_action() -> None:
    config = TinyGPTConfig(
        vocab_size=16,
        max_sequence_length=4,
        depth=2,
        width=8,
        heads=2,
    )
    model = TinyGPT(config)
    tokens = torch.randint(config.vocab_size, (1, 4))
    targets = torch.randint(config.vocab_size, (1, 4))
    chain = BlockChain.from_model(model)
    schedule = make_none_schedule(chain.profile(tokens, n_reps=1))
    observed = []
    run_scheduled_backward(
        chain,
        schedule,
        (tokens, targets),
        language_model_loss,
        action_observer=observed.append,
    )
    assert observed == list(schedule.actions)
