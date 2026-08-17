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


@pytest.mark.parametrize(
    "scheduler",
    ["none", "all", "uniform", "revolve", "dp", "selective"],
)
def test_e2_canonical_gradient_matrix(scheduler: str) -> None:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(61)
    config = TinyGPTConfig(
        vocab_size=64,
        max_sequence_length=256,
        depth=4,
        width=32,
        heads=4,
        dropout=0.2,
    )
    reference = TinyGPT(config)
    scheduled = copy.deepcopy(reference)
    tokens = torch.randint(config.vocab_size, (1, 256))
    targets = torch.randint(config.vocab_size, (1, 256))

    torch.manual_seed(67)
    expected_loss = reference(tokens, targets)
    expected_loss.backward()
    expected_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in reference.named_parameters()
    }

    granularity = "fine" if scheduler in {"dp", "selective"} else "coarse"
    chain = BlockChain.from_model(scheduled, granularity=granularity)
    profile = chain.profile(tokens, n_reps=1)
    if scheduler == "none":
        schedule = make_none_schedule(profile)
    elif scheduler == "all":
        schedule = make_all_schedule(profile)
    elif scheduler == "uniform":
        schedule = make_uniform_schedule(profile, k=2)
    elif scheduler == "revolve":
        schedule = make_revolve_schedule(profile, budget=2)
    elif scheduler == "dp":
        budget = profile.units[0].effective_state_bytes + max(
            unit.activation_bytes for unit in profile.units
        )
        schedule = make_dp_schedule(profile, budget=budget)
    else:
        schedule = make_selective_schedule(profile)

    def loss_fn(logits: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.flatten(0, 1), expected.flatten())

    torch.manual_seed(67)
    result = run_scheduled_backward(
        chain,
        schedule,
        (tokens, targets),
        loss_fn,
    )
    assert torch.equal(result.loss, expected_loss)
    assert all(
        torch.equal(result.gradients[name], expected_gradients[name])
        for name in result.gradients
    )
