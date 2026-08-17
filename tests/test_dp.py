import random

import pytest
import torch

from torchrevolve.chain import ChainProfile, UnitProfile
from torchrevolve.dp import make_schedule, minimum_recompute_cost
from torchrevolve.revolve import minimum_recomputations


def profile(costs: list[float], sizes: list[int]) -> ChainProfile:
    units = tuple(
        UnitProfile(str(index), cost, size, 0)
        for index, (cost, size) in enumerate(zip(costs, sizes, strict=True))
    )
    return ChainProfile(units, 1, 1, torch.float32, "fine")


@pytest.mark.parametrize("n_units", range(1, 33))
@pytest.mark.parametrize("snapshots", range(1, 9))
def test_uniform_dp_reduces_to_revolve(n_units: int, snapshots: int) -> None:
    chain = profile([1.0] * n_units, [16] * n_units)
    schedule = make_schedule(chain, budget=16 * snapshots)
    report = schedule.validate(chain)
    expected = minimum_recomputations(n_units, snapshots)
    assert report.legal, report.violations
    assert schedule.predicted().recompute_cost == expected
    assert minimum_recompute_cost(chain, 16 * snapshots) == expected


def test_dp_uses_heterogeneous_costs() -> None:
    chain = profile([1.0, 10.0, 1.0, 1.0], [10, 10, 10, 10])
    schedule = make_schedule(chain, budget=20)
    assert schedule.validate(chain).legal
    assert schedule.predicted().recompute_cost == 13.0
    assert schedule.actions[1].kind == "forward_store"
    assert schedule.actions[1].unit == 1


def test_fuzzed_heterogeneous_schedules_are_legal() -> None:
    randomizer = random.Random(17)
    for _ in range(100):
        n_units = randomizer.randint(1, 12)
        costs = [float(randomizer.randint(1, 20)) for _ in range(n_units)]
        sizes = [randomizer.randint(1, 8) * 8 for _ in range(n_units)]
        chain = profile(costs, sizes)
        budget = sizes[0] + randomizer.randint(0, sum(sizes[1:]))
        schedule = make_schedule(chain, budget=budget)
        report = schedule.validate(chain)
        assert report.legal, report.violations
        assert report.peak_bytes <= budget


def test_budget_must_hold_initial_state() -> None:
    chain = profile([1.0, 1.0], [32, 16])
    with pytest.raises(ValueError):
        make_schedule(chain, budget=31)
    with pytest.raises(ValueError):
        minimum_recompute_cost(chain, 31)
