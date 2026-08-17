import math

import pytest
import torch

from torchrevolve.chain import ChainProfile, UnitProfile
from torchrevolve.revolve import (
    binomial_capacity,
    combinatorial_recomputations,
    make_schedule,
    minimum_recomputations,
    minimum_repetition_number,
)


def uniform_profile(n_units: int) -> ChainProfile:
    units = tuple(UnitProfile(str(i), 1.0, 16, 0) for i in range(n_units))
    return ChainProfile(units, 1, 1, torch.float32, "coarse")


def brute_force_cost(n_steps: int, snapshots: int) -> int:
    table = [[0] * (snapshots + 1) for _ in range(n_steps + 1)]
    for length in range(2, n_steps + 1):
        table[length][1] = length * (length - 1) // 2
        for slots in range(2, snapshots + 1):
            table[length][slots] = min(
                split + table[split][slots] + table[length - split][slots - 1]
                for split in range(1, length)
            )
    return table[n_steps][snapshots]


def test_binomial_capacity_boundaries() -> None:
    assert binomial_capacity(1, 0) == 1
    assert binomial_capacity(3, 2) == math.comb(5, 3)
    assert minimum_repetition_number(10, 3) == 2


@pytest.mark.parametrize("n_steps", range(1, 65))
@pytest.mark.parametrize("snapshots", range(1, 13))
def test_recompute_count_is_combinatorially_optimal(
    n_steps: int,
    snapshots: int,
) -> None:
    expected = combinatorial_recomputations(n_steps, snapshots)
    assert minimum_recomputations(n_steps, snapshots) == expected
    assert brute_force_cost(n_steps, snapshots) == expected


@pytest.mark.parametrize("n_steps", range(1, 33))
@pytest.mark.parametrize("snapshots", range(1, 9))
def test_generated_schedule_is_legal_and_matches_cost(
    n_steps: int,
    snapshots: int,
) -> None:
    profile = uniform_profile(n_steps)
    schedule = make_schedule(profile, budget=snapshots)
    report = schedule.validate(profile)
    assert report.legal, report.violations
    assert report.peak_snapshots <= snapshots
    assert report.recomputations == minimum_recomputations(n_steps, snapshots)
    assert schedule.predicted().recomputations == report.recomputations
    assert schedule.predicted().peak_bytes == report.peak_bytes


@pytest.mark.parametrize(
    ("n_steps", "snapshots"),
    [(0, 1), (1, 0), (-1, 2)],
)
def test_invalid_revolve_arguments(n_steps: int, snapshots: int) -> None:
    with pytest.raises(ValueError):
        minimum_recomputations(n_steps, snapshots)
