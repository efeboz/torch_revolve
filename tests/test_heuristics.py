import math

import pytest
import torch

from torchrevolve.chain import ChainProfile, UnitProfile
from torchrevolve.heuristics import (
    make_all_schedule,
    make_none_schedule,
    make_selective_schedule,
    make_uniform_schedule,
)


def profile(kinds: list[str]) -> ChainProfile:
    units = tuple(
        UnitProfile(str(index), float(index + 1), 16, 0, kind)
        for index, kind in enumerate(kinds)
    )
    granularity = "fine" if "attention" in kinds else "coarse"
    return ChainProfile(units, 1, 1, torch.float32, granularity)


@pytest.mark.parametrize("n_units", range(1, 33))
def test_none_and_all_are_legal(n_units: int) -> None:
    chain = profile(["block"] * n_units)
    none = make_none_schedule(chain)
    all_blocks = make_all_schedule(chain)
    assert none.validate(chain).legal
    assert none.predicted().recomputations == 0
    assert all_blocks.validate(chain).legal
    assert all_blocks.predicted().recomputations == n_units


@pytest.mark.parametrize("n_units", range(1, 33))
@pytest.mark.parametrize("k", range(1, 9))
def test_uniform_schedule_is_legal(n_units: int, k: int) -> None:
    chain = profile(["block"] * n_units)
    schedule = make_uniform_schedule(chain, k=k)
    report = schedule.validate(chain)
    assert report.legal, report.violations
    assert report.peak_snapshots <= math.ceil(n_units / k)
    assert report.recomputations == n_units


def test_selective_recomputes_only_attention() -> None:
    chain = profile(["attention", "mlp"] * 4)
    schedule = make_selective_schedule(chain)
    report = schedule.validate(chain)
    assert report.legal, report.violations
    assert report.recomputations == 4
    expected_cost = sum(
        unit.forward_seconds for unit in chain.units if unit.kind == "attention"
    )
    assert schedule.predicted().recompute_cost == expected_cost


def test_selective_requires_fine_granularity() -> None:
    with pytest.raises(ValueError):
        make_selective_schedule(profile(["block"] * 3))


def test_snapshot_and_tape_bytes_are_modeled_separately() -> None:
    units = tuple(UnitProfile(str(i), 1.0, 100, 0, "block", 10) for i in range(4))
    chain = ChainProfile(units, 1, 1, torch.float32, "coarse")
    assert make_none_schedule(chain).predicted().peak_bytes == 410
    assert make_all_schedule(chain).predicted().peak_bytes == 140
    assert make_uniform_schedule(chain, k=2).predicted().peak_bytes == 220
