import pytest
import torch

from torchrevolve.chain import ChainProfile, UnitProfile
from torchrevolve.selection import select_schedule


def profile(kinds: list[str]) -> ChainProfile:
    units = tuple(UnitProfile(str(i), 1.0, 16, 0, kind) for i, kind in enumerate(kinds))
    return ChainProfile(units, 1, 1, torch.float32, "fine")


def test_selects_feasible_revolve_and_uniform_schedules() -> None:
    chain = profile(["block"] * 8)
    revolve = select_schedule(chain, "revolve", byte_budget=48)
    uniform = select_schedule(chain, "uniform", byte_budget=96)
    assert revolve.schedule.predicted().peak_bytes <= 48
    assert uniform.schedule.predicted().peak_bytes <= 96


def test_selective_selection() -> None:
    chain = profile(["attention", "mlp"] * 2)
    required = sum(unit.activation_bytes for unit in chain.units)
    selection = select_schedule(chain, "selective", byte_budget=required)
    assert selection.schedule.validate(chain).legal


def test_infeasible_schedule_raises() -> None:
    chain = profile(["block"] * 4)
    with pytest.raises(MemoryError):
        select_schedule(chain, "none", byte_budget=1)
