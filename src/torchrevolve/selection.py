"""Matched-byte-budget selection across scheduler families."""

from __future__ import annotations

from dataclasses import dataclass

from torchrevolve.chain import ChainProfile
from torchrevolve.dp import make_schedule as make_dp_schedule
from torchrevolve.heuristics import (
    make_all_schedule,
    make_none_schedule,
    make_selective_schedule,
    make_uniform_schedule,
)
from torchrevolve.revolve import make_schedule as make_revolve_schedule
from torchrevolve.schedules import Schedule

SCHEDULERS = ("none", "all", "uniform", "revolve", "dp", "selective")


@dataclass(frozen=True)
class ScheduleSelection:
    schedule: Schedule
    scheduler: str
    byte_budget: int
    parameter: int | str


def select_schedule(
    profile: ChainProfile,
    scheduler: str,
    *,
    byte_budget: int,
) -> ScheduleSelection:
    if scheduler not in SCHEDULERS:
        raise ValueError(f"unknown scheduler: {scheduler}")
    if byte_budget <= 0:
        raise ValueError("byte_budget must be positive")
    if scheduler == "dp":
        schedule = make_dp_schedule(profile, budget=byte_budget)
        return _checked(schedule, scheduler, byte_budget, byte_budget)
    if scheduler == "revolve":
        candidates = [
            make_revolve_schedule(profile, budget=slots)
            for slots in range(1, len(profile.units) + 1)
        ]
        return _best_feasible(candidates, scheduler, byte_budget, "snapshot_budget")
    if scheduler == "uniform":
        candidates = [
            make_uniform_schedule(profile, k=k)
            for k in range(1, len(profile.units) + 1)
        ]
        return _best_feasible(candidates, scheduler, byte_budget, "k")
    makers = {
        "none": make_none_schedule,
        "all": make_all_schedule,
        "selective": make_selective_schedule,
    }
    schedule = makers[scheduler](profile)
    return _checked(schedule, scheduler, byte_budget, scheduler)


def _best_feasible(
    candidates: list[Schedule],
    scheduler: str,
    byte_budget: int,
    parameter_name: str,
) -> ScheduleSelection:
    feasible = [
        schedule
        for schedule in candidates
        if schedule.predicted().peak_bytes <= byte_budget
    ]
    if not feasible:
        raise MemoryError(f"{scheduler} has no schedule within {byte_budget} bytes")
    schedule = min(
        feasible,
        key=lambda item: (
            item.predicted().recompute_cost,
            item.predicted().peak_bytes,
        ),
    )
    parameter = schedule.meta.get(parameter_name, "auto")
    return ScheduleSelection(schedule, scheduler, byte_budget, parameter)


def _checked(
    schedule: Schedule,
    scheduler: str,
    byte_budget: int,
    parameter: int | str,
) -> ScheduleSelection:
    if schedule.predicted().peak_bytes > byte_budget:
        raise MemoryError(
            f"{scheduler} requires {schedule.predicted().peak_bytes} bytes, "
            f"budget is {byte_budget}"
        )
    return ScheduleSelection(schedule, scheduler, byte_budget, parameter)
