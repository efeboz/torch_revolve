"""Offline binomial checkpoint schedules for homogeneous chains."""

from __future__ import annotations

import math
from functools import cache

from torchrevolve.chain import ChainProfile
from torchrevolve.schedules import (
    Action,
    Schedule,
    backward,
    forward_drop,
    forward_store,
    restore,
)


def binomial_capacity(snapshots: int, repetitions: int) -> int:
    if snapshots <= 0:
        raise ValueError("snapshots must be positive")
    if repetitions < 0:
        raise ValueError("repetitions cannot be negative")
    return math.comb(snapshots + repetitions, snapshots)


def minimum_repetition_number(n_steps: int, snapshots: int) -> int:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if snapshots <= 0:
        raise ValueError("snapshots must be positive")
    repetitions = 0
    while binomial_capacity(snapshots, repetitions) < n_steps:
        repetitions += 1
    return repetitions


def combinatorial_recomputations(n_steps: int, snapshots: int) -> int:
    repetitions = minimum_repetition_number(n_steps, snapshots)
    return repetitions * n_steps - math.comb(
        snapshots + repetitions,
        snapshots + 1,
    )


@cache
def minimum_recomputations(n_steps: int, snapshots: int) -> int:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if snapshots <= 0:
        raise ValueError("snapshots must be positive")
    if n_steps == 1:
        return 0
    if snapshots == 1:
        return n_steps * (n_steps - 1) // 2
    return min(
        split
        + minimum_recomputations(split, snapshots)
        + minimum_recomputations(n_steps - split, snapshots - 1)
        for split in range(1, n_steps)
    )


def optimal_split(n_steps: int, snapshots: int) -> int:
    if n_steps <= 1 or snapshots <= 1:
        raise ValueError("a split requires at least two steps and two snapshots")
    return min(
        range(1, n_steps),
        key=lambda split: (
            split
            + minimum_recomputations(split, snapshots)
            + minimum_recomputations(n_steps - split, snapshots - 1),
            split,
        ),
    )


def make_schedule(profile: ChainProfile, *, budget: int) -> Schedule:
    if budget <= 0:
        raise ValueError("snapshot budget must be positive")
    actions: list[Action] = []
    _emit_interval(actions, start=0, length=len(profile.units), snapshots=budget)
    recomputations = minimum_recomputations(len(profile.units), budget)
    schedule = Schedule(
        actions=tuple(actions),
        meta={
            "scheduler": "revolve",
            "snapshot_budget": budget,
            "recomputations": recomputations,
            "recompute_cost": _recompute_cost(actions, profile),
        },
    )
    report = schedule.validate(profile)
    report.require_valid()
    metadata = dict(schedule.meta)
    metadata["predicted_peak_bytes"] = report.peak_bytes
    return Schedule(schedule.actions, metadata)


def _emit_interval(
    actions: list[Action],
    *,
    start: int,
    length: int,
    snapshots: int,
) -> None:
    if length == 1:
        actions.extend((forward_drop(start), backward(start)))
        return
    if snapshots == 1:
        end = start + length
        actions.extend(forward_drop(unit) for unit in range(start, end))
        actions.append(backward(end - 1))
        for unit in range(end - 2, start - 1, -1):
            actions.append(restore(start))
            actions.extend(forward_drop(step) for step in range(start, unit + 1))
            actions.append(backward(unit))
        return

    split = optimal_split(length, snapshots)
    split_state = start + split
    actions.extend(forward_drop(unit) for unit in range(start, split_state - 1))
    actions.append(forward_store(split_state - 1))
    _emit_interval(
        actions,
        start=split_state,
        length=length - split,
        snapshots=snapshots - 1,
    )
    actions.append(restore(start))
    _emit_interval(actions, start=start, length=split, snapshots=snapshots)


def _recompute_cost(actions: list[Action], profile: ChainProfile) -> float:
    counts = [0] * len(profile.units)
    for action in actions:
        if action.kind.startswith("forward"):
            counts[action.unit] += 1
    return sum(
        max(0, count - 1) * unit.forward_seconds
        for count, unit in zip(counts, profile.units, strict=True)
    )
