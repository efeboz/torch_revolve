"""Optimal dynamic program for checkpointing heterogeneous chains."""

from __future__ import annotations

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


def minimum_recompute_cost(profile: ChainProfile, byte_budget: int) -> float:
    root_bytes = profile.units[0].effective_state_bytes
    if byte_budget < root_bytes:
        raise ValueError("byte budget cannot hold the chain input snapshot")
    cost, _, _ = _solve(profile, byte_budget)
    result = cost(0, len(profile.units), byte_budget - root_bytes)
    if result == float("inf"):
        raise MemoryError("byte budget cannot execute the chain")
    return result


def make_schedule(profile: ChainProfile, *, budget: int) -> Schedule:
    root_bytes = profile.units[0].effective_state_bytes
    if budget < root_bytes:
        raise ValueError("byte budget cannot hold the chain input snapshot")
    cost, split_at, state_bytes = _solve(profile, budget)
    free_bytes = budget - root_bytes
    optimal_cost = cost(0, len(profile.units), free_bytes)
    if optimal_cost == float("inf"):
        raise MemoryError("byte budget cannot execute the chain")
    actions: list[Action] = []
    _emit_interval(
        actions,
        start=0,
        end=len(profile.units),
        free_bytes=free_bytes,
        split_at=split_at,
        state_bytes=state_bytes,
    )
    schedule = Schedule(
        tuple(actions),
        {
            "scheduler": "dp",
            "byte_budget": budget,
            "recompute_cost": optimal_cost,
        },
    )
    report = schedule.validate(profile)
    report.require_valid()
    metadata = dict(schedule.meta)
    metadata.update(
        predicted_peak_bytes=report.peak_bytes,
        recomputations=report.recomputations,
    )
    return Schedule(schedule.actions, metadata)


def _solve(profile: ChainProfile, byte_budget: int):
    if byte_budget <= 0:
        raise ValueError("byte budget must be positive")
    n_units = len(profile.units)
    forward_costs = tuple(unit.forward_seconds for unit in profile.units)
    activation_bytes = tuple(unit.activation_bytes for unit in profile.units)
    state_bytes = tuple(unit.effective_state_bytes for unit in profile.units)
    prefix_costs = [0.0]
    for cost_value in forward_costs:
        prefix_costs.append(prefix_costs[-1] + cost_value)

    def interval_cost(start: int, end: int) -> float:
        return prefix_costs[end] - prefix_costs[start]

    decisions: dict[tuple[int, int, int], int | None] = {}

    @cache
    def cost(start: int, end: int, free_bytes: int) -> float:
        if start < 0 or end > n_units or start >= end:
            raise ValueError("invalid chain interval")
        if free_bytes < 0:
            return float("inf")
        if end - start == 1:
            decisions[(start, end, free_bytes)] = None
            return 0.0 if activation_bytes[start] <= free_bytes else float("inf")
        best_cost = (
            sum(interval_cost(start, unit + 1) for unit in range(start, end - 1))
            if max(activation_bytes[start:end]) <= free_bytes
            else float("inf")
        )
        best_split = None
        for split in range(start + 1, end):
            required = state_bytes[split]
            if required > free_bytes:
                continue
            if max(activation_bytes[start:split]) > free_bytes:
                continue
            if activation_bytes[split - 1] + required > free_bytes:
                continue
            candidate = (
                interval_cost(start, split)
                + cost(start, split, free_bytes)
                + cost(split, end, free_bytes - required)
            )
            if candidate < best_cost:
                best_cost = candidate
                best_split = split
        decisions[(start, end, free_bytes)] = best_split
        return best_cost

    def split_at(start: int, end: int, free_bytes: int) -> int | None:
        cost(start, end, free_bytes)
        return decisions[(start, end, free_bytes)]

    return cost, split_at, state_bytes


def _emit_interval(
    actions: list[Action],
    *,
    start: int,
    end: int,
    free_bytes: int,
    split_at,
    state_bytes: tuple[int, ...],
) -> None:
    if end - start == 1:
        actions.extend((forward_drop(start), backward(start)))
        return
    split = split_at(start, end, free_bytes)
    if split is None:
        actions.extend(forward_drop(unit) for unit in range(start, end))
        actions.append(backward(end - 1))
        for unit in range(end - 2, start - 1, -1):
            actions.append(restore(start))
            actions.extend(forward_drop(step) for step in range(start, unit + 1))
            actions.append(backward(unit))
        return
    actions.extend(forward_drop(unit) for unit in range(start, split - 1))
    actions.append(forward_store(split - 1))
    _emit_interval(
        actions,
        start=split,
        end=end,
        free_bytes=free_bytes - state_bytes[split],
        split_at=split_at,
        state_bytes=state_bytes,
    )
    actions.append(restore(start))
    _emit_interval(
        actions,
        start=start,
        end=split,
        free_bytes=free_bytes,
        split_at=split_at,
        state_bytes=state_bytes,
    )
