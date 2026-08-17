"""Baseline checkpoint schedules for transformer chains."""

from __future__ import annotations

from torchrevolve.chain import ChainProfile
from torchrevolve.schedules import (
    Action,
    Schedule,
    backward,
    forward_drop,
    forward_keep,
    forward_store,
    restore,
)


def make_schedule(profile: ChainProfile, *, budget: str | int) -> Schedule:
    if budget == "none":
        return make_none_schedule(profile)
    if budget == "all":
        return make_all_schedule(profile)
    if budget == "selective":
        return make_selective_schedule(profile)
    if isinstance(budget, int):
        return make_uniform_schedule(profile, k=budget)
    raise ValueError("budget must be 'none', 'all', 'selective', or a positive k")


def make_none_schedule(profile: ChainProfile) -> Schedule:
    actions = [forward_keep(unit) for unit in range(len(profile.units))]
    actions.extend(backward(unit) for unit in range(len(profile.units) - 1, -1, -1))
    return _finish(profile, actions, scheduler="none")


def make_all_schedule(profile: ChainProfile) -> Schedule:
    n_units = len(profile.units)
    actions: list[Action] = []
    for unit in range(n_units):
        actions.append(
            forward_store(unit) if unit + 1 < n_units else forward_drop(unit)
        )
    for unit in range(n_units - 1, -1, -1):
        actions.extend((restore(unit), forward_drop(unit), backward(unit)))
    return _finish(profile, actions, scheduler="all", snapshot_budget=n_units)


def make_uniform_schedule(profile: ChainProfile, *, k: int) -> Schedule:
    if k <= 0:
        raise ValueError("k must be positive")
    n_units = len(profile.units)
    actions: list[Action] = []
    for unit in range(n_units):
        boundary = (unit + 1) % k == 0 and unit + 1 < n_units
        actions.append(forward_store(unit) if boundary else forward_drop(unit))
    segment_starts = range(0, n_units, k)
    for start in reversed(tuple(segment_starts)):
        end = min(start + k, n_units)
        actions.append(restore(start))
        actions.extend(forward_keep(unit) for unit in range(start, end))
        actions.extend(backward(unit) for unit in range(end - 1, start - 1, -1))
    snapshots = 1 + (n_units - 1) // k
    return _finish(
        profile,
        actions,
        scheduler="uniform",
        snapshot_budget=snapshots,
        k=k,
    )


def make_selective_schedule(profile: ChainProfile) -> Schedule:
    kinds = [unit.kind for unit in profile.units]
    if not kinds or any(kind not in {"attention", "mlp"} for kind in kinds):
        raise ValueError(
            "selective scheduling requires a fine-grained attention/MLP chain"
        )
    actions: list[Action] = []
    for unit, kind in enumerate(kinds):
        actions.append(forward_keep(unit) if kind == "mlp" else forward_drop(unit))
    for unit in range(len(kinds) - 1, -1, -1):
        if kinds[unit] == "mlp":
            actions.append(backward(unit))
        else:
            actions.extend((restore(unit), forward_drop(unit), backward(unit)))
    return _finish(profile, actions, scheduler="selective")


def _finish(
    profile: ChainProfile,
    actions: list[Action],
    *,
    scheduler: str,
    **metadata: object,
) -> Schedule:
    provisional = Schedule(tuple(actions), {"scheduler": scheduler, **metadata})
    report = provisional.validate(profile)
    report.require_valid()
    forward_counts = [0] * len(profile.units)
    for action in actions:
        if action.kind.startswith("forward"):
            forward_counts[action.unit] += 1
    recompute_cost = sum(
        max(0, count - 1) * unit.forward_seconds
        for count, unit in zip(forward_counts, profile.units, strict=True)
    )
    complete_metadata = dict(provisional.meta)
    complete_metadata.update(
        predicted_peak_bytes=report.peak_bytes,
        recomputations=report.recomputations,
        recompute_cost=recompute_cost,
    )
    return Schedule(provisional.actions, complete_metadata)
