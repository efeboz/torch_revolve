"""Schedule actions, cost estimates, and legality simulation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from torchrevolve.chain import ChainProfile

ActionKind = Literal[
    "forward_store",
    "forward_drop",
    "forward_keep",
    "restore",
    "backward",
]


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    unit: int

    def __post_init__(self) -> None:
        if self.unit < 0:
            raise ValueError("action unit must be non-negative")


@dataclass(frozen=True)
class CostEstimate:
    peak_bytes: int
    recompute_cost: float
    recomputations: int


@dataclass(frozen=True)
class LegalityReport:
    legal: bool
    peak_snapshots: int
    peak_bytes: int
    recomputations: int
    violations: tuple[str, ...]

    def require_valid(self) -> None:
        if not self.legal:
            raise ValueError("; ".join(self.violations))


@dataclass(frozen=True)
class Schedule:
    actions: tuple[Action, ...]
    meta: Mapping[str, object]

    def validate(self, chain: ChainProfile) -> LegalityReport:
        n_units = len(chain.units)
        if n_units == 0:
            return LegalityReport(False, 0, 0, 0, ("chain is empty",))

        snapshots = {0}
        kept_tapes: set[int] = set()
        snapshot_bytes = self._state_bytes(chain, 0)
        peak_snapshots = 1
        peak_bytes = snapshot_bytes
        position = 0
        live_tape: int | None = None
        backward_expected = n_units - 1
        forward_count = 0
        violations: list[str] = []

        for action_index, action in enumerate(self.actions):
            unit = action.unit
            if unit >= n_units:
                violations.append(
                    f"action {action_index}: unit {unit} is outside the chain"
                )
                continue
            if action.kind.startswith("forward"):
                if position != unit:
                    violations.append(
                        f"action {action_index}: forward {unit} requires state {unit}, "
                        f"current state is {position}"
                    )
                position = unit + 1
                live_tape = unit
                forward_count += 1
                if action.kind == "forward_store" and unit + 1 < n_units:
                    state = unit + 1
                    if state not in snapshots:
                        snapshots.add(state)
                        snapshot_bytes += self._state_bytes(chain, state)
                elif action.kind == "forward_keep":
                    kept_tapes.add(unit)
            elif action.kind == "restore":
                restored_from_tape = unit > 0 and unit - 1 in kept_tapes
                if unit not in snapshots and not restored_from_tape:
                    violations.append(
                        f"action {action_index}: state {unit} is not retained"
                    )
                position = unit
                live_tape = None
            elif action.kind == "backward":
                if unit != backward_expected:
                    violations.append(
                        f"action {action_index}: backward {unit} expected {backward_expected}"
                    )
                has_tape = live_tape == unit or unit in kept_tapes
                if position != unit + 1 and unit not in kept_tapes:
                    has_tape = False
                if not has_tape:
                    violations.append(
                        f"action {action_index}: backward {unit} has no matching forward tape"
                    )
                backward_expected -= 1
                position = unit
                live_tape = None
                kept_tapes.discard(unit)
                if unit in snapshots:
                    snapshots.remove(unit)
                    snapshot_bytes -= self._state_bytes(chain, unit)
            peak_snapshots = max(peak_snapshots, len(snapshots))
            tape_bytes = sum(chain.units[item].activation_bytes for item in kept_tapes)
            live_bytes = 0
            if live_tape is not None and live_tape not in kept_tapes:
                live_bytes = chain.units[live_tape].activation_bytes
            peak_bytes = max(
                peak_bytes,
                snapshot_bytes + tape_bytes + live_bytes,
            )

        if backward_expected != -1:
            violations.append(
                f"schedule omitted {backward_expected + 1} backward actions"
            )
        recomputations = max(0, forward_count - n_units)
        snapshot_budget = self.meta.get("snapshot_budget")
        if isinstance(snapshot_budget, int) and peak_snapshots > snapshot_budget:
            violations.append(
                f"peak snapshots {peak_snapshots} exceeds budget {snapshot_budget}"
            )
        byte_budget = self.meta.get("byte_budget")
        if isinstance(byte_budget, int) and peak_bytes > byte_budget:
            violations.append(f"peak bytes {peak_bytes} exceeds budget {byte_budget}")
        return LegalityReport(
            legal=not violations,
            peak_snapshots=peak_snapshots,
            peak_bytes=peak_bytes,
            recomputations=recomputations,
            violations=tuple(violations),
        )

    def predicted(self) -> CostEstimate:
        return CostEstimate(
            peak_bytes=int(self.meta.get("predicted_peak_bytes", 0)),
            recompute_cost=float(self.meta.get("recompute_cost", 0.0)),
            recomputations=int(self.meta.get("recomputations", 0)),
        )

    @staticmethod
    def _state_bytes(chain: ChainProfile, state: int) -> int:
        return chain.units[state].effective_state_bytes


def forward_store(unit: int) -> Action:
    return Action("forward_store", unit)


def forward_drop(unit: int) -> Action:
    return Action("forward_drop", unit)


def forward_keep(unit: int) -> Action:
    return Action("forward_keep", unit)


def restore(unit: int) -> Action:
    return Action("restore", unit)


def backward(unit: int) -> Action:
    return Action("backward", unit)
