import pytest

from case_studies.cs1_depth_sweep.run import evaluate_p3, evaluate_p4


def record(
    scheduler: str,
    *,
    depth: int,
    sequence_length: int,
    fraction: float,
    seconds: float,
) -> dict[str, object]:
    return {
        "scheduler": scheduler,
        "depth": depth,
        "sequence_length": sequence_length,
        "budget_fraction": fraction,
        "median_seconds": seconds,
    }


def test_p3_uses_depth_64_quarter_budget() -> None:
    records = [
        record("uniform", depth=64, sequence_length=256, fraction=0.25, seconds=1.0),
        record("revolve", depth=64, sequence_length=256, fraction=0.25, seconds=0.85),
        record("dp", depth=64, sequence_length=256, fraction=0.25, seconds=0.9),
    ]
    result = evaluate_p3(records)
    assert result["status"] == "passed"
    assert result["improvement"] == pytest.approx(0.15)


def test_p4_compares_exact_matched_cells() -> None:
    records = [
        record("revolve", depth=4, sequence_length=1024, fraction=0.25, seconds=1.0),
        record("dp", depth=4, sequence_length=1024, fraction=0.25, seconds=0.9),
        record("revolve", depth=4, sequence_length=2048, fraction=0.25, seconds=2.0),
        record("dp", depth=4, sequence_length=2048, fraction=0.25, seconds=2.1),
    ]
    result = evaluate_p4(records)
    assert result["status"] == "passed"
    assert result["best_time_improvement"] == pytest.approx(0.1)
    assert not result["pivot"]
