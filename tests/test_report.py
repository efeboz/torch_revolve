import torch

from torchrevolve.chain import ChainProfile, UnitProfile
from torchrevolve.report import (
    budget_figure,
    pareto_figure,
    schedule_figure,
    write_budget_report,
    write_html_report,
)
from torchrevolve.revolve import make_schedule


def test_schedule_diagram_and_html_report_render(tmp_path) -> None:
    profile = ChainProfile(
        tuple(UnitProfile(str(index), 1.0, 16, 0) for index in range(4)),
        1,
        1,
        torch.float32,
        "coarse",
    )
    schedule = make_schedule(profile, budget=2)
    figure = schedule_figure(schedule)
    assert len(figure.data) >= 3
    records = [
        {
            "scheduler": "revolve",
            "predicted_peak_bytes": 32,
            "median_seconds": 0.2,
            "q1_seconds": 0.19,
            "q3_seconds": 0.22,
            "label": "budget 50%",
        },
        {
            "scheduler": "none",
            "predicted_peak_bytes": 64,
            "median_seconds": 0.1,
            "q1_seconds": 0.09,
            "q3_seconds": 0.11,
            "label": "budget 100%",
        },
    ]
    assert len(pareto_figure(records).data) == 2
    output = write_html_report(
        tmp_path / "report.html",
        records=records,
        schedules={"revolve(2)": schedule},
        verification={"P1": "passed"},
    )
    contents = output.read_text(encoding="utf-8")
    assert "<!doctype html>" in contents
    assert "revolve(2)" in contents
    assert "P1" in contents
    assert "plotly" in contents.lower()


def test_budget_report_renders(tmp_path) -> None:
    records = [
        {
            "scheduler": scheduler,
            "depth": depth,
            "sequence_length": 8,
            "fits": depth == 2,
            "predicted_peak_bytes": 64 if depth == 2 else None,
        }
        for scheduler in ("none", "dp")
        for depth in (2, 4)
    ]
    largest = {
        "none": records[0],
        "dp": records[2],
    }
    assert len(budget_figure(records).data) == 2
    output = write_budget_report(
        tmp_path / "budget.html",
        records=records,
        largest=largest,
        parity={"losses_equal": True},
    )
    contents = output.read_text(encoding="utf-8")
    assert "Largest trainable configurations" in contents
    assert "losses_equal" in contents
