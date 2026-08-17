"""Plotly schedule diagrams and standalone verification reports."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from pathlib import Path

import plotly.graph_objects as go

from torchrevolve.schedules import Schedule

_ACTION_STYLE = {
    "forward_store": ("square", "#2563eb", "store"),
    "forward_drop": ("circle", "#94a3b8", "forward"),
    "forward_keep": ("diamond", "#16a34a", "retain"),
    "restore": ("triangle-left", "#d97706", "restore"),
    "backward": ("x", "#dc2626", "backward"),
}


def schedule_figure(schedule: Schedule, *, title: str | None = None) -> go.Figure:
    figure = go.Figure()
    for kind, (symbol, color, label) in _ACTION_STYLE.items():
        selected = [
            (index, action.unit)
            for index, action in enumerate(schedule.actions)
            if action.kind == kind
        ]
        if not selected:
            continue
        x_values, y_values = zip(*selected, strict=True)
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name=label,
                marker={"symbol": symbol, "size": 9, "color": color},
                hovertemplate="action %{x}<br>unit %{y}<extra>" + label + "</extra>",
            )
        )
    figure.update_layout(
        title=title or f"{schedule.meta.get('scheduler', 'schedule')} schedule",
        xaxis_title="Action index",
        yaxis_title="Chain unit",
        yaxis={"autorange": "reversed", "dtick": 1},
        template="plotly_white",
        legend={"orientation": "h"},
    )
    return figure


def pareto_figure(records: Sequence[Mapping[str, object]]) -> go.Figure:
    figure = go.Figure()
    scheduler_names = sorted({str(record["scheduler"]) for record in records})
    for scheduler in scheduler_names:
        selected = [record for record in records if record["scheduler"] == scheduler]
        selected.sort(key=lambda record: int(record["predicted_peak_bytes"]))
        figure.add_trace(
            go.Scatter(
                x=[int(record["predicted_peak_bytes"]) for record in selected],
                y=[float(record["median_seconds"]) for record in selected],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [
                        float(record["q3_seconds"]) - float(record["median_seconds"])
                        for record in selected
                    ],
                    "arrayminus": [
                        float(record["median_seconds"]) - float(record["q1_seconds"])
                        for record in selected
                    ],
                },
                mode="lines+markers",
                name=scheduler,
                customdata=[record.get("label", "") for record in selected],
                hovertemplate=(
                    "%{customdata}<br>predicted bytes %{x}<br>median %{y:.6f} s"
                    "<extra>" + scheduler + "</extra>"
                ),
            )
        )
    figure.update_layout(
        title="Activation-memory and wall-clock Pareto frontier",
        xaxis_title="Predicted peak activation bytes",
        yaxis_title="Median step time (seconds)",
        template="plotly_white",
    )
    return figure


def budget_figure(records: Sequence[Mapping[str, object]]) -> go.Figure:
    schedulers = sorted({str(record["scheduler"]) for record in records})
    depths = sorted({int(record["depth"]) for record in records})
    lengths = sorted({int(record["sequence_length"]) for record in records})
    figure = go.Figure()
    for index, scheduler in enumerate(schedulers):
        lookup = {
            (int(record["depth"]), int(record["sequence_length"])): bool(record["fits"])
            for record in records
            if record["scheduler"] == scheduler
        }
        figure.add_trace(
            go.Heatmap(
                x=lengths,
                y=depths,
                z=[
                    [int(lookup.get((depth, length), False)) for length in lengths]
                    for depth in depths
                ],
                zmin=0,
                zmax=1,
                colorscale=[[0, "#fecaca"], [1, "#86efac"]],
                showscale=False,
                visible=index == 0,
                hovertemplate="depth %{y}<br>sequence %{x}<br>fits %{z}<extra>"
                + scheduler
                + "</extra>",
            )
        )
    buttons = []
    for index, scheduler in enumerate(schedulers):
        visible = [item == index for item in range(len(schedulers))]
        buttons.append(
            {
                "label": scheduler,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"Fixed-budget fit: {scheduler}"},
                ],
            }
        )
    figure.update_layout(
        title=f"Fixed-budget fit: {schedulers[0] if schedulers else 'none'}",
        xaxis_title="Sequence length",
        yaxis_title="Depth",
        template="plotly_white",
        updatemenus=[{"buttons": buttons, "direction": "down"}],
    )
    return figure


def write_html_report(
    output: str | Path,
    *,
    records: Sequence[Mapping[str, object]],
    schedules: Mapping[str, Schedule] | None = None,
    verification: Mapping[str, str] | None = None,
    title: str = "torch-revolve report",
) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sections = [f"<h1>{html.escape(title)}</h1>"]
    include_plotly = True
    if records:
        sections.append(
            pareto_figure(records).to_html(
                full_html=False,
                include_plotlyjs=include_plotly,
            )
        )
        include_plotly = False
    if verification:
        rows = "".join(
            f"<tr><th>{html.escape(name)}</th><td>{html.escape(status)}</td></tr>"
            for name, status in verification.items()
        )
        sections.append(f"<h2>Verification</h2><table>{rows}</table>")
    for name, schedule in (schedules or {}).items():
        sections.append(
            schedule_figure(schedule, title=name).to_html(
                full_html=False,
                include_plotlyjs=include_plotly,
            )
        )
        include_plotly = False
    if include_plotly:
        sections.append("<p>No plot records were supplied.</p>")
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:system-ui;margin:2rem;max-width:1100px}"
        "table{border-collapse:collapse}th,td{padding:.4rem;border:1px solid #ccc}</style>"
        "</head><body>" + "".join(sections) + "</body></html>"
    )
    destination.write_text(document, encoding="utf-8")
    return destination


def write_budget_report(
    output: str | Path,
    *,
    records: Sequence[Mapping[str, object]],
    largest: Mapping[str, Mapping[str, object] | None],
    parity: Mapping[str, object] | None,
) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = budget_figure(records).to_html(full_html=False, include_plotlyjs=True)
    rows = []
    for scheduler, record in largest.items():
        if record is None:
            cells = (scheduler, "not trainable", "—", "—")
        else:
            cells = (
                scheduler,
                str(record["depth"]),
                str(record["sequence_length"]),
                str(record["predicted_peak_bytes"]),
            )
        rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells)
            + "</tr>"
        )
    parity_text = "not run" if parity is None else html.escape(str(dict(parity)))
    document = (
        "<!doctype html><html><head><meta charset='utf-8'><title>torch-revolve budget report</title>"
        "<style>body{font-family:system-ui;margin:2rem;max-width:1100px}"
        "table{border-collapse:collapse}th,td{padding:.4rem;border:1px solid #ccc}</style>"
        "</head><body><h1>Fixed activation-budget report</h1>"
        + figure
        + "<h2>Largest trainable configurations</h2>"
        "<table><tr><th>Scheduler</th><th>Depth</th><th>Sequence</th><th>Predicted bytes</th></tr>"
        + "".join(rows)
        + f"</table><h2>Training parity</h2><p>{parity_text}</p></body></html>"
    )
    destination.write_text(document, encoding="utf-8")
    return destination
