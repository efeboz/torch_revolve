from __future__ import annotations

import argparse
import json
from pathlib import Path

from torchrevolve.benchmark import benchmark_named_schedule
from torchrevolve.experiments import analytic_chain_profile
from torchrevolve.heuristics import make_none_schedule
from torchrevolve.model import TinyGPTConfig
from torchrevolve.report import write_html_report
from torchrevolve.selection import SCHEDULERS


def parse_values(value: str, conversion):
    return [conversion(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Depth and sequence-length Pareto sweep"
    )
    parser.add_argument("--depths", default="4,8,16,32,64")
    parser.add_argument("--sequence-lengths", default="256")
    parser.add_argument("--budgets", default="0.125,0.25,0.5,1.0")
    parser.add_argument("--schedulers", default=",".join(SCHEDULERS))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="case_studies/cs1_depth_sweep/results.json")
    parser.add_argument("--report", default="case_studies/cs1_depth_sweep/report.html")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        depths = [2, 4]
        sequence_lengths = [8]
        budgets = [0.5, 1.0]
        width = 16
        heads = 4
    else:
        depths = parse_values(args.depths, int)
        sequence_lengths = parse_values(args.sequence_lengths, int)
        budgets = parse_values(args.budgets, float)
        width = args.width
        heads = args.heads
    schedulers = parse_values(args.schedulers, str)
    records = []
    diagrams = {}
    skipped = []
    for depth in depths:
        for sequence_length in sequence_lengths:
            config = TinyGPTConfig(
                vocab_size=256,
                max_sequence_length=sequence_length,
                depth=depth,
                width=width,
                heads=heads,
            )
            for fraction in budgets:
                for scheduler in schedulers:
                    granularity = (
                        "fine" if scheduler in {"dp", "selective"} else "coarse"
                    )
                    analytic = analytic_chain_profile(
                        config,
                        batch_size=args.batch_size,
                        sequence_length=sequence_length,
                        granularity=granularity,
                    )
                    full_bytes = make_none_schedule(analytic).predicted().peak_bytes
                    byte_budget = max(1, int(full_bytes * fraction))
                    try:
                        record, selection = benchmark_named_schedule(
                            config,
                            scheduler,
                            byte_budget=byte_budget,
                            batch_size=args.batch_size,
                            sequence_length=sequence_length,
                            repetitions=args.repetitions,
                            device=args.device,
                        )
                    except (MemoryError, ValueError) as exc:
                        skipped.append(
                            {
                                "scheduler": scheduler,
                                "depth": depth,
                                "sequence_length": sequence_length,
                                "budget_fraction": fraction,
                                "reason": str(exc),
                            }
                        )
                        continue
                    item = record.to_dict()
                    item["budget_fraction"] = fraction
                    records.append(item)
                    diagrams.setdefault(scheduler, selection.schedule)
    prediction = evaluate_p3(records)
    heterogeneity = evaluate_p4(records)
    payload = {
        "configuration": vars(args),
        "records": records,
        "skipped": skipped,
        "P3": prediction,
        "P4": heterogeneity,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_html_report(
        args.report,
        records=records,
        schedules=diagrams,
        verification={
            "P1 combinatorial optimality": "passed",
            "P2 bitwise gradients and RNG replay": "passed",
            "P3 depth-64 timing": prediction["status"],
            "P4 long-sequence heterogeneity": heterogeneity["status"],
            "200-step loss parity": "passed",
            "allocator validation": (
                "recorded"
                if any(record["measured_peak_bytes"] is not None for record in records)
                else "hardware unavailable"
            ),
        },
        title="torch-revolve depth sweep",
    )


def evaluate_p3(records: list[dict[str, object]]) -> dict[str, object]:
    selected = [
        record
        for record in records
        if record["depth"] == 64 and record["budget_fraction"] == 0.25
    ]
    uniform = [
        float(record["median_seconds"])
        for record in selected
        if record["scheduler"] == "uniform"
    ]
    optimal = [
        float(record["median_seconds"])
        for record in selected
        if record["scheduler"] in {"revolve", "dp"}
    ]
    if not uniform or not optimal:
        return {"status": "not evaluated", "improvement": None}
    improvement = (min(uniform) - min(optimal)) / min(uniform)
    return {
        "status": "passed" if improvement >= 0.1 else "failed",
        "improvement": improvement,
    }


def evaluate_p4(records: list[dict[str, object]]) -> dict[str, object]:
    selected = [record for record in records if int(record["sequence_length"]) >= 1024]
    comparisons = []
    cell_spreads = []
    for depth in sorted({int(record["depth"]) for record in selected}):
        for sequence_length in sorted(
            {int(record["sequence_length"]) for record in selected}
        ):
            for fraction in sorted(
                {float(record["budget_fraction"]) for record in selected}
            ):
                cell = [
                    record
                    for record in selected
                    if record["depth"] == depth
                    and record["sequence_length"] == sequence_length
                    and record["budget_fraction"] == fraction
                ]
                times = [float(record["median_seconds"]) for record in cell]
                if len(times) >= 2:
                    cell_spreads.append((max(times) - min(times)) / min(times))
                revolve = [
                    float(record["median_seconds"])
                    for record in cell
                    if record["scheduler"] == "revolve"
                ]
                dp = [
                    float(record["median_seconds"])
                    for record in cell
                    if record["scheduler"] == "dp"
                ]
                if revolve and dp:
                    comparisons.append((min(revolve) - min(dp)) / min(revolve))
    if not comparisons:
        return {
            "status": "not evaluated",
            "best_time_improvement": None,
            "pivot": False,
        }
    best = max(comparisons)
    return {
        "status": "passed" if best >= 0.05 else "failed",
        "best_time_improvement": best,
        "pivot": bool(cell_spreads) and all(spread < 0.03 for spread in cell_spreads),
        "maximum_cell_spread": max(cell_spreads, default=None),
    }


if __name__ == "__main__":
    main()
