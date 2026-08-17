from __future__ import annotations

import argparse
import json
from pathlib import Path

from torchrevolve.benchmark import benchmark_named_schedule
from torchrevolve.experiments import analytic_chain_profile
from torchrevolve.heuristics import make_none_schedule
from torchrevolve.model import TinyGPTConfig
from torchrevolve.selection import SCHEDULERS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark one torch-revolve configuration"
    )
    parser.add_argument("--scheduler", choices=SCHEDULERS, default="revolve")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--budget-fraction", type=float, default=0.5)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="benchmarks/result.json")
    args = parser.parse_args()
    config = TinyGPTConfig(
        vocab_size=256,
        max_sequence_length=args.sequence_length,
        depth=args.depth,
        width=args.width,
        heads=args.heads,
    )
    granularity = "fine" if args.scheduler in {"dp", "selective"} else "coarse"
    profile = analytic_chain_profile(
        config,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        granularity=granularity,
    )
    full_bytes = make_none_schedule(profile).predicted().peak_bytes
    budget = max(1, int(full_bytes * args.budget_fraction))
    record, _ = benchmark_named_schedule(
        config,
        args.scheduler,
        byte_budget=budget,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        repetitions=args.repetitions,
        device=args.device,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
