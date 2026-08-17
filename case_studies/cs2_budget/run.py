from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from torchrevolve.dp import make_schedule as make_dp_schedule
from torchrevolve.experiments import budget_grid, exclusive_fit, largest_trainable
from torchrevolve.model import TinyGPTConfig
from torchrevolve.report import write_budget_report
from torchrevolve.selection import SCHEDULERS
from torchrevolve.training import train_baseline, train_scheduled


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed activation-budget configuration sweep"
    )
    parser.add_argument("--depths", default="4,8,16,32,64")
    parser.add_argument("--sequence-lengths", default="256,512,1024,2048")
    parser.add_argument("--schedulers", default=",".join(SCHEDULERS))
    parser.add_argument("--budget-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--training-steps", type=int, default=0)
    parser.add_argument("--output", default="case_studies/cs2_budget/results.json")
    parser.add_argument("--report", default="case_studies/cs2_budget/report.html")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        depths = [2, 4]
        sequence_lengths = [4, 8]
        width = 8
        heads = 2
        budget = 1_000_000
    else:
        depths = parse_ints(args.depths)
        sequence_lengths = parse_ints(args.sequence_lengths)
        width = args.width
        heads = args.heads
        budget = args.budget_bytes
    schedulers = [item for item in args.schedulers.split(",") if item]
    base = TinyGPTConfig(
        vocab_size=256,
        max_sequence_length=max(sequence_lengths),
        depth=min(depths),
        width=width,
        heads=heads,
    )
    records = budget_grid(
        base,
        depths=depths,
        sequence_lengths=sequence_lengths,
        schedulers=schedulers,
        byte_budget=budget,
        batch_size=args.batch_size,
    )
    largest = largest_trainable(records)
    headline = exclusive_fit(records, preferred="dp", baseline="uniform")
    parity = None
    if args.training_steps > 0 and headline is not None:
        candidate = headline
        config = TinyGPTConfig(
            vocab_size=256,
            max_sequence_length=int(candidate["sequence_length"]),
            depth=int(candidate["depth"]),
            width=width,
            heads=heads,
            dropout=0.1,
        )
        options = {
            "steps": args.training_steps,
            "batch_size": args.batch_size,
            "sequence_length": int(candidate["sequence_length"]),
            "seed": 53,
        }
        expected = train_baseline(config, **options)
        actual = train_scheduled(
            config,
            lambda profile: make_dp_schedule(profile, budget=budget),
            granularity="fine",
            **options,
        )
        losses_equal = actual.losses == expected.losses
        states_equal = all(
            torch.equal(actual.state[name], expected.state[name])
            for name in actual.state
        )
        parity = {
            "status": "passed" if losses_equal and states_equal else "failed",
            "configuration": candidate,
            "steps": args.training_steps,
            "losses_equal": losses_equal,
            "states_equal": states_equal,
        }
    elif args.training_steps > 0:
        parity = {
            "status": "not run",
            "reason": "no configuration fits DP while excluding every uniform-k schedule",
        }
    payload = {
        "configuration": vars(args),
        "records": records,
        "largest_trainable": largest,
        "headline_candidate": headline,
        "training_parity": parity,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_budget_report(
        args.report,
        records=records,
        largest=largest,
        parity=parity,
    )


if __name__ == "__main__":
    main()
