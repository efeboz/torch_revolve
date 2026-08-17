"""Analytic profiles and experiment-summary helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict

import torch

from torchrevolve.chain import ChainProfile, UnitProfile
from torchrevolve.memmodel import TransformerShape, transformer_activation_bytes
from torchrevolve.model import TinyGPTConfig
from torchrevolve.selection import select_schedule


def analytic_chain_profile(
    config: TinyGPTConfig,
    *,
    batch_size: int,
    sequence_length: int,
    granularity: str = "coarse",
    dtype: torch.dtype = torch.float32,
) -> ChainProfile:
    shape = TransformerShape(
        batch_size,
        sequence_length,
        config.width,
        config.heads,
        config.mlp_ratio,
    )
    memory = transformer_activation_bytes(shape, dtype)
    state_bytes = batch_size * sequence_length * config.width * dtype.itemsize
    attention_cost = float(
        8 * batch_size * sequence_length * config.width**2
        + 4 * batch_size * sequence_length**2 * config.width
    )
    mlp_cost = float(
        4
        * batch_size
        * sequence_length
        * config.width
        * (config.mlp_ratio * config.width)
    )
    attention_parameters = 4 * config.width**2 + 4 * config.width
    hidden = config.mlp_ratio * config.width
    mlp_parameters = 2 * config.width * hidden + hidden + config.width
    units: list[UnitProfile] = []
    if granularity == "coarse":
        for layer in range(config.depth):
            units.append(
                UnitProfile(
                    f"block.{layer}",
                    attention_cost + mlp_cost,
                    memory.total,
                    attention_parameters + mlp_parameters,
                    "block",
                    state_bytes,
                )
            )
    elif granularity == "fine":
        residual_share = memory.norms_and_residuals // 2
        for layer in range(config.depth):
            units.extend(
                (
                    UnitProfile(
                        f"block.{layer}.attention",
                        attention_cost,
                        memory.attention + residual_share,
                        attention_parameters,
                        "attention",
                        state_bytes,
                    ),
                    UnitProfile(
                        f"block.{layer}.mlp",
                        mlp_cost,
                        memory.mlp + residual_share,
                        mlp_parameters,
                        "mlp",
                        state_bytes,
                    ),
                )
            )
    else:
        raise ValueError("granularity must be 'coarse' or 'fine'")
    return ChainProfile(
        tuple(units),
        batch_size,
        sequence_length,
        dtype,
        granularity,
    )


def budget_grid(
    base_config: TinyGPTConfig,
    *,
    depths: Iterable[int],
    sequence_lengths: Iterable[int],
    schedulers: Iterable[str],
    byte_budget: int,
    batch_size: int = 1,
) -> list[dict[str, object]]:
    records = []
    for depth in depths:
        for sequence_length in sequence_lengths:
            config_data = asdict(base_config)
            config_data.update(depth=depth, max_sequence_length=max(sequence_length, 1))
            config = TinyGPTConfig(**config_data)
            for scheduler in schedulers:
                granularity = "fine" if scheduler in {"dp", "selective"} else "coarse"
                profile = analytic_chain_profile(
                    config,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    granularity=granularity,
                )
                try:
                    selection = select_schedule(
                        profile,
                        scheduler,
                        byte_budget=byte_budget,
                    )
                except (MemoryError, ValueError):
                    fits = False
                    peak = None
                    recompute_cost = None
                    parameter = None
                else:
                    fits = True
                    peak = selection.schedule.predicted().peak_bytes
                    recompute_cost = selection.schedule.predicted().recompute_cost
                    parameter = selection.parameter
                records.append(
                    {
                        "scheduler": scheduler,
                        "depth": depth,
                        "sequence_length": sequence_length,
                        "tokens_per_step": batch_size * sequence_length,
                        "configuration_size": depth * sequence_length,
                        "byte_budget": byte_budget,
                        "fits": fits,
                        "predicted_peak_bytes": peak,
                        "recompute_cost": recompute_cost,
                        "parameter": parameter,
                    }
                )
    return records


def largest_trainable(
    records: Iterable[dict[str, object]],
) -> dict[str, dict[str, object] | None]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["scheduler"]), []).append(record)
    return {
        scheduler: max(
            (record for record in items if record["fits"]),
            key=lambda record: (
                int(record["configuration_size"]),
                int(record["depth"]),
            ),
            default=None,
        )
        for scheduler, items in grouped.items()
    }


def exclusive_fit(
    records: Iterable[dict[str, object]],
    *,
    preferred: str,
    baseline: str,
) -> dict[str, object] | None:
    items = list(records)
    lookup = {
        (record["scheduler"], record["depth"], record["sequence_length"]): record
        for record in items
    }
    candidates = [
        record
        for record in items
        if record["scheduler"] == preferred
        and record["fits"]
        and (baseline, record["depth"], record["sequence_length"]) in lookup
        and not lookup[(baseline, record["depth"], record["sequence_length"])]["fits"]
    ]
    return max(
        candidates,
        key=lambda record: (record["configuration_size"], record["depth"]),
        default=None,
    )
