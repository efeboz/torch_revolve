"""Transformer segmentation and per-unit profiling."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from torchrevolve.memmodel import TransformerShape, transformer_activation_bytes


UnitKind = Literal["block", "attention", "mlp"]


@dataclass(frozen=True)
class UnitProfile:
    name: str
    forward_seconds: float
    activation_bytes: int
    parameter_count: int
    kind: UnitKind = "block"

    def __post_init__(self) -> None:
        if self.forward_seconds < 0:
            raise ValueError("forward_seconds cannot be negative")
        if self.activation_bytes < 0:
            raise ValueError("activation_bytes cannot be negative")
        if self.parameter_count < 0:
            raise ValueError("parameter_count cannot be negative")


@dataclass(frozen=True)
class ChainProfile:
    units: tuple[UnitProfile, ...]
    batch_size: int
    sequence_length: int
    dtype: torch.dtype
    granularity: Literal["coarse", "fine"]

    def __post_init__(self) -> None:
        if not self.units:
            raise ValueError("a chain profile must contain at least one unit")
        if self.batch_size <= 0 or self.sequence_length <= 0:
            raise ValueError("batch and sequence dimensions must be positive")

    @property
    def activation_bytes(self) -> int:
        return sum(unit.activation_bytes for unit in self.units)

    @property
    def forward_seconds(self) -> float:
        return sum(unit.forward_seconds for unit in self.units)


@dataclass(frozen=True)
class ChainUnit:
    name: str
    module: nn.Module
    kind: UnitKind


class BlockChain:
    """A transformer represented as a homogeneous or fine-grained chain."""

    def __init__(
        self,
        model: nn.Module,
        units: tuple[ChainUnit, ...],
        granularity: Literal["coarse", "fine"],
    ) -> None:
        if not units:
            raise ValueError("a block chain must contain at least one unit")
        self.model = model
        self.units = units
        self.granularity = granularity

    @staticmethod
    def from_model(
        model: nn.Module,
        *,
        granularity: Literal["coarse", "fine"] = "coarse",
    ) -> "BlockChain":
        blocks = getattr(model, "blocks", None)
        if not isinstance(blocks, nn.ModuleList) or not blocks:
            raise TypeError("model must expose a non-empty nn.ModuleList named 'blocks'")
        if granularity == "coarse":
            units = tuple(
                ChainUnit(name=f"block.{index}", module=block, kind="block")
                for index, block in enumerate(blocks)
            )
        elif granularity == "fine":
            expanded: list[ChainUnit] = []
            for index, block in enumerate(blocks):
                if not hasattr(block, "attention_unit") or not hasattr(block, "mlp_unit"):
                    raise TypeError("fine granularity requires attention_unit and mlp_unit")
                expanded.extend(
                    (
                        ChainUnit(
                            name=f"block.{index}.attention",
                            module=block.attention_unit,
                            kind="attention",
                        ),
                        ChainUnit(
                            name=f"block.{index}.mlp",
                            module=block.mlp_unit,
                            kind="mlp",
                        ),
                    )
                )
            units = tuple(expanded)
        else:
            raise ValueError("granularity must be 'coarse' or 'fine'")
        return BlockChain(model=model, units=units, granularity=granularity)

    def profile(
        self,
        batch: Tensor,
        *,
        device: str | torch.device = "cpu",
        n_reps: int = 20,
    ) -> ChainProfile:
        if n_reps <= 0:
            raise ValueError("n_reps must be positive")
        if batch.ndim != 2:
            raise ValueError("batch must contain token ids with shape (batch, sequence)")
        if not hasattr(self.model, "prepare_inputs") or not hasattr(self.model, "config"):
            raise TypeError("model must expose prepare_inputs and config")

        target_device = torch.device(device)
        self.model.to(target_device)
        tokens = batch.to(target_device)
        config = self.model.config
        shape = TransformerShape(
            batch_size=tokens.shape[0],
            sequence_length=tokens.shape[1],
            width=config.width,
            heads=config.heads,
            mlp_ratio=config.mlp_ratio,
        )
        breakdown = transformer_activation_bytes(shape, next(self.model.parameters()).dtype)
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            hidden = self.model.prepare_inputs(tokens)
            profiles = []
            for unit in self.units:
                for _ in range(2):
                    unit.module(hidden)
                samples = []
                output = hidden
                for _ in range(n_reps):
                    self._synchronize(target_device)
                    started = time.perf_counter()
                    output = unit.module(hidden)
                    self._synchronize(target_device)
                    samples.append(time.perf_counter() - started)
                activation_bytes = self._unit_bytes(unit.kind, breakdown)
                profiles.append(
                    UnitProfile(
                        name=unit.name,
                        forward_seconds=statistics.median(samples),
                        activation_bytes=activation_bytes,
                        parameter_count=sum(p.numel() for p in unit.module.parameters()),
                        kind=unit.kind,
                    )
                )
                hidden = output
        self.model.train(was_training)
        return ChainProfile(
            units=tuple(profiles),
            batch_size=tokens.shape[0],
            sequence_length=tokens.shape[1],
            dtype=next(self.model.parameters()).dtype,
            granularity=self.granularity,
        )

    @staticmethod
    def _unit_bytes(kind: UnitKind, breakdown: object) -> int:
        if kind == "block":
            return breakdown.total
        residual_share = breakdown.norms_and_residuals // 2
        if kind == "attention":
            return breakdown.attention + residual_share
        return breakdown.mlp + residual_share

    @staticmethod
    def _synchronize(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

