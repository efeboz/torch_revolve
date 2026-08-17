"""Analytic activation-memory model for explicit transformer operations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TransformerShape:
    batch_size: int
    sequence_length: int
    width: int
    heads: int
    mlp_ratio: int = 4

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")


@dataclass(frozen=True)
class ActivationBreakdown:
    attention: int
    mlp: int
    norms_and_residuals: int

    @property
    def total(self) -> int:
        return self.attention + self.mlp + self.norms_and_residuals


def dtype_size(dtype: torch.dtype) -> int:
    """Return the storage size of one scalar for a PyTorch dtype."""
    try:
        return torch.empty((), dtype=dtype).element_size()
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"unsupported dtype: {dtype}") from exc


def transformer_activation_bytes(
    shape: TransformerShape,
    dtype: torch.dtype = torch.float32,
) -> ActivationBreakdown:
    """Estimate saved activation bytes for one explicit-matmul block."""
    b = shape.batch_size
    length = shape.sequence_length
    width = shape.width
    heads = shape.heads
    hidden = shape.mlp_ratio * width
    itemsize = dtype_size(dtype)

    attention_elements = 5 * b * length * width + 2 * b * heads * length**2
    mlp_elements = 2 * b * length * hidden + b * length * width
    norm_residual_elements = 2 * b * length * width
    return ActivationBreakdown(
        attention=attention_elements * itemsize,
        mlp=mlp_elements * itemsize,
        norms_and_residuals=norm_residual_elements * itemsize,
    )

