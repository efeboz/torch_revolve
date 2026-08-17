"""Analytic activation-memory model for explicit transformer operations."""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import torch

T = TypeVar("T")


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


@dataclass(frozen=True)
class MemoryValidation:
    predicted_bytes: int
    measured_bytes: int
    relative_error: float
    within_tolerance: bool
    device_type: str


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


def allocator_available(device: str | torch.device) -> bool:
    target = torch.device(device)
    if target.type == "cuda":
        return torch.cuda.is_available()
    if target.type == "mps":
        return torch.backends.mps.is_available()
    return False


def allocator_prediction_bytes(
    logical_bytes: int,
    device: str | torch.device,
    *,
    cumulative_activation_bytes: int | None = None,
) -> int:
    """Return a conservative allocator prediction for logical activation bytes."""
    if logical_bytes < 0:
        raise ValueError("logical_bytes cannot be negative")
    if cumulative_activation_bytes is not None and cumulative_activation_bytes < 0:
        raise ValueError("cumulative_activation_bytes cannot be negative")
    if torch.device(device).type == "mps":
        basis = max(logical_bytes, cumulative_activation_bytes or 0)
        return (5 * basis + 3) // 4
    return logical_bytes


def measure_allocator_delta(
    operation: Callable[[], T],
    *,
    device: str | torch.device,
) -> tuple[T, int]:
    target = torch.device(device)
    if not allocator_available(target):
        raise RuntimeError(f"allocator counters are unavailable for {target.type}")
    gc.collect()
    if target.type == "cuda":
        torch.cuda.synchronize(target)
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(target)
        torch.cuda.reset_peak_memory_stats(target)
        result = operation()
        torch.cuda.synchronize(target)
        peak = torch.cuda.max_memory_allocated(target)
    else:
        torch.mps.synchronize()
        torch.mps.empty_cache()
        baseline = torch.mps.current_allocated_memory()
        result = operation()
        torch.mps.synchronize()
        peak = torch.mps.current_allocated_memory()
    return result, max(0, peak - baseline)


def measure_saved_activation_bytes(
    module: torch.nn.Module,
    operation: Callable[[], T],
) -> tuple[T, int]:
    parameter_storages = {
        parameter.untyped_storage().data_ptr() for parameter in module.parameters()
    }
    saved_bytes = 0

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal saved_bytes
        if tensor.untyped_storage().data_ptr() not in parameter_storages:
            saved_bytes += tensor.numel() * tensor.element_size()
        return tensor

    def unpack(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        result = operation()
    return result, saved_bytes


def validate_activation_measurement(
    shape: TransformerShape,
    measured_bytes: int,
    *,
    dtype: torch.dtype = torch.float32,
    device_type: str,
    tolerance: float = 0.1,
) -> MemoryValidation:
    if measured_bytes < 0:
        raise ValueError("measured_bytes cannot be negative")
    if tolerance < 0:
        raise ValueError("tolerance cannot be negative")
    predicted = transformer_activation_bytes(shape, dtype).total
    relative_error = abs(measured_bytes - predicted) / max(1, predicted)
    return MemoryValidation(
        predicted_bytes=predicted,
        measured_bytes=measured_bytes,
        relative_error=relative_error,
        within_tolerance=relative_error <= tolerance,
        device_type=device_type,
    )


def validate_allocator_measurement(
    shape: TransformerShape,
    measured_bytes: int,
    *,
    dtype: torch.dtype = torch.float32,
    device_type: str,
    tolerance: float = 0.1,
) -> MemoryValidation:
    """Validate a retained-activation measurement against the analytic model."""
    return validate_activation_measurement(
        shape,
        measured_bytes,
        dtype=dtype,
        device_type=device_type,
        tolerance=tolerance,
    )
