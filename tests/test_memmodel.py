import pytest
import torch

from torchrevolve.memmodel import (
    TransformerShape,
    allocator_available,
    allocator_prediction_bytes,
    measure_allocator_delta,
    measure_saved_activation_bytes,
    transformer_activation_bytes,
    validate_activation_measurement,
)
from torchrevolve.model import TinyGPTConfig, TransformerBlock


def test_activation_formula() -> None:
    shape = TransformerShape(batch_size=2, sequence_length=8, width=16, heads=4)
    result = transformer_activation_bytes(shape, torch.float32)
    assert result.attention == (5 * 2 * 8 * 16 + 2 * 2 * 4 * 8**2) * 4
    assert result.mlp == (2 * 2 * 8 * 64 + 2 * 8 * 16) * 4
    assert result.norms_and_residuals == 2 * 2 * 8 * 16 * 4
    assert result.total == result.attention + result.mlp + result.norms_and_residuals


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0, "sequence_length": 8, "width": 16, "heads": 4},
        {"batch_size": 2, "sequence_length": -1, "width": 16, "heads": 4},
        {"batch_size": 2, "sequence_length": 8, "width": 15, "heads": 4},
    ],
)
def test_invalid_shapes(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        TransformerShape(**kwargs)


def test_memory_validation_result() -> None:
    shape = TransformerShape(batch_size=1, sequence_length=4, width=8, heads=2)
    predicted = transformer_activation_bytes(shape).total
    exact = validate_activation_measurement(
        shape,
        predicted,
        device_type="cuda",
    )
    outside = validate_activation_measurement(
        shape,
        predicted * 2,
        device_type="cuda",
    )
    assert exact.within_tolerance
    assert exact.relative_error == 0.0
    assert not outside.within_tolerance


def test_allocator_prediction_accounts_for_mps_rounding() -> None:
    assert allocator_prediction_bytes(327_680, "mps") == 409_600
    assert (
        allocator_prediction_bytes(
            327_680,
            "mps",
            cumulative_activation_bytes=1_000_000,
        )
        == 1_250_000
    )
    assert allocator_prediction_bytes(327_680, "cuda") == 327_680
    assert allocator_prediction_bytes(327_680, "cpu") == 327_680
    with pytest.raises(ValueError):
        allocator_prediction_bytes(-1, "mps")
    with pytest.raises(ValueError):
        allocator_prediction_bytes(
            1,
            "mps",
            cumulative_activation_bytes=-1,
        )


def test_cpu_saved_activations_match_analytic_model() -> None:
    config = TinyGPTConfig(
        vocab_size=32,
        max_sequence_length=32,
        depth=1,
        width=64,
        heads=4,
    )
    block = TransformerBlock(config)
    inputs = torch.randn(2, 32, 64, requires_grad=True)
    output, measured = measure_saved_activation_bytes(block, lambda: block(inputs))
    assert output.shape == inputs.shape
    validation = validate_activation_measurement(
        TransformerShape(2, 32, 64, 4),
        measured,
        device_type="cpu-saved-tensors",
    )
    assert validation.within_tolerance, validation


@pytest.mark.hardware
@pytest.mark.parametrize("device_type", ["cuda", "mps"])
def test_block_memory_model_matches_saved_activations(device_type: str) -> None:
    if not allocator_available(device_type):
        pytest.skip(f"{device_type} is unavailable")
    config = TinyGPTConfig(
        vocab_size=32,
        max_sequence_length=32,
        depth=1,
        width=64,
        heads=4,
    )
    block = TransformerBlock(config).to(device_type)
    inputs = torch.randn(2, 32, 64, device=device_type, requires_grad=True)
    output, measured = measure_saved_activation_bytes(block, lambda: block(inputs))
    assert output.shape == inputs.shape
    validation = validate_activation_measurement(
        TransformerShape(2, 32, 64, 4),
        measured,
        device_type=device_type,
    )
    assert validation.within_tolerance, validation


@pytest.mark.hardware
def test_mps_allocator_rounding_is_conservative() -> None:
    if not allocator_available("mps"):
        pytest.skip("mps is unavailable")
    config = TinyGPTConfig(
        vocab_size=32,
        max_sequence_length=32,
        depth=1,
        width=64,
        heads=4,
    )
    block = TransformerBlock(config).to("mps")
    inputs = torch.randn(2, 32, 64, device="mps", requires_grad=True)
    output, allocated = measure_allocator_delta(lambda: block(inputs), device="mps")
    assert output.shape == inputs.shape
    predicted = transformer_activation_bytes(TransformerShape(2, 32, 64, 4)).total
    assert allocated >= predicted
