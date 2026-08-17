import pytest
import torch

from torchrevolve.memmodel import TransformerShape, transformer_activation_bytes


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

