import warnings

import torch

from torchrevolve.benchmark import benchmark_named_schedule
from torchrevolve.chain import BlockChain
from torchrevolve.memmodel import allocator_available
from torchrevolve.model import TinyGPT, TinyGPTConfig


def test_benchmark_harness_smoke() -> None:
    config = TinyGPTConfig(
        vocab_size=16,
        max_sequence_length=4,
        depth=2,
        width=8,
        heads=2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        record, selection = benchmark_named_schedule(
            config,
            "revolve",
            byte_budget=1_000_000,
            batch_size=1,
            sequence_length=4,
            repetitions=5,
            measure_peak=False,
        )
    assert record.repetitions == 5
    assert 0 < record.q1_seconds <= record.q3_seconds
    assert record.predicted_peak_bytes <= record.byte_budget
    assert record.allocator_prediction_bytes >= record.predicted_peak_bytes
    chain = BlockChain.from_model(TinyGPT(config))
    profile = chain.profile(torch.zeros(1, 4, dtype=torch.long), n_reps=1)
    assert selection.schedule.validate(profile).legal


def test_prediction_upper_bounds_hardware_peak() -> None:
    device = "cuda" if allocator_available("cuda") else "mps"
    if not allocator_available(device):
        import pytest

        pytest.skip("CUDA and MPS are unavailable")
    config = TinyGPTConfig(
        vocab_size=32,
        max_sequence_length=16,
        depth=2,
        width=32,
        heads=4,
    )
    record, _ = benchmark_named_schedule(
        config,
        "none",
        byte_budget=100_000_000,
        batch_size=2,
        sequence_length=16,
        repetitions=5,
        device=device,
    )
    assert record.prediction_upper_bound, record
