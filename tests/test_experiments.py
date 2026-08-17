from torchrevolve.experiments import (
    analytic_chain_profile,
    budget_grid,
    exclusive_fit,
    largest_trainable,
)
from torchrevolve.model import TinyGPTConfig


def test_analytic_profiles_match_across_granularities() -> None:
    config = TinyGPTConfig(depth=3, width=16, heads=4, max_sequence_length=8)
    coarse = analytic_chain_profile(config, batch_size=2, sequence_length=8)
    fine = analytic_chain_profile(
        config,
        batch_size=2,
        sequence_length=8,
        granularity="fine",
    )
    assert len(coarse.units) == 3
    assert len(fine.units) == 6
    assert coarse.activation_bytes == fine.activation_bytes
    assert coarse.forward_seconds == fine.forward_seconds


def test_budget_grid_and_largest_trainable() -> None:
    config = TinyGPTConfig(depth=1, width=8, heads=2, max_sequence_length=8)
    records = budget_grid(
        config,
        depths=[1, 2],
        sequence_lengths=[4, 8],
        schedulers=["none", "revolve", "dp"],
        byte_budget=1_000_000,
    )
    largest = largest_trainable(records)
    assert len(records) == 12
    assert all(record["fits"] for record in records)
    assert largest["none"]["configuration_size"] == 16


def test_exclusive_fit_requires_preferred_only_configuration() -> None:
    records = [
        {
            "scheduler": scheduler,
            "depth": depth,
            "sequence_length": 8,
            "configuration_size": depth * 8,
            "fits": scheduler == "dp" or depth == 2,
        }
        for scheduler in ("dp", "uniform")
        for depth in (2, 4)
    ]
    candidate = exclusive_fit(records, preferred="dp", baseline="uniform")
    assert candidate is not None
    assert candidate["depth"] == 4
    assert exclusive_fit(records[:2], preferred="dp", baseline="missing") is None
