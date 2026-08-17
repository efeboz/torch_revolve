import torch

from torchrevolve.chain import BlockChain
from torchrevolve.model import TinyGPT, TinyGPTConfig


def make_model() -> TinyGPT:
    return TinyGPT(
        TinyGPTConfig(
            vocab_size=32,
            max_sequence_length=8,
            depth=2,
            width=16,
            heads=4,
        )
    )


def test_coarse_chain_profile() -> None:
    model = make_model()
    chain = BlockChain.from_model(model, granularity="coarse")
    profile = chain.profile(torch.zeros(2, 8, dtype=torch.long), n_reps=1)
    assert len(chain.units) == 2
    assert len(profile.units) == 2
    assert all(unit.kind == "block" for unit in profile.units)
    assert profile.activation_bytes > 0
    assert profile.forward_seconds > 0


def test_fine_profile_sums_to_coarse_memory() -> None:
    tokens = torch.zeros(2, 8, dtype=torch.long)
    model = make_model()
    coarse = BlockChain.from_model(model, granularity="coarse").profile(
        tokens, n_reps=1
    )
    fine = BlockChain.from_model(model, granularity="fine").profile(tokens, n_reps=1)
    assert len(fine.units) == 4
    assert [unit.kind for unit in fine.units] == ["attention", "mlp"] * 2
    assert fine.activation_bytes == coarse.activation_bytes


def test_tied_embedding_is_counted_once_by_model() -> None:
    model = make_model()
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    assert len(parameter_ids) == len(set(parameter_ids))
