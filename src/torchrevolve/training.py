"""Deterministic baseline training utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from torchrevolve.model import TinyGPT, TinyGPTConfig


@dataclass(frozen=True)
class TrainingTrace:
    losses: tuple[float, ...]
    state: dict[str, Tensor]


def configure_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_baseline(
    config: TinyGPTConfig,
    *,
    steps: int = 10,
    batch_size: int = 2,
    sequence_length: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> TrainingTrace:
    if steps <= 0 or batch_size <= 0 or sequence_length <= 0:
        raise ValueError("steps, batch_size, and sequence_length must be positive")
    if sequence_length > config.max_sequence_length:
        raise ValueError("sequence length exceeds model configuration")
    configure_determinism(seed)
    target_device = torch.device(device)
    model = TinyGPT(config).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    data_generator = torch.Generator(device=target_device).manual_seed(seed + 1)
    losses = []
    for _ in range(steps):
        tokens = torch.randint(
            config.vocab_size,
            (batch_size, sequence_length + 1),
            generator=data_generator,
            device=target_device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model(tokens[:, :-1], tokens[:, 1:])
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    return TrainingTrace(losses=tuple(losses), state=state)
