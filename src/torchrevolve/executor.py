"""Pure PyTorch executor for explicit chain schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from torchrevolve.chain import BlockChain, ChainProfile, UnitProfile
from torchrevolve.schedules import Schedule


@dataclass(frozen=True)
class GradResult:
    loss: Tensor
    gradients: dict[str, Tensor]
    forward_evaluations: int
    recomputations: int


@dataclass(frozen=True)
class _RNGState:
    cpu: Tensor
    device: Tensor | None


@dataclass(frozen=True)
class _Snapshot:
    value: Tensor
    rng: _RNGState


def run_scheduled_backward(
    chain: BlockChain,
    schedule: Schedule,
    batch: Tensor | tuple[Tensor, ...],
    loss_fn: Callable[..., Tensor],
    *,
    replay_rng: bool = True,
) -> GradResult:
    """Run a scheduled forward and backward pass; gradients land in ``.grad``."""
    if not hasattr(chain.model, "prepare_inputs") or not hasattr(chain.model, "finish"):
        raise TypeError("chain model must expose prepare_inputs and finish")
    model_inputs, loss_args = _split_batch(batch)
    device = model_inputs.device
    profile = _validation_profile(chain, model_inputs)
    schedule.validate(profile).require_valid()

    chain.model.zero_grad(set_to_none=True)
    original_input = chain.model.prepare_inputs(model_inputs)
    current = original_input.detach()
    current_rng = _capture_rng(device)
    snapshots = {0: _Snapshot(current, current_rng)}
    first_forward = [True] * len(chain.units)
    live_input: Tensor | None = None
    live_output: Tensor | None = None
    adjoint: Tensor | None = None
    loss: Tensor | None = None
    forward_evaluations = 0

    for action in schedule.actions:
        unit = action.unit
        if action.kind.startswith("forward"):
            live_input = current.detach().requires_grad_(True)
            if first_forward[unit] or not replay_rng:
                live_output = chain.units[unit].module(live_input)
                next_rng = _capture_rng(device)
            else:
                with _fork_rng(device):
                    _restore_rng(current_rng, device)
                    live_output = chain.units[unit].module(live_input)
                    next_rng = _capture_rng(device)
            first_forward[unit] = False
            current = live_output
            current_rng = next_rng
            forward_evaluations += 1
            if action.kind == "forward_store" and unit + 1 < len(chain.units):
                snapshots[unit + 1] = _Snapshot(current.detach(), current_rng)
        elif action.kind == "restore":
            snapshot = snapshots[action.unit]
            current = snapshot.value
            current_rng = snapshot.rng
            live_input = None
            live_output = None
        elif action.kind == "backward":
            if live_input is None or live_output is None:
                raise RuntimeError(f"backward {unit} has no live forward graph")
            if unit == len(chain.units) - 1 and loss is None:
                logits = chain.model.finish(live_output)
                loss = loss_fn(logits, *loss_args)
                if loss.ndim != 0:
                    raise ValueError("loss_fn must return a scalar tensor")
                loss.backward()
            else:
                if adjoint is None:
                    raise RuntimeError("missing adjoint for scheduled backward")
                torch.autograd.backward(live_output, adjoint)
            if live_input.grad is None:
                raise RuntimeError(f"unit {unit} did not produce an input gradient")
            adjoint = live_input.grad.detach()
            current = live_input.detach()
            live_input = None
            live_output = None
            snapshots.pop(unit, None)

    if loss is None or adjoint is None:
        raise RuntimeError("schedule did not complete the backward sweep")
    original_input.backward(adjoint)
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in chain.model.named_parameters()
        if parameter.grad is not None
    }
    n_units = len(chain.units)
    return GradResult(
        loss=loss.detach().clone(),
        gradients=gradients,
        forward_evaluations=forward_evaluations,
        recomputations=forward_evaluations - n_units,
    )


def _split_batch(batch: Tensor | tuple[Tensor, ...]) -> tuple[Tensor, tuple[Tensor, ...]]:
    if isinstance(batch, Tensor):
        return batch, ()
    if not batch or not all(isinstance(item, Tensor) for item in batch):
        raise TypeError("batch must be a tensor or a non-empty tuple of tensors")
    return batch[0], batch[1:]


def _validation_profile(chain: BlockChain, inputs: Tensor) -> ChainProfile:
    model_dtype = next(chain.model.parameters()).dtype
    width = getattr(chain.model.config, "width", 1)
    state_bytes = inputs.shape[0] * inputs.shape[1] * width * model_dtype.itemsize
    units = tuple(
        UnitProfile(
            name=unit.name,
            forward_seconds=0.0,
            activation_bytes=state_bytes,
            parameter_count=sum(parameter.numel() for parameter in unit.module.parameters()),
            kind=unit.kind,
        )
        for unit in chain.units
    )
    return ChainProfile(
        units=units,
        batch_size=inputs.shape[0],
        sequence_length=inputs.shape[1],
        dtype=model_dtype,
        granularity=chain.granularity,
    )


def _capture_rng(device: torch.device) -> _RNGState:
    device_state = None
    if device.type != "cpu":
        device_module = getattr(torch, device.type, None)
        if device_module is None or not hasattr(device_module, "get_rng_state"):
            raise NotImplementedError(f"RNG replay is unavailable for {device.type}")
        device_state = device_module.get_rng_state(device)
    return _RNGState(torch.get_rng_state(), device_state)


def _restore_rng(state: _RNGState, device: torch.device) -> None:
    torch.set_rng_state(state.cpu)
    if state.device is not None:
        getattr(torch, device.type).set_rng_state(state.device, device)


def _fork_rng(device: torch.device):
    if device.type == "cpu":
        return torch.random.fork_rng(devices=[])
    device_index = device.index if device.index is not None else 0
    return torch.random.fork_rng(devices=[device_index], device_type=device.type)
