"""Pure PyTorch executor for explicit chain schedules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from torchrevolve.chain import BlockChain, ChainProfile, UnitProfile
from torchrevolve.memmodel import TransformerShape, transformer_activation_bytes
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


@dataclass(frozen=True)
class _Tape:
    inputs: Tensor
    output: Tensor
    rng: _RNGState


def run_scheduled_backward(
    chain: BlockChain,
    schedule: Schedule,
    batch: Tensor | tuple[Tensor, ...],
    loss_fn: Callable[..., Tensor],
    *,
    replay_rng: bool = True,
    set_to_none: bool = True,
    capture_gradients: bool = True,
) -> GradResult:
    """Run a scheduled forward and backward pass; gradients land in ``.grad``."""
    if not hasattr(chain.model, "prepare_inputs") or not hasattr(chain.model, "finish"):
        raise TypeError("chain model must expose prepare_inputs and finish")
    model_inputs, loss_args = _split_batch(batch)
    device = model_inputs.device
    profile = _validation_profile(chain, model_inputs)
    schedule.validate(profile).require_valid()

    chain.model.zero_grad(set_to_none=set_to_none)
    original_input = chain.model.prepare_inputs(model_inputs)
    current = original_input.detach()
    current_rng = _capture_rng(device)
    snapshots = {0: _Snapshot(current, current_rng)}
    kept_tapes: dict[int, _Tape] = {}
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
            elif action.kind == "forward_keep":
                kept_tapes[unit] = _Tape(live_input, live_output, current_rng)
        elif action.kind == "restore":
            if action.unit in snapshots:
                snapshot = snapshots[action.unit]
                current = snapshot.value
                current_rng = snapshot.rng
            elif action.unit > 0 and action.unit - 1 in kept_tapes:
                tape = kept_tapes[action.unit - 1]
                current = tape.output.detach()
                current_rng = tape.rng
            else:
                raise RuntimeError(f"state {action.unit} is not retained")
            live_input = None
            live_output = None
        elif action.kind == "backward":
            if unit in kept_tapes:
                tape = kept_tapes.pop(unit)
                backward_input = tape.inputs
                backward_output = tape.output
            elif live_input is not None and live_output is not None:
                backward_input = live_input
                backward_output = live_output
            else:
                raise RuntimeError(f"backward {unit} has no live forward graph")
            if unit == len(chain.units) - 1 and loss is None:
                logits = chain.model.finish(backward_output)
                loss = loss_fn(logits, *loss_args)
                if loss.ndim != 0:
                    raise ValueError("loss_fn must return a scalar tensor")
                loss.backward()
            else:
                if adjoint is None:
                    raise RuntimeError("missing adjoint for scheduled backward")
                torch.autograd.backward(backward_output, adjoint)
            if backward_input.grad is None:
                raise RuntimeError(f"unit {unit} did not produce an input gradient")
            adjoint = backward_input.grad.detach()
            current = backward_input.detach()
            live_input = None
            live_output = None
            snapshots.pop(unit, None)

    if loss is None or adjoint is None:
        raise RuntimeError("schedule did not complete the backward sweep")
    original_input.backward(adjoint)
    gradients = (
        {
            name: parameter.grad.detach().clone()
            for name, parameter in chain.model.named_parameters()
            if parameter.grad is not None
        }
        if capture_gradients
        else {}
    )
    n_units = len(chain.units)
    return GradResult(
        loss=loss.detach().clone(),
        gradients=gradients,
        forward_evaluations=forward_evaluations,
        recomputations=forward_evaluations - n_units,
    )


def _split_batch(
    batch: Tensor | tuple[Tensor, ...],
) -> tuple[Tensor, tuple[Tensor, ...]]:
    if isinstance(batch, Tensor):
        return batch, ()
    if not batch or not all(isinstance(item, Tensor) for item in batch):
        raise TypeError("batch must be a tensor or a non-empty tuple of tensors")
    return batch[0], batch[1:]


def _validation_profile(chain: BlockChain, inputs: Tensor) -> ChainProfile:
    model_dtype = next(chain.model.parameters()).dtype
    config = chain.model.config
    width = getattr(config, "width", 1)
    state_bytes = inputs.shape[0] * inputs.shape[1] * width * model_dtype.itemsize
    breakdown = transformer_activation_bytes(
        TransformerShape(
            inputs.shape[0],
            inputs.shape[1],
            width,
            config.heads,
            config.mlp_ratio,
        ),
        model_dtype,
    )

    def activation_bytes(kind: str) -> int:
        if kind == "block":
            return breakdown.total
        residual_share = breakdown.norms_and_residuals // 2
        if kind == "attention":
            return breakdown.attention + residual_share
        return breakdown.mlp + residual_share

    units = tuple(
        UnitProfile(
            name=unit.name,
            forward_seconds=0.0,
            activation_bytes=activation_bytes(unit.kind),
            parameter_count=sum(
                parameter.numel() for parameter in unit.module.parameters()
            ),
            kind=unit.kind,
            state_bytes=state_bytes,
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
