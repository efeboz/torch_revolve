"""Timing and allocator benchmark harness for scheduled training steps."""

from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.nn import functional as F

from torchrevolve.chain import BlockChain
from torchrevolve.executor import run_scheduled_backward
from torchrevolve.memmodel import allocator_available, measure_allocator_delta
from torchrevolve.model import TinyGPT, TinyGPTConfig
from torchrevolve.selection import ScheduleSelection, select_schedule


@dataclass(frozen=True)
class BenchmarkRecord:
    scheduler: str
    depth: int
    sequence_length: int
    batch_size: int
    byte_budget: int
    parameter: int | str
    predicted_peak_bytes: int
    recomputations: int
    median_seconds: float
    q1_seconds: float
    q3_seconds: float
    repetitions: int
    measured_peak_bytes: int | None
    prediction_upper_bound: bool | None
    label: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def benchmark_named_schedule(
    config: TinyGPTConfig,
    scheduler: str,
    *,
    byte_budget: int,
    batch_size: int = 2,
    sequence_length: int = 256,
    repetitions: int = 5,
    warmup: int = 1,
    device: str | torch.device = "cpu",
    threads: int = 1,
    seed: int = 0,
    measure_peak: bool = True,
) -> tuple[BenchmarkRecord, ScheduleSelection]:
    if repetitions < 5:
        raise ValueError("timing benchmarks require at least five repetitions")
    if warmup < 0 or threads <= 0:
        raise ValueError("warmup cannot be negative and threads must be positive")
    target = torch.device(device)
    torch.manual_seed(seed)
    model = TinyGPT(config).to(target)
    granularity = "fine" if scheduler in {"dp", "selective"} else "coarse"
    chain = BlockChain.from_model(model, granularity=granularity)
    generator = torch.Generator(device=target).manual_seed(seed + 1)
    tokens = torch.randint(
        config.vocab_size,
        (batch_size, sequence_length),
        generator=generator,
        device=target,
    )
    targets = torch.randint(
        config.vocab_size,
        (batch_size, sequence_length),
        generator=generator,
        device=target,
    )
    profile = chain.profile(tokens, device=target, n_reps=3)
    selection = select_schedule(profile, scheduler, byte_budget=byte_budget)

    def loss_fn(logits: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.flatten(0, 1), expected.flatten())

    def step(*, reuse_gradients: bool = False):
        return run_scheduled_backward(
            chain,
            selection.schedule,
            (tokens, targets),
            loss_fn,
            set_to_none=not reuse_gradients,
            capture_gradients=False,
        )

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        for _ in range(warmup):
            step()
        samples = []
        for repetition in range(repetitions):
            torch.manual_seed(seed + 100 + repetition)
            _synchronize(target)
            started = time.perf_counter()
            step(reuse_gradients=True)
            _synchronize(target)
            samples.append(time.perf_counter() - started)
        measured_peak = None
        if measure_peak and allocator_available(target):
            _, measured_peak = measure_allocator_delta(
                lambda: step(reuse_gradients=True),
                device=target,
            )
    finally:
        torch.set_num_threads(previous_threads)
    q1, median, q3 = (float(value) for value in np.percentile(samples, [25, 50, 75]))
    if median and (q3 - q1) / median > 0.1:
        warnings.warn(
            f"high timing dispersion for {scheduler}: IQR is {(q3 - q1) / median:.1%}",
            RuntimeWarning,
            stacklevel=2,
        )
    predicted = selection.schedule.predicted().peak_bytes
    upper_bound = None if measured_peak is None else predicted >= measured_peak
    record = BenchmarkRecord(
        scheduler=scheduler,
        depth=config.depth,
        sequence_length=sequence_length,
        batch_size=batch_size,
        byte_budget=byte_budget,
        parameter=selection.parameter,
        predicted_peak_bytes=predicted,
        recomputations=selection.schedule.predicted().recomputations,
        median_seconds=median,
        q1_seconds=q1,
        q3_seconds=q3,
        repetitions=repetitions,
        measured_peak_bytes=measured_peak,
        prediction_upper_bound=upper_bound,
        label=f"depth={config.depth}, seq={sequence_length}, budget={byte_budget}",
    )
    return record, selection


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
