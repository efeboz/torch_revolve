# torch-revolve

`torch-revolve` is an implementation of optimal adjoint checkpoint scheduling for transformer
chains in pure PyTorch. It connects binomial revolve schedules and heterogeneous-chain dynamic
programming to inspectable execution schedules, deterministic gradient verification, and
memory–time measurements.

The repository is under active implementation. The current foundation provides:

- a deterministic, config-driven tiny GPT with explicit matrix-multiplication attention;
- coarse block and fine attention/MLP chain segmentation;
- per-unit timing, activation-byte, and parameter profiles;
- the analytic activation-memory model specified by the project plan;
- exact homogeneous revolve schedules and heterogeneous byte-budget dynamic programming;
- an independent legality simulator for snapshot and byte budgets;
- a pure PyTorch executor with CPU and device RNG replay;
- bitwise gradient equality tests with deterministic and stochastic models.

The baseline heuristics, reporting layer, hardware memory validation, and experiment case studies
remain to be implemented.

## Installation

Use the existing Python installation directly:

```bash
python3 -m pip install -e .
python3 -m pytest
```

The runtime dependencies are PyTorch, NumPy, and Plotly. Tests use pytest.

## Memory model

For batch size `b`, sequence length `l`, width `d`, head count `h`, and scalar size `w`, the
model assigns these saved-activation estimates to one transformer block:

```text
attention:          w · (5bld + 2bhl²)
MLP:                w · (2bl(4d) + bld)
norms and residuals: w · 2bld
```

The attention implementation uses explicit QKᵀ, masking, softmax, and probability–value
multiplication. This is slower than fused attention but exposes the two `bhl²` tensors assumed
by the analytic model. CPU results use this model as memory truth; later hardware tests compare
it with CUDA or MPS allocator peaks.

## Determinism

CPU verification enables `torch.use_deterministic_algorithms(True)`, seeds model and data
generation, and uses explicit attention operations. MPS is reserved for optional timing and
allocator validation, not bitwise verification. Scheduled execution snapshots logical RNG state
with every activation snapshot and forks the ambient generator for recomputation. This reproduces
dropout masks without advancing training's global RNG stream.

## Scheduler convention

State zero, the chain input, occupies the first snapshot slot. `forward_store(i)` saves state
`i + 1`; `restore(i)` selects a saved state; and `backward(i)` consumes a live local forward
graph. A schedule's recomputation count is its number of forward actions minus the chain length.
With this convention, the maximum chain length at snapshot count `s` and repetition number `r`
is `C(s + r, s)`.

## Scope

The project supports chains at transformer-block or attention/MLP granularity. It does not
target general computation graphs, distributed execution, custom kernels, or `torch.compile`.
The planned contribution is not a claim that eager optimal schedules universally outperform
PyTorch's tuned activation-checkpointing stack. It is a verified and pedagogical bridge from
optimal chain scheduling theory to transformer profiles and PyTorch execution.

See [docs/prior-work.md](docs/prior-work.md) for the current API and literature survey and
[revolve_transformers_plan.md](revolve_transformers_plan.md) for the complete implementation
and experiment plan.
