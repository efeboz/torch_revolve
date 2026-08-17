# Prior work and current PyTorch APIs

Survey date: 2026-08-17. The local implementation was checked against PyTorch 2.13.0 and the
current upstream documentation.

## Positioning

`torch-revolve` focuses on explicit, independently checkable schedules for transformer chains:
binomial optimality for homogeneous chains, budgeted dynamic programming for heterogeneous
chains, deterministic execution including RNG replay, and agreement between schedule theory and
measured recomputation. PyTorch offers broader operator-level and compiler-integrated activation
checkpointing. This project instead makes the restricted chain problem, its proof obligations,
and its predicted memory cost directly inspectable.

## PyTorch API survey

[`torch.utils.checkpoint.checkpoint`](https://docs.pytorch.org/docs/stable/checkpoint.html) has
reentrant and non-reentrant implementations. Upstream recommends `use_reentrant=False`. The
non-reentrant implementation records the forward autograd graph, supports keyword arguments and
all backward entry points, can stop recomputation early, and accepts separate forward and
recompute contexts. Checkpointing preserves CPU RNG state and one additional device type by
default; moving tensors to a previously unseen device inside the checkpointed function prevents
the API from guaranteeing deterministic equivalence.

The same module exposes `CheckpointPolicy` and `create_selective_checkpoint_contexts`. A policy
can require or prefer saving, recomputation, or CPU offload at operator granularity. Cache-entry
mutation checks are enabled by default. This is the correct comparison point for the project's
selective-recompute baseline.

PyTorch 2.13 also documents
[`torch.autograd.graph.region_activation_memory_budget`](https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.graph.region_activation_memory_budget),
a prototype `torch.compile` feature. It takes a ratio from zero to one and uses a min-cut
partitioner's 0–1 knapsack choice to trade saved activations for recomputation within a compiled
graph. It currently permits one budget per compiled graph. This project deliberately excludes
`torch.compile`; comparisons must describe this API rather than implying PyTorch lacks automated
budgeting.

PyTorch's
[`sdpa_kernel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)
can force the math SDPA backend, and the
[reproducibility documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html)
classifies that backend as deterministic when deterministic algorithms are enabled. This project
still uses explicit matrix multiplication so the score and softmax tensors in its analytic memory
model are concrete and backend-independent.

## Literature

- Andreas Griewank and Andrea Walther, “Algorithm 799: Revolve: An Implementation of
  Checkpointing for the Reverse or Adjoint Mode of Computational Differentiation,” ACM TOMS,
  2000. This is the primary source for homogeneous binomial schedules.
- Olivier Beaumont et al., “Optimal Checkpointing for Heterogeneous Chains: How to Train Deep
  Neural Networks with Limited Memory,” 2019. This motivates the byte-budgeted heterogeneous
  dynamic program and the uniform-cost reduction test.
- Paras Jain et al., “Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization,”
  MLSys 2020. Checkmate handles general computation graphs; `torch-revolve` intentionally limits
  itself to chains.
- Tianqi Chen et al., “Training Deep Nets with Sublinear Memory Cost,” 2016. This supplies the
  modern deep-learning account of the square-root checkpointing tradeoff.
- Megatron-LM's selective activation recomputation is the practice-oriented baseline: expensive
  operations are saved selectively while cheaper attention intermediates are recomputed.

The homogeneous implementation uses the offline recurrence
`T(n,s) = min_k [k + T(k,s) + T(n-k,s-1)]`, with
`T(n,1) = n(n-1)/2`. Tests independently compare it with the closed-form binomial count and a
bottom-up brute-force dynamic program over every planned `n ≤ 64, s ≤ 12` pair.
