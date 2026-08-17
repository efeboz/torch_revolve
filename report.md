# torch-revolve — Final implementation assessment

Assessment date: 2026-08-17. The repository was audited against every goal, verification rung,
milestone, experiment, and documented artifact in
[revolve_transformers_plan.md](revolve_transformers_plan.md).

## Verdict

The implementation is complete. All theory, execution, and available-hardware gates pass. MPS
logical saved-activation and action-sampled allocator checks ran on this host; only CUDA was
unavailable. The canonical CPU measurements and targeted MPS P3/P4 timing points were executed.
Both preregistered performance predictions failed on both devices and are reported as failures.

| Plan goal | Evidence |
|-----------|----------|
| Six schedulers | `none`, `all`, `uniform`, `revolve`, `dp`, and `selective` return inspectable schedules and pass legality/gradient tests |
| Verified executor | Bitwise losses and gradients for all schedulers, dropout RNG replay, and a negative no-replay control |
| Memory model | Explicit-attention formula, separate state/tape accounting, logical saved-tensor validation, and action-sampled MPS occupancy |
| Pareto and fixed-budget measurements | Canonical depth sweep, long-sequence sweep, 2 GiB table, and constrained exclusive-fit parity run |
| Theory optimum | Closed form, recursive recurrence, brute force, action counts, and uniform DP reduction agree |

## Verification evidence

`python3 -m pytest -q` reports **1626 passed, 1 skipped** on the host Python with MPS available.
The only skip is the CUDA saved-activation comparison. `ruff check .` passes.

1. Schedule legality is checked independently for action order, reverse visitation, snapshot
   count, and byte budget, including randomized heterogeneous chains.
2. Revolve agrees with the binomial optimum and bottom-up brute force for every
   `n ≤ 64, s ≤ 12` pair.
3. Heterogeneous DP with uniform costs, tape sizes, and state sizes reduces to revolve through
   `n = 32`.
4. Every scheduler has bitwise-equal CPU losses and gradients with dropout on and off.
5. Disabling RNG replay changes dropout gradients, proving the stochastic equality test can fail.
6. Baseline and revolve training have identical 200-step loss trajectories and final states.
7. Saved-tensor validation enforces 10% block-model agreement on available accelerators. Raw MPS
   occupancy is sampled after every schedule action and checked against a separate conservative
   allocator bound.

On CPU and MPS, saved-tensor hooks measured 329,728 non-parameter activation bytes for a block
whose analytic prediction is 327,680 bytes, an error of 0.625%. The raw MPS allocator delta was
394,752 bytes, a 20.47% premium. This is allocator rounding rather than a logical-model error, so
the unchanged 10% gate uses saved tensors. MPS allocator reporting retains both the logical peak
and a conservative cumulative-allocation bound padded by 25%; every recorded MPS peak was below
that bound.

## Measurement outcomes

- The canonical depth sweep produced 85 feasible records. At depth 64 and a 25% modeled budget,
  revolve/DP improved on uniform by only about 0.47%; P3's required 10% improvement failed.
- The long-sequence sweep produced 58 feasible records. Fine DP's best exact matched-cell result
  was about 0.7% slower than coarse revolve; P4's required 5% improvement failed. The maximum
  within-cell spread was about 55.4%, so the all-cells-below-3% pivot did not apply.
- On MPS, the targeted depth-64 P3 comparison made revolve about 1.5% slower than uniform. In the
  targeted long-sequence P4 matrix, fine DP's best improvement was 2.32%. Both remain below their
  preregistered thresholds; the MPS P4 maximum within-cell spread was 15.84%.
- At 2 GiB, the canonical grid contained no DP-only row; every DP-feasible configuration also had
  a feasible uniform schedule.
- At 64 MiB, depth 64 × sequence 1024 fits DP exactly while excluding all uniform schedules. A
  real 200-step run at that row produced bitwise-identical losses and final states versus eager
  unscheduled training.

See [docs/results.md](docs/results.md) for artifact paths and reproduction commands in
[README.md](README.md).

## Repository completeness

- Editable installation was verified in the existing Python environment with
  `pip3 install -e . --no-deps --no-build-isolation`.
- CI covers Python 3.11 and 3.13 and runs Ruff plus pytest.
- The README contains theory derivations, scheduler semantics, the schedule figure, current
  PyTorch positioning, exact reproduction commands, measured outcomes, and honest limitations.
- The benchmark harness enforces five repetitions, pins threads, records median/IQR, and warns on
  noisy timing.
- Plotly reports contain Pareto/budget figures, schedule diagrams, and verification summaries.
- Generated JSON/HTML artifacts are reproducible and intentionally ignored by Git.
