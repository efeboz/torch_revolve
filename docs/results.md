# Measured results

Measurement date: 2026-08-17. Environment: Python 3.13.9, PyTorch 2.13.0, CPU and MPS, with one
PyTorch worker thread per benchmark. Five repetitions were collected for every timing point; the
JSON artifacts contain medians, quartiles, feasibility decisions, and skipped configurations.
CUDA was unavailable. MPS was exercised outside the filesystem sandbox because the sandboxed
process cannot acquire the device.

## Verification gates

| Gate | Result | Evidence |
|------|--------|----------|
| P1: binomial recomputation optimum | Passed | Closed form, recursive recurrence, bottom-up brute force, and generated actions agree for `n ≤ 64, s ≤ 12` |
| P2: bitwise gradient equality | Passed | Every scheduler, coarse/fine chains, dropout on/off, and negative no-replay control |
| 200-step loss trajectory | Passed | Unit test and constrained-budget depth-64 training artifact |
| CPU saved-activation agreement | Passed | 327,680 predicted bytes versus 329,728 measured, 0.625% error |
| MPS saved-activation agreement | Passed | 327,680 predicted bytes versus 329,728 measured, 0.625% error |
| MPS allocator behavior | Passed | 394,752 raw block bytes; action-sampled schedule peaks stayed below the conservative allocator bound |
| Full suite | Passed | 1,626 passed, one CUDA-only skip; Ruff passed |

## Depth sweep and P3

The canonical depth `{4, 8, 16, 32, 64}`, sequence-256 sweep produced 85 feasible matched-budget
records. At depth 64 and the 25% modeled activation budget, the best revolve/DP result improved on
the best uniform-k result by about 0.5%, below the preregistered 10% threshold. P3 therefore
failed. This is reported as a CPU regime where the additional scheduling machinery does not
produce the predicted timing advantage.

Artifacts:

- `case_studies/cs1_depth_sweep/results.json`
- `case_studies/cs1_depth_sweep/report.html`

## Long sequences and P4

The depth `{4, 8}`, sequence `{1024, 2048}` sweep produced 58 feasible records. Fine DP's best
exact matched-cell comparison was about 0.7% slower than coarse revolve, so P4's required 5%
improvement failed. The largest within-cell scheduler spread was about 55.4%, so the separate
all-cells-below-3% overhead pivot was not triggered.

Artifacts:

- `case_studies/cs1_depth_sweep/long-results.json`
- `case_studies/cs1_depth_sweep/long-report.html`

## Fixed budgets

At 2 GiB, every canonical configuration that fit DP also fit at least one uniform-k schedule. The
largest listed DP and uniform configurations were both depth 64 × sequence 2048; the plan's
exclusive-fit headline does not exist at this loose cap.

At a constrained 64 MiB cap, depth 64 × sequence 1024 is a genuine exclusive-fit row: fine DP fits
at exactly 67,108,864 predicted bytes while no uniform-k schedule fits. A real 200-step run at that
configuration produced bitwise-identical loss trajectories and final model states against the
ordinary unscheduled eager model.

Artifacts:

- `case_studies/cs2_budget/results.json` and `report.html` for 2 GiB
- `case_studies/cs2_budget/constrained-results.json` and `constrained-report.html` for 64 MiB

Generated JSON and HTML are ignored by Git because they are reproducible and the standalone
Plotly reports embed their JavaScript payload. The commands in the README recreate them.

## Additional MPS timing

The targeted MPS depth-64, 25%-budget comparison did not recover P3: revolve was about 1.5%
slower than uniform, while DP was slower still. The targeted MPS long-sequence matrix also did
not recover P4: fine DP's best matched-cell improvement was 2.32%, below the required 5%. Its
maximum within-cell spread was 15.84%, so the overhead-dominated all-cells-below-3% pivot did not
apply.

MPS allocator occupancy was sampled after every schedule action. Because MPS may defer freeing
allocations across recomputations, each record contains both the logical schedule peak used for
budgeting and a conservative allocator bound based on cumulative scheduled forward bytes plus a
25% rounding margin. All recorded P3/P4 occupancy values were below this bound.

Artifacts:

- `case_studies/cs1_depth_sweep/mps-p3-results.json` and `mps-p3-report.html`
- `case_studies/cs1_depth_sweep/mps-p4-results.json` and `mps-p4-report.html`
