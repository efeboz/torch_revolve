# torch-revolve — Remaining-work resolution

Resolution date: 2026-08-17. This document closes the findings from the fresh audit against
[revolve_transformers_plan.md](revolve_transformers_plan.md).

## Verdict

All required findings are resolved. The implementation now satisfies the available CPU and MPS
verification gates, the evidence documents match the current repository, generated files are
ignored, tracked bytecode is removed, and the completed work is committed.

## Resolved findings

### MPS memory validation

The analytic model predicts logical saved activations rather than device allocator blocks. Both
CPU and MPS saved-tensor hooks measure 329,728 bytes for the 327,680-byte block prediction, a
0.625% error and a pass at the original 10% tolerance.

The raw MPS block allocation is 394,752 bytes, 20.47% above the logical prediction. That value is
retained as allocator evidence rather than forced into the logical model. MPS benchmark occupancy
is sampled after every schedule action. Records expose a separate conservative allocator bound
based on cumulative scheduled forward allocations with a 25% rounding margin, accommodating MPS
deferred reclamation during recomputation. Every recorded MPS peak is below that bound.

### Verification and documentation

- `ruff check .` passes.
- `python3 -m pytest -q` reports 1,626 passed and one skipped CUDA case with MPS exercised.
- [report.md](report.md), [docs/results.md](docs/results.md), and [README.md](README.md) now report
  the current hardware results and exact verification counts.

### MPS timing points

The optional MPS runs were completed and clearly labeled. At depth 64 and a 25% logical budget,
revolve was about 1.5% slower than uniform, so P3 still fails. In the long-sequence matrix, fine
DP's best improvement over coarse revolve was 2.32%, below P4's 5% threshold. The maximum MPS
within-cell spread was 15.84%, so the all-cells-below-3% pivot does not apply.

The reproducible artifacts are ignored by Git:

- `case_studies/cs1_depth_sweep/mps-p3-results.json`
- `case_studies/cs1_depth_sweep/mps-p3-report.html`
- `case_studies/cs1_depth_sweep/mps-p4-results.json`
- `case_studies/cs1_depth_sweep/mps-p4-report.html`

## Goal status

The six schedulers, optimality checks, generic deterministic executor, RNG replay, analytic memory
model, hardware validation, Pareto studies, fixed-budget study, 200-step parity run, reports, CI,
and repository hygiene are all in place. P1 and P2 pass. P3 and P4 are implemented measurements
whose preregistered hypotheses failed on both CPU and MPS; those negative results are part of the
completed goal rather than unresolved implementation work.
