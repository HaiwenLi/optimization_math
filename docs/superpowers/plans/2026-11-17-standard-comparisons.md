# Standard optimization comparisons implementation plan

**Approved design:** extend the existing benchmark flow, not solver APIs.
Use independent known optima, compare applicable algorithms on identical
instances and tolerances, fix demonstrated bugs, regenerate plots/reports,
and commit locally without pushing.

## Constraints

- Keep runtime dependencies unchanged; benchmarks may use NumPy, SciPy and Matplotlib.
- Only relative paths and non-sensitive synthetic data in tracked files.
- Distinguish a data-generating signal from a regularized optimum.
- Count actual forward/adjoint calls, including initialization and certificates.
- Use seeds 0, 1, 2 and Hessian condition targets 1, 10, 100 where applicable.
- No worktrees on this TSD machine; isolate on a local branch.
- No push. Do not claim universal absence of bugs.

## Task 1: Independent fixtures and comparison runner

- [x] Add `benchmarks/standard_problems.py`: immutable cases with independent
  objective, exact optimum and optional generating truth. Cover nonnegative
  and signed LASSO/elastic net, box QP/NNLS, normalized sparse logistic,
  two-point TV, group shrinkage and spectral shrinkage.
- [x] Add `benchmarks/bench_standard.py`: real operator counters, timing,
  convergence/certificate checks, optimum error separate from recovery
  error, and solver dispatch. Compare FISTA/ADMM on nonnegative problems,
  signed FISTA/PDHG on signed problems, projected gradient/SciPy on QP,
  FISTA/SciPy on logistic, existing structured/TV solvers with exact oracles.
- [x] Tests first in `tests/test_standard_comparisons.py`: literal analytic
  anchors, oracle KKT checks, counter accuracy, identical inputs, empty-run
  rejection, convergence and objective/solution accuracy across the matrix.
- [x] Observe red tests, implement, run green. Fix production bugs only
  with a failing minimal regression and record each root cause.

## Task 2: Evidence and documentation

- [x] Add JSON/Markdown report and convergence plots through the runner.
  Report platform/library versions, seeds, tolerances, reference provenance
  and timing limitations. Non-applicable operator counts are null, not zero.
- [x] Run the full matrix and the full pytest suite with warnings as errors.
- [x] Inspect rendered plots and publish a tracked summary with relative
  image links. Keep volatile bulk output under ignored `benchmarks/out/`.
- [x] Update README with reproduction commands and interpretation.

## Task 3: Independent review and local delivery

- [x] Fresh-context review of oracle independence, comparison fairness,
  stopping honesty and any production fixes; resolve objective findings.
- [x] Run graph update, staged whitespace/privacy checks and fresh tests.
- [x] Commit only task files locally; verify no push and a clean work tree.

## Task interfaces

Task 1 owns the fixtures, runner and tests. Task 2 consumes their results
without changing solver semantics. Task 3 consumes the complete diff and
verification evidence. All tasks share the approved constraints above;
the only cross-task mutable files are the runner and this progress record.

## Progress

- Preflight: existing `metrics.py` remains unchanged to avoid changing legacy
  benchmark accounting. New measured-call fields belong to the new runner.
- Preflight: analytic references are reference values, not invented iterative
  competitors; structured cases honestly have one implemented solver.
- Confirmed additional regression: incomplete group `[0]` on identity design
  with `y=(0,2)` falsely reports convergence at zero, objective 2 instead of 0.
  Reject non-partition groups in both the solver and public certificate.
- Confirmed additional regression: two-point TV with `y=(0,2)+1e8` and
  `lam_tv=0.25` reaches the exact solution but reports `max_iter` with gap
  `0.4375`; subtracting large squared norms destroys the dual objective.
  Use the algebraically equivalent difference-based dual formula and test
  invariance under a large constant offset.
- Task 1: complete; independent reference review approved after correcting
  cold-import timing and aligning SciPy's certificate cadence/stopping gate.
- Production fixes: complete; 75 structured regressions passed, scoped
  independent review READY.
- Task 2: complete; 177/177 runs passed, full suite 260 passed with warnings
  as errors. Three rendered plots inspected; tracked snapshot hashes match
  all five generated artifacts (Markdown, compact JSON, three PNGs).
- Graph navigation index updated locally; generated graph artifacts remain
  excluded from commits.
- Final review: one P2 remained (nonfinite report values crashed plot
  rendering and could strand a stale README). Fixed with parametrized
  NaN/inf regressions (red 12 failed -> green 16/16 in the report module),
  explicit nonfinite annotations instead of a 1e-16 floor, `n/a (nonfinite)`
  Markdown cells and JSON nulls; full suite 272 passed, matrix 177/177,
  snapshot resynced (PNGs byte-identical, timings refreshed).
