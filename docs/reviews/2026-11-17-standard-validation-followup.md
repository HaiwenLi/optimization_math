# Standard validation follow-up

This extends the earlier local correctness review (`934627c`). All work is
local; no push is authorized.

## Demonstrated production defects

### Group layouts could falsely certify an unsolved problem

`solve_fista_group(I, [0,2], 1, [[0]])` previously returned `converged`
at zero with primal and dual both 2. With the stated partial-group
penalty the optimum is `(0,2)`, objective 0: the certificate ignored the
unpenalized coordinate.

The implemented proximal rule and dual construction support only complete,
disjoint group partitions. Solver and certificate now share a validator
that rejects omissions, overlaps, duplicate indices, fractional/float or
boolean indices, empty groups and invalid bounds. Arbitrarily ordered
complete integer partitions remain valid. Overlapping-group optimization
has not been implemented or silently approximated.

### TV dual cancellation rejected exact solutions with large DC offsets

For `y=(0,2)+offset`, `lambda_tv=0.25`, the exact optimum is
`(0.25,1.75)+offset`. At offsets `1e8` and `1e12`, the earlier dual
subtracted two large squared norms and returned a spurious gap `0.4375`,
causing `max_iter` despite reaching the optimum.

The equivalent implementation now uses
`lambda_tv*dot(Dy,p) - 0.5*norm(lambda_tv*D.T@p)**2`. The offset disappears
before evaluation. Public certificate inputs are checked for shape,
finiteness and dual feasibility; only tiny dual-bound roundoff is clipped.
Regression tests compare the **absolute**, offset-subtracted solution;
a relative tolerance against the large offset would have hidden errors.

The structured-regression task observed **45 failed / 30 passed** before
the fixes and **75 passed** afterwards.

## Report robustness defect caught in final review

A failed run with nonfinite `normalized_objective_error`/`solution_error`
(NaN or inf) crashed `_plot_accuracy` at render time ("Data cannot be
log-scaled because all values are <= 0"), so Markdown generation was never
reached and a previous README could survive as stale evidence.

The reporter now filters nonfinite measurements out of every plotted range
and renders them as explicit annotations ("nonfinite error, not plotted",
"no finite data") instead of flooring them to 1e-16, which would look
deceptively perfect. The same filtering applies to convergence histories:
nonfinite checkpoints are skipped and flagged, and an entirely empty panel
gets an explicit "no finite data" label with a fixed log range. Markdown
error cells print `n/a (nonfinite)` rather than a formatted NaN, JSON keeps
mapping nonfinite floats to `null`, and failed verdicts stay failed.
Parametrized regressions (NaN and inf, all-invalid and mixed groups,
invalid/mixed/empty histories) passed 12/12 after the fix; the full suite
reached 272 passed with warnings as errors and the matrix still passed
177/177.

## Comparison fairness defects caught in review

- The first SciPy call included a cold `scipy.optimize` import inside
  timing. Imports are now outside all measured solver runs.
- SciPy originally continued to tighter native tolerances and checked a
  certificate every iteration, while repository solvers used the common
  `1e-6` gate at stride 5. It now checks the initial point, uses the same
  monitoring cadence and intentionally stops when the scheduled common
  certificate passes. Native termination still requires a valid final
  certificate. ADMM's additional residual gate and residual-triggered
  certificate work remain part of its actual measured cost.

Targeted tests verify initial zero-optimum termination, scheduled checks,
intentional callback exit, recorded configuration and propagation of
unrelated errors. Independent reviewers approved both the mathematical
references and the scoped production fixes after these corrections.

## Evidence

- [Reference derivations and protocol](../benchmarks/standard-optimization-comparisons.md).
- [177-run results, compact per-run metrics and plots](../benchmarks/2026-11-17-standard-results/README.md).
- Frozen fixtures use independent analytic or explicit KKT solutions,
  never a competing solver as ground truth.
- Detailed histories are regenerated under ignored `benchmarks/out/standard/`.
- Nonfinite failed report measurements are serialized as JSON `null` with
  explicit failed verdicts, not silently counted as passed.

This is a small-instance correctness and comparison suite. It does not
prove universal absence of bugs or validate large-scale imaging workloads.

Legacy TV/structured benchmarks were also rerun after the fixes. In the
longer legacy TV example, `lambda_tv=3` still reaches its 20,000-iteration
cap with gap approximately `5.44e-4`; it correctly returns `max_iter`,
not a false convergence claim. Increasing the iteration budget or adding
a faster TV algorithm is a performance follow-up, not evidence that every
finite budget must converge. This legacy run is separate from the 177-run
known-optimum comparison matrix.
