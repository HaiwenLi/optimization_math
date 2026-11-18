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

## EGM Goldfish review round (2026-11-18)

A four-persona fresh-context Goldfish review (EGM protocol: hostile design
critic, structural, docs-vs-code, zoom-out architect) audited the whole tree
and produced 1 Blocker, 5 Major, 12 Nit findings. All were reproduced or
refuted by probe before action; items that did not reproduce were dropped.

Fixed with red-to-green regressions (31 new regression cases in
`tests/test_egm_regressions.py` — 28 platform-independent plus 3
Windows-only argument cases; suite passes with warnings as errors, 303 on
this machine; matrix still 177/177; snapshot resynced):

- **QP false optimality on indefinite/asymmetric Q (Blocker).**
  `solve_qp_box(diag(1,-1), 0)` returned `converged` with gap exactly zero on
  an unbounded problem, and an asymmetric Q received a finite "dual bound"
  from the minorant of a different matrix (eigvalsh reads one triangle).
  Both entry points now share `_validate_qp_hessian`: symmetric up to a
  scale-dependent roundoff allowance, positive semidefinite up to the same
  eigensolver allowance; roundoff-level cases and singular PSD stay accepted.
- **`solve_fista_group` lipschitz contract divergence (Major).** The default
  estimate added mu while a user-supplied value was used raw, so the sibling
  convention (`lipschitz = ||A||^2`) diverged to overflow for mu=20. Supplied
  values now follow the shared `||A||^2` convention with mu added internally;
  every solver docstring now states its `lipschitz` units and `b` support.
- **Permissive benchmark defaults (Major).** The certificate-semantics label
  and the solution-tolerance gate were hand-maintained conditionals whose
  else-branches silently accepted unknown problems (mislabeling artifacts, or
  a 1000x looser solution gate). Both are now exhaustive mappings that raise
  `KeyError` on unknown problems.
- **`--out-dir` guard (Major).** `Path('/abs').is_absolute()` is False on
  Windows, so `/x` escaped to the drive root, `C:x` was drive-relative, and
  `--out-dir .` would overwrite repository-root files. The guard now rejects
  any anchored/absolute path, `..`, and `.`, with argument-rejection tests.
- **Nuclear module (Major + Nit).** The public docstring claimed a "spectral
  norm of the complement" residual the code never computed; it now states the
  normalized Frobenius proximal fixed-point residual, and the protocol
  document warns about nuclear alongside TV. Matrix shapes of `X`/`X0` are
  validated at entry instead of surfacing as broadcast errors.
- **README navigation (Major).** The directory tree omitted the three
  standard-comparison modules and their tests; now listed.
- **Reporter robustness/usability (Nits).** Special tolerances and the ADMM
  configuration are single-sourced constants; the operator-calls title
  derives its lambda from the data; `configuration` serializes as a JSON
  object; mixed counted/uncounted groups render `mixed (n/a)` instead of
  crashing; matplotlib joined the test extra so report regressions cannot
  silently skip; the legacy signed-LASSO figure notes the two curves are
  different objectives; a dangling sentence fragment in the protocol doc was
  repaired and the group-dual survey documents the elastic-net/offset form.

Declined as accepted trade-offs (recorded, not lost): a per-problem semantics
tag on the `Certificate` dataclass (public API churn; consumers now fail
loud), unifying the nine result dataclasses (all current misreads fail loud),
consolidating the seven input validators (refactor without a failing
regression), and moving review diffs out of the ignored out directory.

Two reviewers independently verified the counter infrastructure has no bypass
and that fixture references are genuinely independent re-derivations; the
docs-vs-code auditor recomputed every results-table cell from tracked
metrics.json and matched all 13 rows.

## EGM Round 2 verification (2026-11-18)

A second fresh-context swarm re-audited the fixed tree. All Round 1 fixes
were verified by probe: the QP validator's acceptance boundary sits exactly
at the 8*eps*n*max|eigenvalue| allowance (accepts -3.5e-15, rejects -1e-14);
the group-FISTA step change altered no existing caller (estimate path
byte-identical; all explicit-lipschitz callers use mu=0 or oversized values);
the exhaustive mappings cover exactly the eight fixture problems; the
out-dir guard rejects every escape form tried on Windows; regenerated PNGs
are byte-identical and the snapshot differs from fresh generation only in
timing columns.

Round 2 found no Blockers and one Major: the mixed-count regression test
passed on pre-fix code (its two rows formed separate all-uncounted groups,
and the assertion matched boilerplate text). The test now builds a genuinely
mixed group in both orderings and asserts the `mixed (n/a)` cell. Round 2
hardening applied on top: exhaustive KKT-gate mapping, module-level
algorithm-competitor map with dispatch validation (misspelled algorithm
names fail loud), actionable unknown-problem message naming the dicts to
update, whitespace/trailing-dot out-dir component rejection, fail-loud
color lookup, and fixture problem-name validation in `StandardCase`.
Remaining Round 2 nits recorded as accepted: the round-2 review diff
excluded the untracked test file and resynced snapshot by construction
(the commit contains both), and a boundary-tightening of the QP acceptance
test documents the exact allowance.

Round 2 solver-math verification also probed and cleared the TV default
step size (monotone gap decay to n=20000), dual-bound validity on 1600+
random cases across all six certificates, screening safety (200 trials,
zero unsafe screens), and QP optima against SciPy on 300 random problems
(worst excess 1.8e-15). Final touch-ups from that pass: the nuclear
certificate now validates y's size explicitly (loudness no longer depends
on a downstream dot), a dead SVD call was removed, the group-dual survey
and code comment label the dual point as nu = theta r (the value was
always computed at that point), and the TV/PDHG docstrings state their
(no) step knobs explicitly.

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
