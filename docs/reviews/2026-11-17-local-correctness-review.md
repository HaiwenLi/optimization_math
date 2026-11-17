# Local correctness review

- Baseline: `e14e4ee`.
- Scope: repository source, convergence certificates, analytic edge cases,
  benchmark assertions/report generators, and installation metadata.
- Delivery: local commit only; no push or remote operations.
- Independent review: a fresh-context reviewer checked the numerical fixes
  and returned READY after the singular-Q correction.

## Confirmed findings and fixes

### R1 — P1: screening used a gap belonging to a different dual point

`src/optimization_math/screening.py`, `compute_safe_screen`.

For `A=[[1]], y=[2], x=[1], lam=1, theta=0.5`, with `gap` omitted, the
baseline screened the only coordinate even though the exact optimum is
`x*=1`. The default certificate had zero gap for its own `theta=1`, not
for the supplied dual point. The correct supplied-point gap is `5/8`.

Fixed by computing the gap for the actual dual point, checking theta's
range/dual feasibility, and validating finite backgrounds even on cached
input paths. An explicit cached gap remains a caller-supplied valid
**absolute** gap upper bound for the same problem, iterate and dual point.

### R2 — P1: QP infinite bounds created a false lower bound

`src/optimization_math/certificates_qp.py`, `compute_certificate_qp`.

For `Q=[[1e-12]], c=[-1e-6]`, without bounds, the baseline solver returned
`converged` at iteration zero with `x=0` and zero gap. In fact `x*=1e6`
and `f*=-0.5`. Replacing infinite bounds by zero created multipliers for
nonexistent constraints and invalidated the lower-bound claim.

Replaced this construction with a separable quadratic minorant over the
actual box. A scale-dependent eigensolver roundoff allowance prevents a
singular PSD matrix from being treated as strictly positive definite.
Regression cases include the exact null direction of
`Q=[[1,3],[3,9]], c=[3,-1]`, both unconstrained and with one-sided bounds.
NaN and impossible infinite endpoints are rejected.

Limitation: for numerically singular Q with unbounded directions, the
linear minorant can return `dual=-inf, gap=inf` even when the actual QP
is bounded. This is deliberately conservative, not an unboundedness
diagnosis. Dense eigenvalue decomposition adds certificate cost; no
large-scale performance claim is made.

### R3 — P2: logistic dual normalization was not tight at zero optima

`src/optimization_math/certificates_logistic.py`.

For two identical unit samples with positive labels and `lam=1`, `w=0`
is optimal, but the baseline reported gap `log(2)` and exhausted
`max_iter`. The loss is a sample **mean**: the residual-derived dual
must start at `theta=1/N`, not `1`. Corrected normalization and added
multi-sample exact-optimum tests.

### R4 — P2: zero logistic design divided by zero with a warm start

`src/optimization_math/solvers/fista_logistic.py`.

`X=[[0]], y=[1], lam=1, w0=[1]` raised `ZeroDivisionError` because the
automatic Lipschitz value was zero. A positive fallback now permits the
soft-threshold step to reach the exact zero optimum. Empty sample or
feature dimensions are rejected by both solver and certificate.

### R5 — P2: benchmark artifacts retained incorrect formulas and claims

`benchmarks/bench_screening.py`, `benchmarks/bench_pdhg.py`,
`benchmarks/bench_qp.py`, and corresponding survey documents.

- Screening plots still divided the radius by theta and labeled an
  absolute gap as relative. They now call the production screening rule
  and sweep only valid gap upper bounds for the fixed dual point.
- The PDHG report generator still claimed twice FISTA's core operator-pair
  count. Both have one core pair per iteration; tables are estimates
  excluding certificate, backtracking and restart overhead, not measured
  total calls.
- QP report wording incorrectly claimed machine-precision recovery.
  Rerun errors are approximately `3.1e-5` to `7.3e-5` against the SciPy
  reference.
- Corrected claims that all convex optima are unique and that ordinary
  LASSO FISTA requires an inverse operator. README now reflects the
  repository's already-established public status.

### R6 — P2: the previous radius regression was non-discriminating

`tests/test_screening.py`, `test_screen_matches_explicit_radius_formula`.

The random zero-iterate case did not ensure the correct and old radii
gave different masks. Replaced it with the hand-calculated
`A=diag(1,0.75), y=(2,0), x=(0,0), lam=1` fixture:
correct mask `(False, True)`, old-formula mask `(False, False)`.
An in-memory mutation check verified that distinction.

### R7 — P2: documented test installation omitted a required dependency

`pyproject.toml`.

QP benchmark tests import SciPy for their independent reference, but the
test extra only installed pytest. Added SciPy to the test extra and a
benchmark extra containing SciPy and Matplotlib. No new runtime dependency.

## Verification

- Baseline: **135 passed**.
- Final numerical/benchmark suite: **162 passed**, with warnings as errors.
- Regression tests were run red before the behavioral fixes.
- Re-ran screening, PDHG, QP and logistic benchmark entrypoints; their
  assertions passed. Regenerated reports and plots remain under ignored
  `benchmarks/out/`.
- Test invocation used:
  `python -c "import benchmarks, pytest; raise SystemExit(pytest.main(['-q','-W','error']))"`.
  In this harness, direct `python -m pytest -q` failed benchmark-package
  imports; explicitly importing the local package avoids that execution
  environment issue. A clean-environment installation was not performed.
- This is an evidence-based correctness review, not a proof of exhaustive
  correctness or validation of deferred imaging/large-scale applications.
