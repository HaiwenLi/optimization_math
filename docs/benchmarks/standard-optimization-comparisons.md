# Standard optimization comparisons

## What is ground truth here?

The suite distinguishes two different targets:

- **Optimization truth**: the minimizer of the stated, regularized
  objective. Constructed analytically or from explicit KKT conditions,
  independently of the solvers under test.
- **Generating truth**: the synthetic signal used to produce noisy
  observations, when such a signal exists. Regularization and noise can
  make the objective's optimum differ from that signal.

Objective accuracy and solution accuracy are measured against optimization
truth. Recovery error against generating truth is a separate quantity and
is not used to decide whether an optimizer solved the problem correctly.
Manufactured-KKT problems need not have a generating truth.

## Comparison protocol

Only algorithms solving the **same objective and constraints** share a
comparison. In particular, nonnegative ADMM is not compared to signed
LASSO as if they were interchangeable.

| Problem | Implemented competitors | Independent reference |
|---|---|---|
| Nonnegative LASSO / elastic net | FISTA, ADMM | Orthogonal-design shrinkage; manufactured KKT |
| Signed LASSO / elastic net | FISTA, PDHG | Orthogonal-design soft threshold; manufactured KKT |
| Box QP / NNLS | Projected gradient, SciPy optimizer | Prescribed solution with explicit KKT multipliers |
| Sparse logistic regression | FISTA, SciPy optimizer | Separable normalized-logistic optimum |
| Two-point TV denoising | Chambolle projection | Exact fused/unfused two-point solution |
| Group LASSO | Group FISTA | Exact disjoint-group shrinkage |
| Nuclear norm denoising | Nuclear FISTA | Prescribed singular vectors and thresholded singular values |

An analytic formula is a reference, not an invented competing iterative
solver. Structured/TV cases with only one implemented algorithm therefore
validate correctness but do not establish a speed ranking.

For the small deterministic instances, seeds are 0, 1 and 2 and prescribed
conditioning levels are 1, 10 and 100 where meaningful. Identity proximal
and two-point cases do not have an independently variable condition
number. Logistic design scaling is not a measurement of the optimum's
logistic Hessian condition number.

## Measurements and interpretation

- Independent objective error, normalized by `max(1, abs(reference_objective))`.
- Distance to the unique reference optimum, normalized by
  `max(1, norm(reference_solution))`.
- Generating-signal recovery error, only when that target is available.
- Certificate stationarity signal and relative gap.
- Iterations and wall-clock seconds.
- **Measured** forward and adjoint calls where an operator interface exists,
  including initialization, power estimation, line search, CG and certificate
  evaluations. Dense-only implementations have unavailable counts rather
  than fabricated zero counts.

The selected references have unique optima; convexity alone would not
justify comparing solution vectors in a nonunique problem.

TV's certificate stationarity field is an absolute primal-dual gap, not
the same KKT norm used by LASSO. SciPy optimizer success is not sufficient:
its result must also meet the comparison's accuracy and certificate checks.
Runtime is descriptive, machine-dependent evidence, not a pass/fail gate.
Small cases do not establish large-scale performance.

## Reproduction

Install source-checkout benchmark dependencies:

```powershell
python -m pip install -e ".[test,benchmark]"
python -m benchmarks.bench_standard
```

Machine-readable runs, the Markdown table and convergence plots are written
under `benchmarks/out/standard/`. Full output is intentionally ignored by
Git; the tracked summary and selected plots document the local experiment.
