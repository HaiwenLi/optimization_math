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
| Nonnegative LASSO / elastic net | FISTA, ADMM | Orthogonal-design shrinkage |
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

Default vector problems have 8 variables; nuclear denoising is 4x4 and TV
uses two points. These sizes make independent exact validation inexpensive;
they are not evidence of large-scale throughput or imaging reconstruction
performance.

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
the returned point is also checked against the known optimum and the
applicable certificate.

## Independent reference construction

### Orthogonal-design LASSO and elastic net

For `A=U diag(d)` with orthogonal `U`, set `q=A.T@(y-b)`. The signed
optimum is `sign(q)*max(abs(q)-lambda,0)/(d*d+mu)`; the nonnegative
optimum is `max(q-lambda,0)/(d*d+mu)`. These are scalar minimizations,
not calls to a solver's proximal implementation.

For correlated full-rank manufactured cases, prescribe `x_star` and a
subgradient `s` with `s_i=sign(x_star_i)` on nonzeros and `|s_i|<1`
on zeros. Choose a residual satisfying
`A.T@r=-lambda*s-mu*x_star`, then `y=A@x_star+b-r`. These equations
certify the prescribed optimum directly.

### Box QP and NNLS

For positive definite `Q`, prescribe a feasible `x_star` and a gradient
`g`: positive on lower-bound coordinates, negative on upper-bound
coordinates, zero on interior coordinates. Set `c=g-Q@x_star`. The
box KKT conditions are then satisfied by construction. SciPy is an
independent competitor, not the source of this suite's ground truth.

### Normalized logistic loss

For `X=diag(a)`, labels `y_i` in `{-1,+1}`, `N` samples and positive
`a_i`, each coefficient minimizes
`log(1+exp(-y_i*a_i*w_i))/N + lambda*abs(w_i)`.
Its optimum is zero when `a_i <= 2*N*lambda`; otherwise it is
`y_i*log(a_i/(N*lambda)-1)/a_i`. The factor `N` is essential.

### TV and structured proximal problems

For two-point TV with observations `y_0,y_1`, let
`delta=(y_1-y_0)/2` and `mean=(y_0+y_1)/2`. Set
`h=sign(delta)*max(abs(delta)-lambda_tv,0)`. The optimum is
`(mean-h,mean+h)`, covering both fused and unfused regimes.

For an identity group-denoising problem, each disjoint block has
optimum `max(1-lambda/norm(y_group),0)*y_group`; a zero block stays
zero. With scalar design scale `a` and ridge weight `mu`, apply
block thresholding to `a*y_group` and divide by `a*a+mu`.

For nuclear denoising, construct the observation from known orthogonal
singular vectors and prescribed singular values `s`. Its reference is
constructed from those vectors with singular values
`max(s-lambda,0)`, independently of the production singular-threshold
helper.
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
