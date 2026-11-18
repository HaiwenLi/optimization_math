"""Independent-optimum comparison matrix; run with ``python -m benchmarks.bench_standard``.

All operator calls made by supported operator solvers (including certificates,
initialization, CG, backtracking and history) pass through CountingOperator.
Dense-only QP, logistic and TV APIs have null call counts, never estimates.
"""
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from benchmarks.standard_problems import make_standard_cases
from optimization_math.certificates_logistic import compute_certificate_logistic
from optimization_math.certificates_qp import compute_certificate_qp
from optimization_math.solvers.admm import solve_admm
from optimization_math.solvers.chambolle_tv import solve_tv_chambolle
from optimization_math.solvers.fista import solve_fista
from optimization_math.solvers.fista_group import solve_fista_group
from optimization_math.solvers.fista_logistic import solve_fista_logistic
from optimization_math.solvers.fista_nuclear import solve_fista_nuclear
from optimization_math.solvers.fista_signed import solve_fista_signed
from optimization_math.solvers.pdhg import solve_pdhg
from optimization_math.solvers.qp_box import solve_qp_box


# 1e-6 certificate accuracy plus independent objective and solution checks.
# The normalized objective check uses max(1, |f*|); the solution check uses
# max(1, ||x*||). These prevent misleading relative errors near zero.
CERT_TOL = 1e-6
OBJECTIVE_TOL = 1e-6
SOLUTION_TOL = 1e-3
# Identity proximal maps are exact to roundoff after one step; two-point TV
# is iterative, so its solution gate is looser while the objective gate stays.
EXACT_PROX_SOLUTION_TOL = 1e-9
TV_SOLUTION_TOL = 1e-5
OPTIONS = dict(max_iter=30000, kkt_tol=CERT_TOL, gap_tol=CERT_TOL, certificate_stride=5)
SCIPY_NATIVE_OPTIONS = dict(ftol=1e-15, gtol=1e-11, maxls=50)
# ADMM inner CG must be far tighter than the outer certificate target.
ADMM_OPTIONS = dict(cg_tol=1e-12, residual_tol=1e-6)
# Exhaustive on purpose: an unknown problem must fail loud, never fall through
# to a generic label or a wrong tolerance.
_CERTIFICATE_SEMANTICS = {
    "signed_lasso": "repository problem-specific KKT residual; relative_gap is normalized",
    "nonnegative_lasso": "repository problem-specific KKT residual; relative_gap is normalized",
    "box_qp": "repository problem-specific KKT residual; relative_gap is normalized",
    "nnls": "repository problem-specific KKT residual; relative_gap is normalized",
    "logistic": "repository problem-specific KKT residual; relative_gap is normalized",
    "tv": "absolute_primal_dual_gap (not a KKT norm)",
    "nuclear": "normalized Frobenius proximal fixed-point residual",
    "group": "repository problem-specific KKT residual; relative_gap is normalized",
}
_SOLUTION_LIMITS = {
    "signed_lasso": SOLUTION_TOL, "nonnegative_lasso": SOLUTION_TOL,
    "box_qp": SOLUTION_TOL, "nnls": SOLUTION_TOL, "logistic": SOLUTION_TOL,
    "tv": TV_SOLUTION_TOL,
    "group": EXACT_PROX_SOLUTION_TOL, "nuclear": EXACT_PROX_SOLUTION_TOL,
}
# Only TV's stationarity field is an absolute gap, so only its KKT gate scales
# with the objective magnitude. Exhaustive for the same fail-loud reason.
_KKT_SCALES_WITH_OBJECTIVE = {problem: problem == "tv"
                              for problem in _SOLUTION_LIMITS}
# Which algorithms compete on which problem (single source for benchmarking
# and dispatch validation; unknown pairs fail loud in _run).
_ALGORITHMS = {
    "signed_lasso": ("fista_signed", "pdhg"),
    "nonnegative_lasso": ("fista", "admm"),
    "box_qp": ("qp_box", "scipy_lbfgsb"), "nnls": ("qp_box", "scipy_lbfgsb"),
    "logistic": ("fista_logistic", "scipy_lbfgsb"),
    "tv": ("chambolle_tv",), "group": ("fista_group",), "nuclear": ("fista_nuclear",),
}


class CountingOperator:
    """Count actual forward and adjoint invocations; never infer from iterations."""
    def __init__(self, operator):
        self.operator = operator
        self.shape = operator.shape
        self.forward_calls = 0
        self.adjoint_calls = 0

    def matvec(self, x):
        self.forward_calls += 1
        return self.operator.matvec(x) if hasattr(self.operator, "matvec") else self.operator @ x

    def rmatvec(self, y):
        self.adjoint_calls += 1
        return self.operator.rmatvec(y) if hasattr(self.operator, "rmatvec") else self.operator.T @ y


class _MatrixIdentity:
    def __init__(self, shape):
        self.shape = shape

    def matvec(self, x):
        return x.ravel()

    def rmatvec(self, y):
        return y.reshape(self.shape)


@dataclass(frozen=True)
class HistoryPoint:
    iteration: int
    objective: float
    kkt_residual: float
    relative_gap: float


@dataclass(frozen=True)
class StandardRun:
    problem: str
    case: str
    seed: int
    conditioning: float
    mu: float
    lam: float
    algorithm: str
    reference_type: str
    status: str
    iterations: int
    time_sec: float
    objective: float
    reference_objective: float
    absolute_objective_error: float
    normalized_objective_error: float
    solution_error: float
    recovery_error: float | None
    kkt_residual: float
    relative_gap: float
    forward_calls: int | None
    adjoint_calls: int | None
    history: tuple[HistoryPoint, ...]
    input_digest: str
    certificate_semantics: str
    configuration: dict[str, float | int]

    def to_dict(self):
        return asdict(self)


def _point(cert):
    return HistoryPoint(int(cert.iteration), float(cert.primal),
                        float(cert.kkt_residual), float(cert.relative_gap))


class _CertificateStop(Exception):
    """Private callback termination carrying the certified iterate, not an error."""
    def __init__(self, x, certificate):
        self.x = x.copy()
        self.certificate = certificate
        super().__init__("common certificate tolerances satisfied")


def _scipy(case, *, max_iter=30000, kkt_tol=CERT_TOL, gap_tol=CERT_TOL,
           certificate_stride=5):
    """Stop on the same initial/scheduled certificate gate as repository solvers.

    Tight native tolerances discourage premature native termination; they are
    not the benchmark convergence target. An intentional callback stop needs
    no native success flag. A native return requires both success and a passing
    final certificate, checked even when it is off the monitoring cadence.
    """
    history = []
    if case.problem == "logistic":
        n = case.A.shape[1]

        def unpack(z):
            return z[:n] - z[n:]

        def fun(z):
            w = unpack(z)
            margins = case.y * (case.A @ w)
            loss = np.mean(np.logaddexp(0., -margins))
            # sigmoid(-margin) = exp(-logaddexp(0, margin)), stable at extremes.
            grad = -case.A.T @ (case.y * np.exp(-np.logaddexp(0., margins))) / len(case.y)
            return float(loss + case.lam*z.sum()), np.r_[grad+case.lam, -grad+case.lam]

        def certificate(w, iteration):
            return compute_certificate_logistic(case.A, case.y, w, case.lam, iteration=iteration)

        start = np.zeros(2*n)
        bounds = [(0., None)] * (2*n)
    else:
        def unpack(z):
            return z

        def fun(z):
            return case.objective(z), case.Q @ z + case.c

        def certificate(w, iteration):
            return compute_certificate_qp(case.Q, case.c, w, case.lower, case.upper, iteration=iteration)

        start = np.zeros_like(case.x_opt)
        bounds = list(zip(case.lower, case.upper))
    def passed(cert):
        return cert.kkt_residual <= kkt_tol and cert.relative_gap <= gap_tol

    cert = certificate(unpack(start), 0)
    history.append(_point(cert))
    if passed(cert):
        return unpack(start), "converged", 0, cert, tuple(history)

    iterations = 0

    def callback(z):
        nonlocal iterations, cert
        iterations += 1
        if iterations % certificate_stride == 0 or iterations == max_iter:
            x = unpack(z)
            cert = certificate(x, iterations)
            history.append(_point(cert))
            if passed(cert):
                raise _CertificateStop(x, cert)

    try:
        result = minimize(fun, start, jac=True, method="L-BFGS-B", bounds=bounds,
                          callback=callback,
                          options=dict(maxiter=max_iter, **SCIPY_NATIVE_OPTIONS))
    except _CertificateStop as stop:
        return stop.x, "converged", stop.certificate.iteration, stop.certificate, tuple(history)
    x = unpack(result.x)
    if history[-1].iteration != result.nit:
        cert = certificate(x, result.nit)
        history.append(_point(cert))
    # A SciPy success flag alone is explicitly insufficient.
    status = ("converged" if result.success and passed(cert)
              else f"failed: {result.message}")
    return x, status, int(result.nit), cert, tuple(history)


def _run(case, algorithm):
    if algorithm not in _ALGORITHMS[case.problem]:
        raise ValueError(f"algorithm {algorithm!r} is not a competitor for "
                         f"problem {case.problem!r}")
    op = None
    config = dict(OPTIONS)
    if algorithm == "admm":
        config.update(ADMM_OPTIONS)
    elif algorithm == "scipy_lbfgsb":
        config.update(maxiter=OPTIONS["max_iter"], **SCIPY_NATIVE_OPTIONS)
    elif algorithm == "chambolle_tv":
        config = dict(tol=CERT_TOL, max_iter=30000)
    start = perf_counter()
    if algorithm == "scipy_lbfgsb":
        x, status, iterations, cert, history = _scipy(case, **OPTIONS)
    else:
        if case.problem in ("signed_lasso", "nonnegative_lasso", "group"):
            op = CountingOperator(case.A)
            kwargs = dict(mu=case.mu, b=case.b, x0=np.zeros_like(case.x_opt), **OPTIONS)
            solvers = dict(fista=solve_fista, fista_signed=solve_fista_signed,
                           pdhg=solve_pdhg, admm=solve_admm)
            if case.problem == "group":
                result = solve_fista_group(op, case.y, case.lam, case.groups, **kwargs)
            else:
                if algorithm == "admm":
                    kwargs.update(ADMM_OPTIONS)
                result = solvers[algorithm](op, case.y, case.lam, **kwargs)
            x = result.x
        elif case.problem in ("box_qp", "nnls"):
            result = solve_qp_box(case.Q, case.c, case.lower, case.upper,
                                  x0=np.zeros_like(case.x_opt), **OPTIONS)
            x = result.x
        elif case.problem == "logistic":
            result = solve_fista_logistic(case.A, case.y, case.lam,
                                         w0=np.zeros_like(case.x_opt), **OPTIONS)
            x = result.w
        elif case.problem == "tv":
            result = solve_tv_chambolle(case.y, case.lam, tol=CERT_TOL, max_iter=30000)
            x = result.x
        elif case.problem == "nuclear":
            op = CountingOperator(_MatrixIdentity(case.x_opt.shape))
            result = solve_fista_nuclear(op, case.y, case.lam,
                                        X0=np.zeros_like(case.x_opt), **OPTIONS)
            x = result.X
        else:
            raise ValueError(f"unknown problem: {case.problem}")
        status, iterations, cert = result.status, result.iterations, result.certificate
        certificates = result.certificates if algorithm == "admm" else result.history
        history = tuple(_point(c) for c in certificates)
    elapsed = perf_counter() - start
    objective, reference = case.objective(x), case.objective(case.x_opt)
    error = abs(objective - reference)
    semantics = _CERTIFICATE_SEMANTICS[case.problem]
    return StandardRun(
        case.problem, case.name, case.seed, case.conditioning, case.mu, case.lam,
        algorithm, case.reference_type, status, int(iterations), elapsed,
        objective, reference, error, error/max(1., abs(reference)),
        float(np.linalg.norm(x-case.x_opt)/max(1., np.linalg.norm(case.x_opt))),
        None if case.truth is None else float(np.linalg.norm(x-case.truth)/max(1., np.linalg.norm(case.truth))),
        float(cert.kkt_residual), float(cert.relative_gap),
        None if op is None else op.forward_calls, None if op is None else op.adjoint_calls,
        history, case.input_digest, semantics, dict(sorted(config.items())),
    )


def benchmark_standard(seeds=(0, 1, 2), conditions=(1., 10., 100.)):
    """Return frozen StandardRun records; ``row.to_dict()`` is JSON serializable.

    Does not suppress failed runs or assert, so callers can inspect/report failures.
    Independent exact references are metadata, not timed iterative competitors.
    """
    algorithms = _ALGORITHMS
    return [_run(case, algorithm) for case in make_standard_cases(seeds, conditions)
            for algorithm in algorithms[case.problem]]


def _assert_benchmark(rows):
    assert rows, "empty benchmark matrix"
    for row in rows:
        label = f"{row.case}/{row.algorithm}"
        try:
            solution_limit = _SOLUTION_LIMITS[row.problem]
        except KeyError:
            raise AssertionError(
                f"{label}: unknown problem {row.problem!r}; add it to "
                "_SOLUTION_LIMITS, _CERTIFICATE_SEMANTICS and _ALGORITHMS "
                "in benchmarks/bench_standard.py") from None
        values = (row.objective, row.reference_objective, row.absolute_objective_error,
                  row.normalized_objective_error, row.solution_error, row.kkt_residual,
                  row.relative_gap, row.time_sec)
        assert np.all(np.isfinite(values)), f"{label}: nonfinite metrics"
        assert row.status == "converged", f"{label}: {row.status}"
        assert row.time_sec >= 0 and row.iterations >= 0, label
        assert row.normalized_objective_error <= OBJECTIVE_TOL, f"{label}: objective error"
        assert row.solution_error <= solution_limit, f"{label}: solution error {row.solution_error}"
        assert -1e-12 <= row.relative_gap <= CERT_TOL, f"{label}: gap {row.relative_gap}"
        # TV stops on relative gap; its kkt_residual is the unnormalized gap.
        limit_scale = max(1., abs(row.objective)) if _KKT_SCALES_WITH_OBJECTIVE[row.problem] else 1.
        assert 0 <= row.kkt_residual <= CERT_TOL * limit_scale, \
            f"{label}: certificate {row.kkt_residual}"
        assert row.history and all(np.isfinite(
            (h.objective, h.kkt_residual, h.relative_gap)).all() for h in row.history), label
        if row.recovery_error is not None:
            assert np.isfinite(row.recovery_error), label


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="benchmarks/out/standard")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    # anchor catches Windows drive-absolute/relative and UNC forms that
    # is_absolute() alone misses; '.' would overwrite repository-root files;
    # whitespace-only or padded parts only create oddly named directories.
    if (out.anchor or out.is_absolute() or ".." in out.parts or out == Path(".")
            or any(not part.strip() or part != part.strip()
                   or part.endswith((".", " ")) for part in out.parts)):
        parser.error("--out-dir must be a relative subdirectory inside the "
                     "repository, without '..', '.', absolute forms or "
                     "whitespace-padded components")
    from benchmarks.standard_report import write_standard_report

    rows = benchmark_standard()
    serializable = [row.to_dict() for row in rows]
    write_standard_report(rows, out)
    _assert_benchmark(rows)
    print(f"{len(rows)} standard comparisons passed; report: {out / 'README.md'}")
    return serializable


if __name__ == "__main__":
    main()
