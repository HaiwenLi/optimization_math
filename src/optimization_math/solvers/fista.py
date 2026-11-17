from dataclasses import dataclass
from typing import Literal

import numpy as np

from optimization_math.operators import as_linear_operator
from optimization_math.certificates import (
    Certificate,
    _vector,
    compute_certificate,
)
from optimization_math.solvers._common import (
    _estimate_lipschitz,
    _positive_float,
    _positive_int,
    _validate_common,
)


@dataclass(frozen=True)
class FistaResult:
    x: np.ndarray
    status: Literal["converged", "max_iter"]
    iterations: int
    total_iterations: int
    lipschitz: float
    restart_count: int
    certificate: Certificate
    history: tuple[Certificate, ...]


def _nonnegative_elastic_net_prox(z, *, step, lam, mu):
    return np.maximum(0.0, z - step * lam) / (1.0 + step * mu)


def lambda_max(A, y, *, b=None):
    op = as_linear_operator(A)
    y = _vector(y, "y")
    b_vec = np.zeros_like(y) if b is None else _vector(b, "b")
    if y.shape != (op.shape[0],) or b_vec.shape != y.shape:
        raise ValueError("y and b must match the operator output dimension")
    g0 = op.rmatvec(b_vec - y)
    return float(max(0.0, -np.min(g0)))


def _gradient_restart_needed(grad, x_new, x_anchor):
    return bool(np.dot(grad, x_new - x_anchor) > 0.0)


def _backtracked_prox(op, y, b_vec, z, grad, L, lam, mu, enabled, f_z):
    # Recovery envelope: L may double at most 60 times per call (growth cap ~2**60),
    # so a supplied `lipschitz` more than about 2**60 below the stable value exhausts
    # the budget and raises FloatingPointError rather than looping forever.
    for _ in range(60):
        x_new = _nonnegative_elastic_net_prox(z - grad / L, step=1.0 / L, lam=lam, mu=mu)
        if not enabled:
            return x_new, L
        delta = x_new - z
        r_new = op.matvec(x_new) + b_vec - y
        f_new = 0.5 * np.dot(r_new, r_new)
        upper = f_z + np.dot(grad, delta) + 0.5 * L * np.dot(delta, delta)
        tolerance = 1e-12 * max(1.0, abs(f_z), abs(upper))
        if f_new <= upper + tolerance:
            return x_new, L
        L *= 2.0
    raise FloatingPointError("Backtracking failed to find a finite stable step")


def _run_stage(op, y, b_vec, x, *, lam, mu, L, restart, backtracking,
               max_iter, kkt_tol, gap_tol, certificate_stride, stage):
    z = x.copy()
    momentum = 1.0
    restart_count = 0
    history = [compute_certificate(op, y, x, lam, mu=mu, b=b_vec, stage=stage, iteration=0)]
    status = "max_iter"
    iterations = 0

    if history[-1].kkt_residual <= kkt_tol and history[-1].relative_gap <= gap_tol:
        status = "converged"
    else:
        for k in range(1, max_iter + 1):
            r_z = op.matvec(z) + b_vec - y
            grad = op.rmatvec(r_z)
            x_new, L = _backtracked_prox(
                op, y, b_vec, z, grad, L, lam, mu, backtracking, 0.5 * np.dot(r_z, r_z)
            )
            if not np.all(np.isfinite(x_new)):
                raise FloatingPointError("FISTA produced a non-finite iterate")

            if restart and _gradient_restart_needed(grad, x_new, x):
                r_x = op.matvec(x) + b_vec - y
                grad_x = op.rmatvec(r_x)
                x_new, L = _backtracked_prox(
                    op, y, b_vec, x, grad_x, L, lam, mu, backtracking,
                    0.5 * np.dot(r_x, r_x),
                )
                if not np.all(np.isfinite(x_new)):
                    raise FloatingPointError("FISTA produced a non-finite iterate")
                momentum = 1.0
                restart_count += 1

            if k % certificate_stride == 0 or k == max_iter:
                cert = compute_certificate(op, y, x_new, lam, mu=mu, b=b_vec, stage=stage, iteration=k)
                history.append(cert)
                if cert.kkt_residual <= kkt_tol and cert.relative_gap <= gap_tol:
                    x = x_new
                    iterations = k
                    status = "converged"
                    break

            next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
            z = x_new + ((momentum - 1.0) / next_momentum) * (x_new - x)
            x = x_new
            momentum = next_momentum
            iterations = k

    return x, status, iterations, L, restart_count, history[-1], history


def solve_fista(A, y, lam, *, mu=0.0, b=None, x0=None, lipschitz=None,
                max_iter=1000, kkt_tol=1e-5, gap_tol=1e-4,
                certificate_stride=10, power_iter=50, power_tol=1e-6,
                restart=True, backtracking=True,
                lambda_path=None, continuation_tol_scale=10.0):
    max_iter = _positive_int(max_iter, "max_iter")
    certificate_stride = _positive_int(certificate_stride, "certificate_stride")
    power_iter = _positive_int(power_iter, "power_iter")
    kkt_tol = _positive_float(kkt_tol, "kkt_tol")
    gap_tol = _positive_float(gap_tol, "gap_tol")
    power_tol = _positive_float(power_tol, "power_tol")
    if lipschitz is not None:
        lipschitz = _positive_float(lipschitz, "lipschitz")

    op = as_linear_operator(A)
    y, b_vec, x, lam, mu = _validate_common(op, y, lam, mu, b, x0)
    try:
        path = [float(lam)] if lambda_path is None else [float(v) for v in lambda_path]
    except (TypeError, ValueError):
        raise ValueError("lambda_path must be an iterable of positive finite numbers") from None
    if (not path or path[-1] != lam or any(not np.isfinite(v) or v <= 0 for v in path)
            or any(path[i] <= path[i + 1] for i in range(len(path) - 1))):
        raise ValueError("lambda_path must be positive, strictly decreasing, and end at lam")
    try:
        continuation_tol_scale = float(continuation_tol_scale)
    except (TypeError, ValueError):
        raise ValueError("continuation_tol_scale must be a finite number of at least one") from None
    if not np.isfinite(continuation_tol_scale) or continuation_tol_scale < 1.0:
        raise ValueError("continuation_tol_scale must be a finite number of at least one")

    L = float(lipschitz) if lipschitz is not None else _estimate_lipschitz(
        op, power_iter=power_iter, power_tol=power_tol
    )
    status = "max_iter"
    iterations = 0
    total_iterations = 0
    restart_count = 0
    certificate = None
    history = []

    for stage, stage_lam in enumerate(path):
        scale = 1.0 if stage == len(path) - 1 else continuation_tol_scale
        x, status, iterations, L, stage_restarts, certificate, stage_history = _run_stage(
            op, y, b_vec, x,
            lam=stage_lam, mu=mu, L=L,
            restart=restart, backtracking=backtracking,
            max_iter=max_iter, kkt_tol=kkt_tol * scale, gap_tol=gap_tol * scale,
            certificate_stride=certificate_stride, stage=stage,
        )
        total_iterations += iterations
        restart_count += stage_restarts
        history.extend(stage_history)

    return FistaResult(
        x, status, iterations, total_iterations, L, restart_count, certificate, tuple(history)
    )
