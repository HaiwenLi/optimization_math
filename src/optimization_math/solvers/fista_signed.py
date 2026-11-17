from dataclasses import dataclass
from typing import Literal

import numpy as np

from optimization_math.certificates import Certificate, _vector
from optimization_math.certificates_signed import compute_certificate_signed
from optimization_math.operators import as_linear_operator
from optimization_math.solvers._common import _estimate_lipschitz, _positive_float, _positive_int


@dataclass(frozen=True)
class FistaSignedResult:
    x: np.ndarray
    status: Literal["converged", "max_iter"]
    iterations: int
    total_iterations: int
    lipschitz: float
    restart_count: int
    certificate: Certificate
    history: tuple[Certificate, ...]


def _soft_threshold_prox(z, *, step, lam, mu):
    magnitude = np.maximum(np.abs(z) - step * lam, 0.0)
    return np.sign(z) * magnitude / (1.0 + step * mu)


def lambda_max_signed(A, y, *, b=None):
    op = as_linear_operator(A)
    y = _vector(y, "y")
    b_vec = np.zeros_like(y) if b is None else _vector(b, "b")
    if y.shape != (op.shape[0],) or b_vec.shape != y.shape:
        raise ValueError("y and b must match the operator output dimension")
    g0 = op.rmatvec(b_vec - y)
    return float(np.linalg.norm(g0, ord=np.inf))


def _validate_signed(op, y, lam, mu, b, x0):
    y = _vector(y, "y")
    b_vec = np.zeros_like(y) if b is None else _vector(b, "b")
    if y.shape != (op.shape[0],) or b_vec.shape != y.shape:
        raise ValueError("y and b must match the operator output dimension")
    x = np.zeros(op.shape[1], dtype=np.float64) if x0 is None else _vector(x0, "x0").copy()
    if x.shape != (op.shape[1],):
        raise ValueError("x0 must match the operator input dimension")
    try:
        lam = float(lam)
        mu = float(mu)
    except (TypeError, ValueError):
        raise ValueError("lam must be positive and mu must be nonnegative") from None
    if not np.isfinite(lam) or not np.isfinite(mu) or lam <= 0 or mu < 0:
        raise ValueError("lam must be positive and mu must be nonnegative")
    return y, b_vec, x, lam, mu


def _backtracked_soft_prox(op, y, b_vec, z, grad, L, lam, mu, enabled, f_z):
    for _ in range(60):
        x_new = _soft_threshold_prox(z - grad / L, step=1.0 / L, lam=lam, mu=mu)
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


def solve_fista_signed(A, y, lam, *, mu=0.0, b=None, x0=None, lipschitz=None,
                       max_iter=1000, kkt_tol=1e-5, gap_tol=1e-4,
                       certificate_stride=10, power_iter=50, power_tol=1e-6,
                       restart=True, backtracking=True):
    max_iter = _positive_int(max_iter, "max_iter")
    certificate_stride = _positive_int(certificate_stride, "certificate_stride")
    power_iter = _positive_int(power_iter, "power_iter")
    kkt_tol = _positive_float(kkt_tol, "kkt_tol")
    gap_tol = _positive_float(gap_tol, "gap_tol")
    power_tol = _positive_float(power_tol, "power_tol")
    if lipschitz is not None:
        lipschitz = _positive_float(lipschitz, "lipschitz")

    op = as_linear_operator(A)
    y, b_vec, x, lam, mu = _validate_signed(op, y, lam, mu, b, x0)
    L = float(lipschitz) if lipschitz is not None else _estimate_lipschitz(
        op, power_iter=power_iter, power_tol=power_tol
    )

    z = x.copy()
    momentum = 1.0
    restart_count = 0
    history = [compute_certificate_signed(op, y, x, lam, mu=mu, b=b_vec, stage=0, iteration=0)]
    status = "max_iter"
    iterations = 0

    if history[-1].kkt_residual <= kkt_tol and history[-1].relative_gap <= gap_tol:
        status = "converged"
    else:
        for k in range(1, max_iter + 1):
            r_z = op.matvec(z) + b_vec - y
            grad = op.rmatvec(r_z)
            x_new, L = _backtracked_soft_prox(
                op, y, b_vec, z, grad, L, lam, mu, backtracking, 0.5 * np.dot(r_z, r_z)
            )
            if not np.all(np.isfinite(x_new)):
                raise FloatingPointError("FISTA produced a non-finite iterate")

            if restart and np.dot(grad, x_new - x) > 0.0:
                r_x = op.matvec(x) + b_vec - y
                grad_x = op.rmatvec(r_x)
                x_new, L = _backtracked_soft_prox(
                    op, y, b_vec, x, grad_x, L, lam, mu, backtracking,
                    0.5 * np.dot(r_x, r_x),
                )
                if not np.all(np.isfinite(x_new)):
                    raise FloatingPointError("FISTA produced a non-finite iterate")
                momentum = 1.0
                restart_count += 1

            if k % certificate_stride == 0 or k == max_iter:
                cert = compute_certificate_signed(op, y, x_new, lam, mu=mu, b=b_vec, stage=0, iteration=k)
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

    return FistaSignedResult(
        x, status, iterations, iterations, L, restart_count, history[-1], tuple(history)
    )
