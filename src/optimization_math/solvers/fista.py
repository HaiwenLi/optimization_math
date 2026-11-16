from dataclasses import dataclass
from typing import Literal

import numpy as np

from optimization_math.operators import as_linear_operator
from optimization_math.certificates import (
    Certificate,
    _vector,
    compute_certificate,
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


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value, name):
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive finite number") from None
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return out


def _estimate_lipschitz(op, *, power_iter, power_tol):
    rng = np.random.default_rng(0)
    v = rng.standard_normal(op.shape[1])
    v /= np.linalg.norm(v)
    estimate = 0.0
    for _ in range(power_iter):
        w = op.rmatvec(op.matvec(v))
        norm = np.linalg.norm(w)
        if norm == 0:
            return 1.0
        w /= norm
        new_estimate = float(np.dot(w, op.rmatvec(op.matvec(w))))
        if estimate > 0 and abs(new_estimate - estimate) <= power_tol * max(1.0, abs(new_estimate)):
            estimate = new_estimate
            break
        estimate = new_estimate
        v = w
    return max(estimate, np.finfo(np.float64).tiny)


def _validate_common(op, y, lam, mu, b, x0):
    y = _vector(y, "y")
    b_vec = np.zeros_like(y) if b is None else _vector(b, "b")
    if y.shape != (op.shape[0],) or b_vec.shape != y.shape:
        raise ValueError("y and b must match the operator output dimension")
    x = np.zeros(op.shape[1], dtype=np.float64) if x0 is None else _vector(x0, "x0").copy()
    if x.shape != (op.shape[1],):
        raise ValueError("x0 must match the operator input dimension")
    if np.any(x < 0):
        raise ValueError("x0 must be nonnegative")
    if not np.isfinite(lam) or not np.isfinite(mu) or lam <= 0 or mu < 0:
        raise ValueError("lam must be positive and mu must be nonnegative")
    return y, b_vec, x


def solve_fista(A, y, lam, *, mu=0.0, b=None, x0=None, lipschitz=None,
                max_iter=1000, kkt_tol=1e-5, gap_tol=1e-4,
                certificate_stride=10, power_iter=50, power_tol=1e-6):
    max_iter = _positive_int(max_iter, "max_iter")
    certificate_stride = _positive_int(certificate_stride, "certificate_stride")
    power_iter = _positive_int(power_iter, "power_iter")
    kkt_tol = _positive_float(kkt_tol, "kkt_tol")
    gap_tol = _positive_float(gap_tol, "gap_tol")
    power_tol = _positive_float(power_tol, "power_tol")
    if lipschitz is not None:
        lipschitz = _positive_float(lipschitz, "lipschitz")

    op = as_linear_operator(A)
    y, b_vec, x = _validate_common(op, y, lam, mu, b, x0)
    L = float(lipschitz) if lipschitz is not None else _estimate_lipschitz(
        op, power_iter=power_iter, power_tol=power_tol
    )
    z = x.copy()
    momentum = 1.0
    history = [compute_certificate(op, y, x, lam, mu=mu, b=b_vec, stage=0, iteration=0)]
    status = "max_iter"
    iterations = 0

    if history[-1].kkt_residual <= kkt_tol and history[-1].relative_gap <= gap_tol:
        status = "converged"
    else:
        for k in range(1, max_iter + 1):
            r_z = op.matvec(z) + b_vec - y
            grad = op.rmatvec(r_z)
            x_new = _nonnegative_elastic_net_prox(z - grad / L, step=1.0 / L, lam=lam, mu=mu)
            if not np.all(np.isfinite(x_new)):
                raise FloatingPointError("FISTA produced a non-finite iterate")

            if k % certificate_stride == 0 or k == max_iter:
                cert = compute_certificate(op, y, x_new, lam, mu=mu, b=b_vec, stage=0, iteration=k)
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

    return FistaResult(x, status, iterations, iterations, L, 0, history[-1], tuple(history))
