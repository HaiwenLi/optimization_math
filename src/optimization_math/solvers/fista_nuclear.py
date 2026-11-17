from dataclasses import dataclass
from typing import Literal

import numpy as np

from optimization_math.certificates import Certificate
from optimization_math.certificates_nuclear import (
    as_matrix_operator,
    compute_certificate_nuclear,
    soft_threshold_singular,
)
from optimization_math.solvers._common import _positive_float, _positive_int


@dataclass(frozen=True)
class FistaNuclearResult:
    X: np.ndarray
    status: Literal["converged", "max_iter"]
    iterations: int
    lipschitz: float
    certificate: Certificate
    history: tuple[Certificate, ...]


def _estimate_nuclear_lipschitz(A, shape, *, power_iter=50):
    """Estimate ||A||_2^2 for the matrix-valued operator via power iteration."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal(shape)
    X /= np.linalg.norm(X)
    est = 0.0
    for _ in range(power_iter):
        W = A.rmatvec(A.matvec(X))
        norm = np.linalg.norm(W)
        if norm == 0:
            return 1.0
        W /= norm
        est = float(np.dot(W.ravel(), A.rmatvec(A.matvec(W)).ravel()))
        X = W
    return max(est, np.finfo(np.float64).tiny) * 1.05


def solve_fista_nuclear(A, y, lam, *, X0=None, lipschitz=None,
                        max_iter=1000, kkt_tol=1e-5, gap_tol=1e-4,
                        certificate_stride=10, restart=True, power_iter=50):
    max_iter = _positive_int(max_iter, "max_iter")
    certificate_stride = _positive_int(certificate_stride, "certificate_stride")
    power_iter = _positive_int(power_iter, "power_iter")
    kkt_tol = _positive_float(kkt_tol, "kkt_tol")
    gap_tol = _positive_float(gap_tol, "gap_tol")
    if lipschitz is not None:
        lipschitz = _positive_float(lipschitz, "lipschitz")

    A = as_matrix_operator(A)
    y = np.asarray(y, dtype=np.float64)
    shape = A.shape  # expects A.shape to expose the matrix shape mapping
    X = np.zeros(shape) if X0 is None else np.asarray(X0, dtype=np.float64).copy()
    try:
        lam = float(lam)
    except (TypeError, ValueError):
        raise ValueError("lam must be a positive number") from None
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be a positive number")
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        raise ValueError("y must be a finite 1D vector")
    if not np.all(np.isfinite(X)):
        raise ValueError("X0 must be finite")

    L = float(lipschitz) if lipschitz is not None else _estimate_nuclear_lipschitz(
        A, shape, power_iter=power_iter
    )
    Z = X.copy()
    momentum = 1.0
    history = [compute_certificate_nuclear(A, y, X, lam, stage=0, iteration=0)]
    status = "max_iter"
    iterations = 0

    if history[-1].kkt_residual <= kkt_tol and history[-1].relative_gap <= gap_tol:
        status = "converged"
        return FistaNuclearResult(X, status, iterations, L, history[-1], tuple(history))

    for k in range(1, max_iter + 1):
        r = A.matvec(Z) - y
        grad = A.rmatvec(r)
        X_new = soft_threshold_singular(Z - grad / L, lam / L)
        if not np.all(np.isfinite(X_new)):
            raise FloatingPointError("FISTA produced a non-finite iterate")

        if restart and np.sum(grad * (X_new - X)) > 0.0:
            r_x = A.matvec(X) - y
            grad_x = A.rmatvec(r_x)
            X_new = soft_threshold_singular(X - grad_x / L, lam / L)
            momentum = 1.0

        if k % certificate_stride == 0 or k == max_iter:
            cert = compute_certificate_nuclear(A, y, X_new, lam, stage=0, iteration=k)
            history.append(cert)
            if cert.kkt_residual <= kkt_tol and cert.relative_gap <= gap_tol:
                X = X_new
                iterations = k
                status = "converged"
                break

        next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        Z = X_new + ((momentum - 1.0) / next_momentum) * (X_new - X)
        X = X_new
        momentum = next_momentum
        iterations = k

    return FistaNuclearResult(X, status, iterations, L, history[-1], tuple(history))
