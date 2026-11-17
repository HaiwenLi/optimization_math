from dataclasses import dataclass
from typing import Literal

import numpy as np

from optimization_math.certificates import Certificate
from optimization_math.certificates_logistic import (
    _logistic_loss,
    _sigmoid,
    compute_certificate_logistic,
)
from optimization_math.solvers._common import _positive_float, _positive_int, _soft_threshold_prox


@dataclass(frozen=True)
class FistaLogisticResult:
    w: np.ndarray
    status: Literal["converged", "max_iter"]
    iterations: int
    lipschitz: float
    certificate: Certificate
    history: tuple[Certificate, ...]


def _logistic_loss_and_grad(X, y, w):
    N = X.shape[0]
    margins = y * (X @ w)
    sigma = _sigmoid(-margins)
    loss = float(np.mean(_logistic_loss(margins)))
    grad = -(X.T @ (y * sigma)) / N
    return loss, grad


def solve_fista_logistic(X, y, lam, *, w0=None, lipschitz=None,
                         max_iter=1000, kkt_tol=1e-5, gap_tol=1e-4,
                         certificate_stride=10, restart=True, backtracking=True):
    max_iter = _positive_int(max_iter, "max_iter")
    certificate_stride = _positive_int(certificate_stride, "certificate_stride")
    kkt_tol = _positive_float(kkt_tol, "kkt_tol")
    gap_tol = _positive_float(gap_tol, "gap_tol")
    if lipschitz is not None:
        lipschitz = _positive_float(lipschitz, "lipschitz")

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("X must be 2D and y 1D with matching first dimension")
    if not np.all((y == 1.0) | (y == -1.0)):
        raise ValueError("y must contain only +1 and -1 labels")
    try:
        lam = float(lam)
    except (TypeError, ValueError):
        raise ValueError("lam must be a positive number") from None
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be a positive number")
    p = X.shape[1]
    w = np.zeros(p, dtype=np.float64) if w0 is None else np.asarray(w0, dtype=np.float64).copy()
    if w.shape != (p,):
        raise ValueError("w0 must match the number of features")

    # Default Lipschitz: ||X||_2^2 / (4 N) bound for the normalized logistic loss.
    if lipschitz is None:
        lipschitz = float(np.linalg.norm(X, 2) ** 2) / (4.0 * X.shape[0])
    L = float(lipschitz)

    def _backtrack(w_inner, grad, L_inner):
        f_z, _ = _logistic_loss_and_grad(X, y, w_inner)
        for _ in range(60):
            w_new = _soft_threshold_prox(w_inner - grad / L_inner, step=1.0 / L_inner,
                                         lam=lam, mu=0.0)
            if not backtracking:
                return w_new, L_inner
            f_new, _ = _logistic_loss_and_grad(X, y, w_new)
            upper = f_z + np.dot(grad, w_new - w_inner) + 0.5 * L_inner * np.dot(
                w_new - w_inner, w_new - w_inner
            )
            if f_new <= upper + 1e-12 * max(1.0, abs(f_z), abs(upper)):
                return w_new, L_inner
            L_inner *= 2.0
        raise FloatingPointError("Backtracking failed to find a finite stable step")

    z = w.copy()
    momentum = 1.0
    history = [compute_certificate_logistic(X, y, w, lam, stage=0, iteration=0)]
    status = "max_iter"
    iterations = 0

    if history[-1].kkt_residual <= kkt_tol and history[-1].relative_gap <= gap_tol:
        status = "converged"
    else:
        for k in range(1, max_iter + 1):
            _, grad = _logistic_loss_and_grad(X, y, z)
            w_new, L = _backtrack(z, grad, L)
            if not np.all(np.isfinite(w_new)):
                raise FloatingPointError("FISTA produced a non-finite iterate")

            if restart and np.dot(grad, w_new - w) > 0.0:
                _, grad_w = _logistic_loss_and_grad(X, y, w)
                w_new, L = _backtrack(w, grad_w, L)
                if not np.all(np.isfinite(w_new)):
                    raise FloatingPointError("FISTA produced a non-finite iterate")
                momentum = 1.0

            if k % certificate_stride == 0 or k == max_iter:
                cert = compute_certificate_logistic(X, y, w_new, lam, stage=0, iteration=k)
                history.append(cert)
                if cert.kkt_residual <= kkt_tol and cert.relative_gap <= gap_tol:
                    w = w_new
                    iterations = k
                    status = "converged"
                    break

            next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
            z = w_new + ((momentum - 1.0) / next_momentum) * (w_new - w)
            w = w_new
            momentum = next_momentum
            iterations = k

    return FistaLogisticResult(w, status, iterations, L, history[-1], tuple(history))
