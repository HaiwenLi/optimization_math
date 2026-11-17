from dataclasses import dataclass
from typing import Literal

import numpy as np

from optimization_math.certificates import Certificate
from optimization_math.certificates_qp import compute_certificate_qp
from optimization_math.solvers._common import _positive_float, _positive_int


@dataclass(frozen=True)
class QpBoxResult:
    x: np.ndarray
    status: Literal["converged", "max_iter"]
    iterations: int
    lipschitz: float
    certificate: Certificate
    history: tuple[Certificate, ...]


def _project_box(x, l, u):
    return np.clip(x, l, u)


def _estimate_qp_lipschitz(Q, *, power_iter=50):
    """Estimate ||Q||_2 via power iteration, inflated by a safety factor.

    The raw Rayleigh quotient is a *lower* bound on ||Q||_2; projected gradient
    needs L >= ||Q||_2, so we inflate slightly to stay in the safe regime.
    """
    rng = np.random.default_rng(0)
    v = rng.standard_normal(Q.shape[0])
    v /= np.linalg.norm(v)
    est = 0.0
    for _ in range(power_iter):
        w = Q @ v
        norm = np.linalg.norm(w)
        if norm == 0:
            return 1.0
        w /= norm
        est = float(np.dot(w, Q @ w))
        v = w
    return max(est, np.finfo(np.float64).tiny) * 1.05


def solve_qp_box(Q, c, l=None, u=None, *, x0=None, lipschitz=None,
                 max_iter=2000, kkt_tol=1e-5, gap_tol=1e-4,
                 certificate_stride=10, power_iter=50):
    """Box-constrained QP via projected gradient (FISTA-accelerated projection)."""
    max_iter = _positive_int(max_iter, "max_iter")
    certificate_stride = _positive_int(certificate_stride, "certificate_stride")
    power_iter = _positive_int(power_iter, "power_iter")
    kkt_tol = _positive_float(kkt_tol, "kkt_tol")
    gap_tol = _positive_float(gap_tol, "gap_tol")
    if lipschitz is not None:
        lipschitz = _positive_float(lipschitz, "lipschitz")

    Q = np.asarray(Q, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be square")
    n = Q.shape[0]
    if c.shape != (n,):
        raise ValueError("c must match Q's dimension")
    l = np.full(n, -np.inf) if l is None else np.asarray(l, dtype=np.float64)
    u = np.full(n, np.inf) if u is None else np.asarray(u, dtype=np.float64)
    if l.shape != (n,) or u.shape != (n,) or np.any(l > u):
        raise ValueError("l and u must match dimension and satisfy l <= u")

    L = float(lipschitz) if lipschitz is not None else _estimate_qp_lipschitz(Q, power_iter=power_iter)
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    x = _project_box(x, l, u)

    z = x.copy()
    momentum = 1.0
    history = [compute_certificate_qp(Q, c, x, l, u, stage=0, iteration=0)]
    status = "max_iter"
    iterations = 0

    if history[-1].kkt_residual <= kkt_tol and history[-1].relative_gap <= gap_tol:
        status = "converged"
        return QpBoxResult(x, status, iterations, L, history[-1], tuple(history))

    for k in range(1, max_iter + 1):
        grad = Q @ z + c
        x_new = _project_box(z - grad / L, l, u)
        if not np.all(np.isfinite(x_new)):
            raise FloatingPointError("projected gradient produced a non-finite iterate")

        if k % certificate_stride == 0 or k == max_iter:
            cert = compute_certificate_qp(Q, c, x_new, l, u, stage=0, iteration=k)
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

    return QpBoxResult(x, status, iterations, L, history[-1], tuple(history))
