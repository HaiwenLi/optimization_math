from dataclasses import dataclass
from typing import Literal

import numpy as np

from optimization_math.certificates import Certificate
from optimization_math.operators_difference import first_difference_operator


@dataclass(frozen=True)
class ChambolleTvResult:
    x: np.ndarray
    status: Literal["converged", "max_iter"]
    iterations: int
    certificate: Certificate
    history: tuple[Certificate, ...]


def certificate_tv_chambolle(D, y, x, lam_tv, p):
    """Build a Certificate from a Chambolle iterate (x, dual p).

    For Chambolle the meaningful optimality signal is the primal-dual gap (the
    subgradient on plateaus Dx=0 is genuinely free and sign(0)=0 is not the
    correct subgradient). The ``kkt_residual`` field therefore carries the
    absolute primal-dual gap, which goes to zero exactly at the optimum.
    """
    primal = 0.5 * np.dot(x - y, x - y) + lam_tv * np.sum(np.abs(D.matvec(x)))
    dual = 0.5 * np.dot(y, y) - 0.5 * np.dot(y - lam_tv * D.rmatvec(p),
                                             y - lam_tv * D.rmatvec(p))
    abs_gap = primal - dual
    gap = abs_gap / max(1.0, abs(primal), abs(dual))
    if -1e-12 <= gap < 0:
        gap = 0.0
    elif gap < 0:
        raise FloatingPointError("Material negative dual gap indicates a certificate error")
    return Certificate(0, 0, float(lam_tv), float(primal), float(dual), float(max(abs_gap, 0.0)), gap)


def solve_tv_chambolle(y, lam_tv, *, tau=None, tol=1e-5, max_iter=5000):
    """Chambolle-projection TV denoiser with certificate-gated convergence."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must be finite")
    lam_tv = float(lam_tv)
    if not np.isfinite(lam_tv) or lam_tv < 0:
        raise ValueError("lam_tv must be nonnegative")
    n = y.shape[0]
    D = first_difference_operator(n)

    if lam_tv == 0.0:
        cert = certificate_tv_chambolle(D, y, y.copy(), lam_tv, np.zeros(n - 1))
        return ChambolleTvResult(y.copy(), "converged", 0, cert, (cert,))

    p = np.zeros(n - 1)
    tau = 0.25 / lam_tv if tau is None else float(tau)
    if not np.isfinite(tau) or tau <= 0 or tau > 0.25 / lam_tv:
        raise ValueError("tau must be in (0, 1/(4 lam_tv)]")
    x = y.copy()
    history = []
    status = "max_iter"
    for k in range(1, max_iter + 1):
        g = D.matvec(y - lam_tv * D.rmatvec(p))
        p = (p + tau * g) / (1.0 + tau * np.abs(g))
        x = y - lam_tv * D.rmatvec(p)
        if k % 10 == 0 or k == max_iter:
            cert = certificate_tv_chambolle(D, y, x, lam_tv, p)
            cert = Certificate(0, k, cert.lam, cert.primal, cert.dual, cert.kkt_residual, cert.relative_gap)
            history.append(cert)
            if cert.relative_gap <= tol:
                status = "converged"
                break
    cert = history[-1]
    return ChambolleTvResult(x, status, k, cert, tuple(history))
