"""Chambolle-Pock primal-dual (PDHG) solver for signed LASSO / elastic-net.

Solves

    min_x  0.5 ||A x + b - y||^2  +  lam ||x||_1  +  0.5 * mu ||x||^2

via the primal-dual (Chambolle-Pock, 2011) iteration on the saddle point

    min_x max_z  <A x, z> + G(x) - F*(z),

with  G(x) = lam ||x||_1 + 0.5 mu ||x||^2  and  F(z) = 0.5 ||z + b - y||^2
(so  F*(z) = 0.5 ||z||^2 + <z, y - b>), and K = A.

The iteration (theta = 1):

    z^{k+1} = prox_{sigma F*}( z^k + sigma A xbar^k ),
    x^{k+1} = prox_{tau G}( x^k - tau A^T z^{k+1} ),
    xbar^{k+1} = x^{k+1} + theta (x^{k+1} - x^k).

Step sizes satisfy tau * sigma * ||A||^2 <= 1 (theta = 1). Convergence is
certified with the *same* signed-LASSO certificate as FISTA, so the primal-dual
solver plugs into the framework's optimality gate unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from optimization_math.certificates import Certificate
from optimization_math.certificates_signed import compute_certificate_signed
from optimization_math.operators import as_linear_operator
from optimization_math.solvers._common import (
    _estimate_lipschitz,
    _positive_float,
    _positive_int,
)


@dataclass(frozen=True)
class PdhgResult:
    x: np.ndarray
    status: Literal["converged", "max_iter"]
    iterations: int
    total_iterations: int
    tau: float
    sigma: float
    certificate: Certificate
    history: tuple[Certificate, ...]


def _prox_f_star(z, y, b_vec, sigma):
    """prox_{sigma F*} for F(z) = 0.5||z + b - y||^2:
    F*(z) = 0.5||z||^2 + <z, y - b>, so prox_{sigma F*}(z) = (z - sigma*(y-b)) / (1+sigma).
    """
    return (z - sigma * (y - b_vec)) / (1.0 + sigma)


def solve_pdhg(A, y, lam, *, mu=0.0, b=None, x0=None, tau=None, sigma=None,
               lipschitz=None, max_iter=2000, kkt_tol=1e-5, gap_tol=1e-4,
               certificate_stride=10, power_iter=50, power_tol=1e-6, theta=1.0):
    """Solve signed LASSO / elastic-net via Chambolle-Pock primal-dual.

    Convergence is gated on the signed-LASSO certificate (KKT + relative dual
    gap), exactly as FISTA -- the iterate is declared "converged" only when the
    *certificate* is satisfied, never from a raw residual. ``lipschitz`` is
    ||A||^2 of A (used only for the tau*sigma*||A||^2 <= 1 coupling; the ridge
    term mu stays in the strongly convex primal term); ``b`` is an optional
    output-space offset. ``total_iterations`` mirrors ``iterations`` (PDHG
    has no continuation stages).
    """
    max_iter = _positive_int(max_iter, "max_iter")
    certificate_stride = _positive_int(certificate_stride, "certificate_stride")
    power_iter = _positive_int(power_iter, "power_iter")
    kkt_tol = _positive_float(kkt_tol, "kkt_tol")
    gap_tol = _positive_float(gap_tol, "gap_tol")
    power_tol = _positive_float(power_tol, "power_tol")
    if not np.isfinite(theta) or theta <= 0.0:
        raise ValueError("theta must be positive")
    if theta > 1.0:
        # For theta > 1 the step-size condition changes (tau*sigma*||K||^2 <
        # 4/(1+2theta), Banert et al.), which the code below does not enforce.
        # Standard Chambolle-Pock uses theta in (0, 1]; restrict to that range.
        raise ValueError("theta must be in (0, 1]")

    op = as_linear_operator(A)
    y = np.asarray(y, dtype=np.float64)
    b_vec = np.zeros_like(y) if b is None else np.asarray(b, dtype=np.float64)
    if y.ndim != 1 or y.shape != (op.shape[0],) or b_vec.shape != y.shape:
        raise ValueError("y and b must match the operator output dimension")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(b_vec)):
        raise ValueError("y and b must be finite")
    n = op.shape[1]
    x = np.zeros(n, dtype=np.float64) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    if x.shape != (n,):
        raise ValueError("x0 must match the operator input dimension")
    if not np.all(np.isfinite(x)):
        raise ValueError("x0 must be finite")
    try:
        lam = float(lam)
        mu = float(mu)
    except (TypeError, ValueError):
        raise ValueError("lam must be positive and mu nonnegative") from None
    if not np.isfinite(lam) or not np.isfinite(mu) or lam <= 0 or mu < 0:
        raise ValueError("lam must be positive and mu nonnegative")

    # The saddle-point coupling is K = A only: the elastic-net quadratic is kept
    # in the (strongly convex) G(x) = lam||x||_1 + 0.5 mu||x||^2, so the step-size
    # condition is tau * sigma * ||A||^2 <= 1 (theta = 1). ||A||^2 is estimated
    # without the mu term (mu only enters prox_G's denominator 1+tau*mu).
    if lipschitz is not None:
        L = _positive_float(lipschitz, "lipschitz")
    else:
        L = float(_estimate_lipschitz(op, power_iter=power_iter, power_tol=power_tol))

    # Primal-dual step sizes: default tau = sigma = 1/sqrt(L) gives
    # tau * sigma * L = 1, satisfying the theta=1 condition.
    if tau is None:
        tau = 1.0 / np.sqrt(L)
    else:
        tau = _positive_float(tau, "tau")
    if sigma is None:
        sigma = 1.0 / np.sqrt(L)
    else:
        sigma = _positive_float(sigma, "sigma")
    if tau * sigma * L > 1.0 + 1e-12:
        raise ValueError("tau * sigma * ||A||^2 must be <= 1 for theta=1")

    z = np.zeros_like(y)
    xbar = x.copy()
    history = [compute_certificate_signed(op, y, x, lam, mu=mu, b=b_vec, stage=0, iteration=0)]
    status = "max_iter"
    iterations = 0

    if history[-1].kkt_residual <= kkt_tol and history[-1].relative_gap <= gap_tol:
        status = "converged"
    else:
        for k in range(1, max_iter + 1):
            # z-step: prox_{sigma F*}(z + sigma A xbar)
            z = _prox_f_star(z + sigma * op.matvec(xbar), y, b_vec, sigma)
            # x-step: prox_{tau G}(x - tau A^T z)
            x_new = _prox_g(x - tau * op.rmatvec(z), tau, lam, mu)
            if not np.all(np.isfinite(x_new)):
                raise FloatingPointError("PDHG produced a non-finite iterate")

            xbar_new = x_new + theta * (x_new - x)
            x = x_new
            xbar = xbar_new

            if k % certificate_stride == 0 or k == max_iter:
                cert = compute_certificate_signed(op, y, x, lam, mu=mu, b=b_vec, stage=0, iteration=k)
                history.append(cert)
                if cert.kkt_residual <= kkt_tol and cert.relative_gap <= gap_tol:
                    iterations = k
                    status = "converged"
                    break
            iterations = k

    return PdhgResult(
        x, status, iterations, iterations, float(tau), float(sigma),
        history[-1], tuple(history)
    )


def _prox_g(z, tau, lam, mu):
    """prox_{tau G} for G(x) = lam||x||_1 + 0.5 mu ||x||^2  (soft threshold with shrink)."""
    magnitude = np.maximum(np.abs(z) - tau * lam, 0.0)
    return np.sign(z) * magnitude / (1.0 + tau * mu)
