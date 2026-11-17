from optimization_math.certificates import Certificate
from optimization_math.operators import as_linear_operator

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _phi_conjugate(s):
    """Fenchel conjugate of phi(t) = log(1 + exp(-t)), for s in [-1, 0].

    phi*(s) = -s log(-s) + (1 + s) log(1 + s), with the limits
    phi*(0) = phi*(-1) = 0 handled exactly (avoiding log(0) NaNs).
    """
    s = np.asarray(s, dtype=np.float64)
    if not np.all((s >= -1.0) & (s <= 0.0)):
        raise ValueError("phi* is only defined on [-1, 0]")
    out = np.zeros_like(s)
    interior = (s > -1.0) & (s < 0.0)
    out[interior] = (
        -s[interior] * np.log(-s[interior])
        + (1.0 + s[interior]) * np.log(1.0 + s[interior])
    )
    return out


def compute_certificate_logistic(X, y, w, lam, *, stage=0, iteration=0):
    """Sparse logistic regression optimality certificate.

    X : (N, p) design matrix, y : (N,) labels in {-1, +1}, w : (p,) weights.
    Returns a ``Certificate`` whose ``primal`` is the objective value and
    ``dual`` is the Fenchel dual lower bound.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 1 or w.ndim != 1:
        raise ValueError("X must be 2D, y and w must be 1D")
    N, p = X.shape
    if y.shape != (N,) or w.shape != (p,):
        raise ValueError("X, y, w shapes do not match")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(w)):
        raise ValueError("X, y, w must be finite")
    try:
        lam = float(lam)
    except (TypeError, ValueError):
        raise ValueError("lam must be a positive number") from None
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be a positive number")

    margins = y * (X @ w)          # y_i w^T x_i
    sigma = _sigmoid(-margins)     # misclassification probability of sample i

    # Gradient of the (normalized) loss: -(1/N) sum_i y_i x_i sigma_i.
    g = -(X.T @ (y * sigma)) / N

    primal = float(np.mean(np.log(1.0 + np.exp(-margins))) + lam * np.sum(np.abs(w)))

    # KKT: active |w_j|>0 -> g_j = -lam sign(w_j); inactive |w_j|=0 -> |g_j| <= lam.
    nonzero = np.abs(w) > 0
    active_violation = np.max(
        np.abs(g[nonzero] + lam * np.sign(w[nonzero])), initial=0.0
    )
    inactive_violation = np.max(np.maximum(np.abs(g[~nonzero]) - lam, 0.0), initial=0.0)
    kkt = max(active_violation, inactive_violation) / max(lam, np.linalg.norm(g, ord=np.inf))

    # Feasible dual point nu_i = theta * sigma_i, with theta clipped by both the
    # l-inf constraint and the phi* domain bound N*nu_i <= 1.
    g_inf = float(np.linalg.norm(g, ord=np.inf))
    sigma_max = float(np.max(sigma)) if N else 0.0
    theta = 1.0
    if g_inf > 0:
        theta = min(theta, lam / (N * g_inf))
    if sigma_max > 0:
        theta = min(theta, 1.0 / (N * sigma_max))
    s = -N * theta * sigma
    dual = float(-np.mean(_phi_conjugate(s)))
    gap = (primal - dual) / max(1.0, abs(primal), abs(dual))
    if -1e-12 <= gap < 0:
        gap = 0.0
    elif gap < 0:
        raise FloatingPointError("Material negative dual gap indicates a certificate error")

    return Certificate(stage, iteration, float(lam), primal, dual, float(kkt), gap)
