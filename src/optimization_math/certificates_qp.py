from optimization_math.certificates import Certificate

import numpy as np


def compute_certificate_qp(Q, c, x, l, u, *, stage=0, iteration=0):
    """Box-constrained QP optimality certificate.

    min 0.5 x^T Q x + c^T x  s.t.  l <= x <= u,  Q symmetric PSD.

    KKT: on the interior g_i = 0; at the lower bound g_i >= 0; at the upper
    bound g_i <= 0 (g = Qx + c). The complementarity gap sums the weighted
    bound slacks and goes to zero exactly at optimality (strong duality).
    """
    Q = np.asarray(Q, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be square")
    n = Q.shape[0]
    if c.shape != (n,) or x.shape != (n,):
        raise ValueError("c and x must match Q's dimension")
    if not np.all(np.isfinite(Q)) or not np.all(np.isfinite(c)) or not np.all(np.isfinite(x)):
        raise ValueError("Q, c, x must be finite")

    # Bounds default to (-inf, +inf).
    l = np.full(n, -np.inf) if l is None else np.asarray(l, dtype=np.float64)
    u = np.full(n, np.inf) if u is None else np.asarray(u, dtype=np.float64)
    if l.shape != (n,) or u.shape != (n,):
        raise ValueError("l and u must match Q's dimension")
    if np.any(l > u):
        raise ValueError("l must be <= u")
    if np.any(x < l) or np.any(x > u):
        raise ValueError("x must be inside the box [l, u]")

    g = Q @ x + c

    # KKT violations, coordinate-wise.
    viol = np.zeros(n)
    fixed = (u - l) <= 1e-12          # equality-constrained coordinate: g_i free
    interior = (x > l) & (x < u) & ~fixed
    lower = (x <= l + 1e-12) & ~fixed
    upper = (x >= u - 1e-12) & ~fixed
    viol[interior] = np.abs(g[interior])
    viol[lower] = np.maximum(0.0, -g[lower])
    viol[upper] = np.maximum(0.0, g[upper])
    # fixed coordinates: viol stays 0 (multiplier free)
    kkt = float(np.max(viol) / max(1.0, np.linalg.norm(g, ord=np.inf)))

    primal = 0.5 * np.dot(x, Q @ x) + np.dot(c, x)

    # Lagrange multipliers from stationarity: g - mu_l + mu_u = 0.
    mu_l = np.maximum(0.0, g)
    mu_u = np.maximum(0.0, -g)
    # Complementarity gap: mu_l^T (x - l) + mu_u^T (u - x), nonnegative, ->0 optimum.
    comp_gap = float(
        np.dot(mu_l, x - np.where(np.isfinite(l), l, 0.0))
        + np.dot(mu_u, np.where(np.isfinite(u), u, 0.0) - x)
    )
    # Guard infinite bounds: if a bound is infinite the corresponding mu is 0 at
    # stationarity anyway (g would need to be ±inf), so the above is safe; still
    # clip to nonnegative for roundoff.
    comp_gap = max(comp_gap, 0.0)
    gap = comp_gap / max(1.0, abs(primal), abs(primal - comp_gap))

    return Certificate(stage, iteration, 0.0,
                       float(primal), float(primal - comp_gap), float(kkt), gap)
