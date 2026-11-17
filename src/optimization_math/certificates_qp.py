from optimization_math.certificates import Certificate

import numpy as np


def compute_certificate_qp(Q, c, x, l, u, *, stage=0, iteration=0):
    """Box-constrained QP optimality certificate.

    min 0.5 x^T Q x + c^T x  s.t.  l <= x <= u,  Q symmetric PSD.

    KKT: on the interior g_i = 0; at the lower bound g_i >= 0; at the upper
    bound g_i <= 0 (g = Qx + c). A separable quadratic minorant provides
    a lower bound even for one-sided or fully unbounded boxes.
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
    if (np.any(np.isnan(l)) or np.any(np.isnan(u)) or np.any(np.isposinf(l))
            or np.any(np.isneginf(u)) or np.any(l > u)):
        raise ValueError("bounds must define a nonempty box without NaN")
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

    # For PSD Q and m=lambda_min(Q), f(x+d) >= f(x)+g.T*d+m/2*||d||^2.
    # Minimize this minorant over l-x <= d <= u-x. Unlike assigning
    # multipliers to nonexistent infinite bounds, this is a genuine lower
    # bound away from stationarity as well as at an optimum.
    eigenvalues = np.linalg.eigvalsh(Q)
    # A mathematically zero eigenvalue may round slightly positive. Subtract
    # a scale-dependent eigensolver roundoff allowance before using curvature;
    # otherwise an unbounded singular QP could receive a finite "lower bound".
    roundoff = 8.0 * np.finfo(float).eps * n * np.max(np.abs(eigenvalues))
    curvature = max(0.0, float(eigenvalues[0] - roundoff))
    if curvature > 0.0:
        d = np.clip(-g / curvature, l - x, u - x)
        improvement = float(np.dot(g, d) + 0.5 * curvature * np.dot(d, d))
    else:
        # Linear minorant; avoid 0*inf on zero-gradient coordinates.
        positive = g > 0.0
        negative = g < 0.0
        improvement = float(np.dot(g[positive], (l - x)[positive])
                            + np.dot(g[negative], (u - x)[negative]))
    dual = float(primal + min(improvement, 0.0))
    gap = ((primal - dual) / max(1.0, abs(primal), abs(dual))
           if np.isfinite(dual) else np.inf)

    # Certificate.lam carries the l1 penalty weight; box-QP has none, so 0.0
    # signals "no sparsity term". The `relative_gap` field carries the normalized
    # minorant gap (a valid upper bound on objective suboptimality).
    return Certificate(stage, iteration, 0.0,
                       float(primal), dual, float(kkt), gap)
