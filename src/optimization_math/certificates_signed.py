from optimization_math.certificates import Certificate, _vector
from optimization_math.operators import as_linear_operator

import numpy as np


def compute_certificate_signed(A, y, x, lam, *, mu=0.0, b=None, stage=0, iteration=0):
    """Signed (unconstrained) LASSO / elastic-net optimality certificate.

    Same conventions as ``compute_certificate``, but without the ``x >= 0``
    box: the KKT condition is the plain soft-threshold rule and the dual
    slack ``theta`` is taken over the full ``||Aᵀr||_inf`` magnitude.
    """
    op = as_linear_operator(A)
    y = _vector(y, "y")
    x = _vector(x, "x")
    b_vec = np.zeros_like(y) if b is None else _vector(b, "b")
    if y.shape != (op.shape[0],) or b_vec.shape != y.shape or x.shape != (op.shape[1],):
        raise ValueError("Certificate vector shapes do not match the operator")
    try:
        lam = float(lam)
        mu = float(mu)
    except (TypeError, ValueError):
        raise ValueError("lam must be positive and mu must be nonnegative") from None
    if not np.isfinite(lam) or not np.isfinite(mu) or lam <= 0 or mu < 0:
        raise ValueError("lam must be positive and mu must be nonnegative")

    r = op.matvec(x) + b_vec - y
    q = op.rmatvec(r) + mu * x
    primal = 0.5 * np.dot(r, r) + lam * np.sum(np.abs(x)) + 0.5 * mu * np.dot(x, x)

    # Soft-threshold KKT: active coordinates satisfy q_i = -lam * sign(x_i),
    # inactive satisfy |q_i| <= lam.
    nonzero = np.abs(x) > 0
    active_violation = np.max(
        np.abs(q[nonzero] + lam * np.sign(x[nonzero])), initial=0.0
    )
    inactive_violation = np.max(np.maximum(np.abs(q[~nonzero]) - lam, 0.0), initial=0.0)
    kkt = max(active_violation, inactive_violation) / max(lam, np.linalg.norm(q, ord=np.inf))

    # Dual slack over the full gradient magnitude; no sign restriction without
    # the box constraint.
    q_inf = np.linalg.norm(q, ord=np.inf)
    theta = 1.0 if q_inf <= lam else lam / q_inf
    dual = -0.5 * theta**2 * (np.dot(r, r) + mu * np.dot(x, x)) + theta * np.dot(b_vec - y, r)
    gap = (primal - dual) / max(1.0, abs(primal), abs(dual))
    if -1e-12 <= gap < 0:
        gap = 0.0
    elif gap < 0:
        raise FloatingPointError("Material negative dual gap indicates a certificate error")

    return Certificate(stage, iteration, float(lam), float(primal), float(dual), float(kkt), float(gap))
