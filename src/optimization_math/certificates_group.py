from optimization_math.certificates import Certificate
from optimization_math.operators import as_linear_operator

import numpy as np


def _group_slices(groups):
    """Normalize groups (list of coordinate arrays) into a list of int arrays."""
    out = [np.asarray(g, dtype=np.intp) for g in groups]
    if any(g.ndim != 1 or g.size == 0 for g in out):
        raise ValueError("groups must be a list of nonempty index arrays")
    return out


def compute_certificate_group(A, y, x, lam, groups, *, mu=0.0, b=None, stage=0, iteration=0):
    """Group-LASSO optimality certificate.

    min 0.5||Ax - y||^2 + lam sum_g ||x_(g)||_2 (+ optional mu/2||x||^2).
    """
    op = as_linear_operator(A)
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    b_vec = np.zeros_like(y) if b is None else np.asarray(b, dtype=np.float64)
    if y.ndim != 1 or x.ndim != 1:
        raise ValueError("y and x must be 1D")
    n = x.shape[0]
    if y.shape != (op.shape[0],) or x.shape != (op.shape[1],) or b_vec.shape != y.shape:
        raise ValueError("A, y, x, b shapes do not match")
    lam = float(lam)
    mu = float(mu)
    if not np.isfinite(lam) or lam <= 0 or not np.isfinite(mu) or mu < 0:
        raise ValueError("lam must be positive and mu nonnegative")

    r = op.matvec(x) + b_vec - y
    g = op.rmatvec(r) + mu * x
    primal = 0.5 * np.dot(r, r) + 0.5 * mu * np.dot(x, x)

    slices = _group_slices(groups)
    all_idx = np.concatenate(slices) if slices else np.array([], dtype=np.intp)
    if all_idx.size and (all_idx.min() < 0 or all_idx.max() >= n):
        raise ValueError("group indices out of range")
    max_viol = 0.0
    max_gnorm = 0.0
    for s in slices:
        gs = g[s]
        xs = x[s]
        gnorm = float(np.linalg.norm(gs))
        max_gnorm = max(max_gnorm, gnorm)
        xnorm = float(np.linalg.norm(xs))
        primal += lam * xnorm
        if xnorm > 0:
            viol = float(np.linalg.norm(gs + lam * xs / xnorm))
        else:
            viol = max(0.0, gnorm - lam)
        max_viol = max(max_viol, viol)
    kkt = max_viol / max(lam, max_gnorm)

    # Dual: nu = -theta r, theta from the group gradient g = A^T r + mu x,
    # mirroring the reference certificate's use of q = A^T r + mu x for theta.
    max_group_gnorm = max((float(np.linalg.norm(g[s])) for s in slices), default=0.0)
    theta = 1.0 if max_group_gnorm <= lam else lam / max_group_gnorm
    dual = (
        -0.5 * theta**2 * (np.dot(r, r) + mu * np.dot(x, x))
        + theta * np.dot(b_vec - y, r)
    )
    gap = (primal - dual) / max(1.0, abs(primal), abs(dual))
    if -1e-12 <= gap < 0:
        gap = 0.0
    elif gap < 0:
        raise FloatingPointError("Material negative dual gap indicates a certificate error")

    return Certificate(stage, iteration, lam, float(primal), float(dual), float(kkt), gap)
