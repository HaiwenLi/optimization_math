from optimization_math.certificates import Certificate
from optimization_math.operators import as_linear_operator

import numpy as np


def _normalize_group_partition(groups, n):
    """Require nonempty integer groups partitioning all n coordinates."""
    out = []
    for group in groups:
        g = np.asarray(group)
        if g.ndim != 1 or g.size == 0:
            raise ValueError("groups must be a list of nonempty index arrays")
        if not np.issubdtype(g.dtype, np.integer) or any(
            isinstance(index, (bool, np.bool_)) for index in group
        ):
            raise ValueError("group indices must have integer dtype, not bool or float")
        # Check before casting so unsigned indices cannot wrap into valid ones.
        if np.any(g < 0) or np.any(g >= n):
            raise ValueError("group indices out of range")
        out.append(g.astype(np.intp, copy=False))
    if len(out) == 0:
        raise ValueError("groups must be a nonempty list of index arrays")
    all_idx = np.concatenate(out)
    if all_idx.size != n or np.unique(all_idx).size != n:
        raise ValueError("groups must partition every coordinate exactly once")
    return out


def compute_certificate_group(A, y, x, lam, groups, *, mu=0.0, b=None, stage=0, iteration=0):
    """Group-LASSO optimality certificate.

    min 0.5||Ax - y||^2 + lam sum_g ||x_(g)||_2 (+ optional mu/2||x||^2).

    Groups must be nonempty integer index arrays forming a disjoint complete
    partition of the coordinates; overlapping or unpenalized layouts are unsupported.
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
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(b_vec)):
        raise ValueError("A, y, x, b must be finite")
    try:
        lam = float(lam)
        mu = float(mu)
    except (TypeError, ValueError):
        raise ValueError("lam must be positive and mu nonnegative") from None
    if not np.isfinite(lam) or lam <= 0 or not np.isfinite(mu) or mu < 0:
        raise ValueError("lam must be positive and mu nonnegative")

    slices = _normalize_group_partition(groups, n)
    r = op.matvec(x) + b_vec - y
    g = op.rmatvec(r) + mu * x
    primal = 0.5 * np.dot(r, r) + 0.5 * mu * np.dot(x, x)

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

    # Dual: nu = theta r, theta from the group gradient g = A^T r + mu x,
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
