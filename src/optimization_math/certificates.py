from dataclasses import dataclass

import numpy as np

from optimization_math.operators import as_linear_operator


@dataclass(frozen=True)
class Certificate:
    stage: int
    iteration: int
    lam: float
    primal: float
    dual: float
    kkt_residual: float
    relative_gap: float


def _vector(value, name):
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 1 or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return out


def compute_certificate(A, y, x, lam, *, mu=0.0, b=None, stage=0, iteration=0):
    op = as_linear_operator(A)
    y = _vector(y, "y")
    x = _vector(x, "x")
    b_vec = np.zeros_like(y) if b is None else _vector(b, "b")
    if y.shape != (op.shape[0],) or b_vec.shape != y.shape or x.shape != (op.shape[1],):
        raise ValueError("Certificate vector shapes do not match the operator")
    if not np.isfinite(lam) or not np.isfinite(mu) or lam <= 0 or mu < 0:
        raise ValueError("lam must be positive and mu must be nonnegative")
    if np.any(x < 0):
        raise ValueError("x must be nonnegative")

    r = op.matvec(x) + b_vec - y
    q = op.rmatvec(r) + mu * x
    primal = 0.5 * np.dot(r, r) + lam * np.sum(x) + 0.5 * mu * np.dot(x, x)

    positive = x > 0
    active_violation = np.max(np.abs(q[positive] + lam), initial=0.0)
    inactive_violation = np.max(np.maximum(-lam - q[~positive], 0.0), initial=0.0)
    kkt = max(active_violation, inactive_violation) / max(lam, np.linalg.norm(q, ord=np.inf))

    negative = q < 0
    theta = min(1.0, float(np.min(lam / -q[negative], initial=1.0))) if np.any(negative) else 1.0
    dual = (
        -0.5 * theta**2 * (np.dot(r, r) + mu * np.dot(x, x))
        + theta * np.dot(b_vec - y, r)
    )
    gap = (primal - dual) / max(1.0, abs(primal), abs(dual))
    if -1e-12 <= gap < 0:
        gap = 0.0
    elif gap < 0:
        raise FloatingPointError("Material negative dual gap indicates a certificate error")

    return Certificate(stage, iteration, float(lam), float(primal), float(dual), float(kkt), float(gap))
