import numpy as np

from optimization_math.certificates import _vector


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value, name):
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive finite number") from None
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return out


def _soft_threshold_prox(z, *, step, lam, mu):
    """Soft-threshold proximity operator: sign(z) max(|z|-step*lam, 0) / (1+step*mu)."""
    magnitude = np.maximum(np.abs(z) - step * lam, 0.0)
    return np.sign(z) * magnitude / (1.0 + step * mu)


def _estimate_lipschitz(op, *, power_iter, power_tol):
    rng = np.random.default_rng(0)
    v = rng.standard_normal(op.shape[1])
    v /= np.linalg.norm(v)
    estimate = 0.0
    for _ in range(power_iter):
        w = op.rmatvec(op.matvec(v))
        norm = np.linalg.norm(w)
        if norm == 0:
            return 1.0
        w /= norm
        new_estimate = float(np.dot(w, op.rmatvec(op.matvec(w))))
        if estimate > 0 and abs(new_estimate - estimate) <= power_tol * max(1.0, abs(new_estimate)):
            estimate = new_estimate
            break
        estimate = new_estimate
        v = w
    # Rayleigh quotient converges to ||A||^2 from below; inflate to stay >= the
    # true Lipschitz constant (projected-gradient/FISTA need L >= ||A||^2).
    return max(estimate, np.finfo(np.float64).tiny) * 1.05


def _validate_common(op, y, lam, mu, b, x0):
    y = _vector(y, "y")
    b_vec = np.zeros_like(y) if b is None else _vector(b, "b")
    if y.shape != (op.shape[0],) or b_vec.shape != y.shape:
        raise ValueError("y and b must match the operator output dimension")
    x = np.zeros(op.shape[1], dtype=np.float64) if x0 is None else _vector(x0, "x0").copy()
    if x.shape != (op.shape[1],):
        raise ValueError("x0 must match the operator input dimension")
    if np.any(x < 0):
        raise ValueError("x0 must be nonnegative")
    try:
        lam = float(lam)
        mu = float(mu)
    except (TypeError, ValueError):
        raise ValueError("lam must be positive and mu must be nonnegative") from None
    if not np.isfinite(lam) or not np.isfinite(mu) or lam <= 0 or mu < 0:
        raise ValueError("lam must be positive and mu must be nonnegative")
    return y, b_vec, x, lam, mu
