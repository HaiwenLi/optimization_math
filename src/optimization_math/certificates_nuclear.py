from optimization_math.certificates import Certificate

import numpy as np


def _spectral_norm(M):
    return np.linalg.norm(M, 2)


def compute_certificate_nuclear(A, y, X, lam, *, stage=0, iteration=0):
    """Nuclear-norm (low-rank) denoising certificate.

    min 0.5||A(X) - y||^2 + lam||X||_*, A linear on X (matrix).

    Stationarity is measured by the normalized proximal fixed-point residual
    ||X - S_lam(X - grad)||_F / max(1, ||X||_F), which goes to zero exactly at
    the optimum and avoids the subdifferential-subspace decomposition. The
    dual bound uses spectral feasibility of A*(theta*r); ``kkt_residual`` is
    therefore NOT a KKT norm like the LASSO certificates.
    """
    A = as_matrix_operator(A)
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix")
    if X.shape != tuple(A.shape):
        raise ValueError("X must match A's matrix shape")
    try:
        lam = float(lam)
    except (TypeError, ValueError):
        raise ValueError("lam must be a positive number") from None
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be a positive number")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(X)):
        raise ValueError("y and X must be finite")
    out = A.matvec(X)
    if y.ndim != 1 or y.shape[0] != out.shape[0]:
        raise ValueError("y must be a 1D vector matching A's output size")
    r = out - y
    grad = A.rmatvec(r)
    primal = 0.5 * np.dot(r, r) + lam * np.sum(np.linalg.svd(X, compute_uv=False))

    # KKT via proximal fixed point: X = S_lam(X - grad) at the optimum. The
    # residual ||X - S_lam(X - grad)||_F goes to zero exactly at optimality and
    # avoids the sub-differential subspace decomposition.
    prox = soft_threshold_singular(X - grad, lam)
    kkt = float(np.linalg.norm(X - prox) / max(1.0, np.linalg.norm(X)))

    # Dual: nu feasible if ||A* nu||_2 <= lam; use nu = theta * r with
    # theta = min(1, lam / ||A* r||_2).
    atr = A.rmatvec(r)
    natr = _spectral_norm(atr)
    theta = 1.0 if natr <= lam else lam / natr
    dual = -0.5 * theta**2 * np.dot(r, r) + theta * np.dot(-y, r)
    gap = (primal - dual) / max(1.0, abs(primal), abs(dual))
    if -1e-12 <= gap < 0:
        gap = 0.0
    elif gap < 0:
        raise FloatingPointError("Material negative dual gap indicates a certificate error")

    return Certificate(stage, iteration, float(lam), float(primal), float(dual), float(kkt), gap)


def soft_threshold_singular(X, t):
    """Singular-value soft-thresholding: S_t(X) = U diag(max(s-t,0)) V^T."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s = np.maximum(s - t, 0.0)
    return (U * s) @ Vt


def as_matrix_operator(A):
    """Wrap an operator whose matvec/rmatvec map matrices<->vectors."""
    if hasattr(A, "matvec") and hasattr(A, "rmatvec") and hasattr(A, "shape"):
        return A
    raise TypeError("A must implement matvec/rmatvec and shape")
