import numpy as np

from optimization_math.certificates_nuclear import (
    compute_certificate_nuclear,
    soft_threshold_singular,
)
from optimization_math.solvers.fista_nuclear import solve_fista_nuclear


class IdentityOp:
    def __init__(self, shape):
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    def matvec(self, X):
        return X.ravel()

    def rmatvec(self, v):
        return v.reshape(self._shape)


def test_nuclear_denoising_matches_singular_value_thresholding():
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((6, 6))
    lam = 1.0
    X_star = soft_threshold_singular(Y, lam)
    A = IdentityOp((6, 6))
    r = solve_fista_nuclear(A, Y.ravel(), lam, lipschitz=1.0,
                            certificate_stride=1, kkt_tol=1e-8, gap_tol=1e-9)
    assert r.status == "converged"
    np.testing.assert_allclose(r.X, X_star, atol=1e-6)


def test_nuclear_recovers_low_rank_structure():
    rng = np.random.default_rng(1)
    m = n = 8
    U = rng.standard_normal((m, 3))
    V = rng.standard_normal((n, 3))
    X_true = U @ V.T  # rank 3
    Y = X_true + 0.05 * rng.standard_normal((m, n))
    lam = 0.1
    A = IdentityOp((m, n))
    r = solve_fista_nuclear(A, Y.ravel(), lam, lipschitz=1.0,
                            certificate_stride=1, kkt_tol=1e-6, gap_tol=1e-7)
    assert r.status == "converged"
    # Denoised matrix should be closer to truth than the raw observation.
    assert np.linalg.norm(r.X - X_true) < np.linalg.norm(Y - X_true)
    # ... and remain low rank (small eigenvalues shrunken to zero).
    s = np.linalg.svd(r.X, compute_uv=False)
    assert np.sum(s > 1e-6) <= 4


def test_nuclear_certificate_tight_at_optimum():
    rng = np.random.default_rng(2)
    Y = rng.standard_normal((5, 5))
    lam = 0.7
    X_star = soft_threshold_singular(Y, lam)
    cert = compute_certificate_nuclear(IdentityOp((5, 5)), Y.ravel(), X_star, lam)
    assert cert.kkt_residual < 1e-8
    assert cert.relative_gap < 1e-8
