"""Tests for GAP-safe screening (``optimization_math.screening``)."""
import numpy as np
import pytest

from optimization_math.screening import (
    column_norms_squared,
    compute_safe_screen,
)
from optimization_math.operators import as_linear_operator
from optimization_math.solvers.fista_signed import solve_fista_signed


def _make_problem(seed=0, m=40, n=60, k=6, noise=0.01):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    x_true = np.zeros(n)
    sup = rng.choice(n, size=k, replace=False)
    x_true[sup] = rng.standard_normal(k)
    y = A @ x_true + noise * rng.standard_normal(m)
    lam = 0.1 * np.linalg.norm(A.T @ y, ord=np.inf)
    return A, y, x_true, lam


def test_column_norms_squared_matches_dense():
    A = np.random.default_rng(0).standard_normal((10, 7))
    op = as_linear_operator(A)
    want = np.sum(A ** 2, axis=0)
    got = column_norms_squared(op)
    assert np.allclose(got, want, rtol=1e-10, atol=1e-12)


def test_safe_screen_never_removes_opt_support_at_optimum():
    """Safety: at the converged optimum, screening must not remove any
    coordinate that is nonzero AT THE OPTIMUM. (It may remove weak generating
    truth coordinates that LASSO legitimately zeroes; that is not a violation.)"""
    A, y, x_true, lam = _make_problem(seed=0)
    res = solve_fista_signed(A, y, lam, kkt_tol=1e-8, gap_tol=1e-8,
                             certificate_stride=1, max_iter=20000)
    assert res.status == "converged"
    screened = compute_safe_screen(A, y, res.x, lam)
    opt_support = np.abs(res.x) > 1e-8
    assert not np.any(screened & opt_support), (
        f"screening removed optimum support: {np.where(screened & opt_support)[0]}"
    )


def test_safe_screen_zero_gap_screens_all_inactive():
    """Effectiveness: at the exact optimum the gap is ~0, so every inactive
    coordinate with |a_i^T nu| < lam is screened. Verify screening is nonempty
    and consistent with the certificate's active set."""
    A, y, x_true, lam = _make_problem(seed=1, m=40, n=60, k=4)
    res = solve_fista_signed(A, y, lam, kkt_tol=1e-10, gap_tol=1e-10,
                             certificate_stride=1, max_iter=20000)
    screened = compute_safe_screen(A, y, res.x, lam)
    nonzero = np.abs(res.x) > 1e-8
    # Screened coordinates must all be zero in the optimum (safety again).
    assert not np.any(screened & nonzero)
    # And there must be at least some screened-out coordinates (sparse truth).
    assert int(np.sum(screened)) > 0


def test_safe_screen_rejects_bad_input():
    A, y, x_true, lam = _make_problem(seed=0, m=10, n=15)
    with pytest.raises(ValueError):
        compute_safe_screen(A, y, x_true, -1.0)  # lam <= 0
    with pytest.raises(ValueError):
        compute_safe_screen(A, y, x_true, 0.0)  # lam == 0
    with pytest.raises(ValueError):
        compute_safe_screen(A, y, x_true[:-1], lam)  # shape mismatch
    with pytest.raises(ValueError):
        compute_safe_screen(A, y, x_true, lam, mu=-1.0)  # mu < 0
    with pytest.raises(NotImplementedError):
        compute_safe_screen(A, y, x_true, lam, mu=0.1)  # elastic net unsupported
    with pytest.raises(ValueError):
        compute_safe_screen(A, y, x_true, lam, gap=-1.0)  # negative gap
    with pytest.raises(ValueError):
        compute_safe_screen(A, y, x_true, lam, gap=np.nan)  # nan gap


def test_screen_matches_explicit_radius_formula():
    """Regression for the radius formula: the safe screen must equal the
    explicit GAP-safe rule r_ball = sqrt(2*gap) with nu = theta*r.

    This is evaluated at a *non-converged* iterate (theta < 1), where the old
    (wrong) formula sqrt(2*gap/theta^2) would produce a strictly *larger* radius
    and hence screen strictly *fewer* coordinates -- so this test catches the
    theta factor specifically, which the converged-iterate tests cannot.
    """
    A, y, x_true, lam = _make_problem(seed=2, m=40, n=60, k=6)
    # A genuinely non-optimal starting point: x = 0 (all coordinates inactive).
    x = np.zeros_like(x_true)
    r = A @ x - y
    q = A.T @ r
    q_inf = np.linalg.norm(q, ord=np.inf)
    theta = 1.0 if q_inf <= lam else lam / q_inf
    assert theta < 1.0, "precondition failed: need theta < 1 at x=0"

    # Explicit GAP-safe rule (correct formula).
    from optimization_math.certificates_signed import compute_certificate_signed
    cert = compute_certificate_signed(A, y, x, lam)
    gap = max(cert.primal - cert.dual, 0.0)
    col_norms = np.sqrt(column_norms_squared(A))
    nu = theta * r
    r_ball = float(np.sqrt(2.0 * gap))
    expected = (np.abs(A.T @ nu) + col_norms * r_ball) < lam

    got = compute_safe_screen(A, y, x, lam)
    assert np.array_equal(got, expected), (
        "compute_safe_screen must implement r_ball = sqrt(2*gap) (no theta factor)"
    )
