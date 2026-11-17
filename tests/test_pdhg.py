"""Tests for the primal-dual (Chambolle-Pock) LASSO solver."""
import numpy as np
import pytest

from optimization_math.solvers.pdhg import solve_pdhg
from optimization_math.solvers.fista_signed import solve_fista_signed
from optimization_math.certificates_signed import compute_certificate_signed
from optimization_math.operators import as_linear_operator


def _make_problem(seed=0, m=50, n=80, k=8, noise=0.02):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    x_true = np.zeros(n)
    sup = rng.choice(n, size=k, replace=False)
    x_true[sup] = rng.standard_normal(k)
    y = A @ x_true + noise * rng.standard_normal(m)
    lam = 0.1 * np.linalg.norm(A.T @ y, ord=np.inf)
    lipschitz = float(np.linalg.norm(A, 2) ** 2) * 1.2
    return A, y, x_true, lam, lipschitz


def test_pdhg_converges_and_matches_fista():
    A, y, x_true, lam, lipschitz = _make_problem(seed=0)
    r_pdhg = solve_pdhg(A, y, lam, lipschitz=lipschitz, kkt_tol=1e-6, gap_tol=1e-6,
                        certificate_stride=1, max_iter=20000)
    r_fista = solve_fista_signed(A, y, lam, lipschitz=lipschitz, kkt_tol=1e-6, gap_tol=1e-6,
                                 certificate_stride=1, max_iter=20000)
    assert r_pdhg.status == "converged"
    assert r_fista.status == "converged"
    # Same optimum (the problem is strongly convex in the smooth part; both
    # first-order methods converge to the unique minimizer).
    assert np.allclose(r_pdhg.x, r_fista.x, rtol=1e-3, atol=1e-3)


def test_pdhg_rejects_bad_step_sizes():
    A, y, x_true, lam, lipschitz = _make_problem(seed=0, m=20, n=30, k=3)
    # tau*sigma*||A||^2 > 1 must raise.
    with pytest.raises(ValueError):
        solve_pdhg(A, y, lam, tau=1.0, sigma=1.0 / lipschitz * 100.0)
    with pytest.raises(ValueError):
        solve_pdhg(A, y, lam, theta=0.0)
    with pytest.raises(ValueError):
        solve_pdhg(A, y, lam, theta=2.0)  # theta > 1 unsupported
    with pytest.raises(ValueError):
        solve_pdhg(A, y, lam, lipschitz=0.0)  # invalid lipschitz
    with pytest.raises(ValueError):
        solve_pdhg(A, y, lam, lipschitz=-1.0)
    with pytest.raises(ValueError):
        solve_pdhg(A, y, lam, lipschitz=np.nan)


def test_pdhg_rejects_bad_input():
    A, y, x_true, lam, lipschitz = _make_problem(seed=0, m=20, n=30, k=3)
    with pytest.raises(ValueError):
        solve_pdhg(A, y, -lam)
    with pytest.raises(ValueError):
        solve_pdhg(A, y, lam, mu=-1.0)
    with pytest.raises(ValueError):
        solve_pdhg(A, y[:-1], lam)  # shape mismatch
    with pytest.raises(ValueError):
        solve_pdhg(A, np.r_[y, np.nan], lam)  # non-finite


def test_pdhg_elastic_net_matches_fista():
    """With mu > 0 the certificate and optimum must still agree with FISTA."""
    A, y, x_true, lam, lipschitz = _make_problem(seed=3, m=40, n=60, k=5)
    mu = 0.05
    # PDHG's `lipschitz` means ||A||^2 (the saddle coupling K=A); mu enters only
    # the prox_G denominator. FISTA's `lipschitz` means ||A||^2 + mu (its smooth
    # part includes the elastic-net quadratic).
    r_pdhg = solve_pdhg(A, y, lam, mu=mu, lipschitz=lipschitz,
                        kkt_tol=1e-6, gap_tol=1e-6, certificate_stride=1, max_iter=20000)
    r_fista = solve_fista_signed(A, y, lam, mu=mu, lipschitz=lipschitz + mu,
                                 kkt_tol=1e-6, gap_tol=1e-6, certificate_stride=1, max_iter=20000)
    assert r_pdhg.status == "converged"
    assert r_fista.status == "converged"
    assert np.allclose(r_pdhg.x, r_fista.x, rtol=1e-3, atol=1e-3)


def test_pdhg_certificate_consistent():
    """The certificate evaluated at the returned iterate must be satisfied."""
    A, y, x_true, lam, lipschitz = _make_problem(seed=1, m=30, n=50, k=4)
    r = solve_pdhg(A, y, lam, lipschitz=lipschitz, kkt_tol=1e-6, gap_tol=1e-6,
                   certificate_stride=1, max_iter=20000)
    cert = r.certificate
    assert cert.kkt_residual <= 1e-6
    assert cert.relative_gap <= 1e-6
