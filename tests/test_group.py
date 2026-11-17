import numpy as np
import pytest

from optimization_math.certificates_group import compute_certificate_group
from optimization_math.solvers.fista_group import solve_fista_group


def test_single_group_analytic_solution():
    y = np.array([3.0, 4.0])
    lam = 2.0
    x_star = y * max(0.0, 1.0 - lam / np.linalg.norm(y))
    r = solve_fista_group(np.eye(2), y, lam, [np.array([0, 1])],
                          lipschitz=1.0, certificate_stride=1,
                          kkt_tol=1e-8, gap_tol=1e-9)
    assert r.status == "converged"
    np.testing.assert_allclose(r.x, x_star, atol=1e-6)


def test_group_joint_selection_effect():
    rng = np.random.default_rng(0)
    n = 9
    groups = [np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8])]
    A = rng.standard_normal((6, n))
    x_true = np.zeros(n)
    x_true[0:3] = rng.standard_normal(3)  # only group 0 active
    y = A @ x_true + 0.01 * rng.standard_normal(6)
    lam = 0.1
    r = solve_fista_group(A, y, lam, groups,
                          lipschitz=float(np.linalg.norm(A, 2) ** 2 * 1.2),
                          certificate_stride=1, kkt_tol=1e-5, gap_tol=1e-5, max_iter=20000)
    assert r.status == "converged"
    # Inactive groups should be driven to (near) zero.
    assert np.linalg.norm(r.x[3:]) < 0.1
    # Active group recovered with low relative error.
    assert np.linalg.norm(r.x[0:3] - x_true[0:3]) / np.linalg.norm(x_true[0:3]) < 0.2


def test_certificate_tight_at_group_optimum():
    y = np.array([3.0, 4.0])
    lam = 2.0
    x_star = y * max(0.0, 1.0 - lam / np.linalg.norm(y))
    cert = compute_certificate_group(np.eye(2), y, x_star, lam, [np.array([0, 1])])
    assert cert.kkt_residual == pytest.approx(0.0, abs=1e-10)
    assert cert.relative_gap == pytest.approx(0.0, abs=1e-10)


def test_rejects_bad_groups():
    with pytest.raises(ValueError):
        solve_fista_group(np.eye(2), np.array([1.0, 2.0]), 1.0, [np.array([0, 5])],
                          lipschitz=1.0)
