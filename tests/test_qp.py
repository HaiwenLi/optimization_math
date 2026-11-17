import numpy as np
import pytest

from optimization_math.certificates_qp import compute_certificate_qp
from optimization_math.solvers.qp_box import solve_qp_box


def test_diagonal_q_analytic_solution():
    Q = np.diag([2.0, 4.0, 1.0])
    c = np.array([-1.0, 6.0, -0.5])
    l = np.array([0.0, -10.0, -10.0])
    u = np.array([10.0, 10.0, 10.0])
    x_star = np.clip(-c / np.diag(Q), l, u)
    r = solve_qp_box(Q, c, l, u, certificate_stride=1, kkt_tol=1e-8, gap_tol=1e-8)
    assert r.status == "converged"
    np.testing.assert_allclose(r.x, x_star, atol=1e-6)


def test_nnls_matches_nonnegative_least_squares():
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    y = np.array([2.0, 1.0])
    Q = A.T @ A
    c = -A.T @ y
    r = solve_qp_box(Q, c, l=np.zeros(2), certificate_stride=1, kkt_tol=1e-8, gap_tol=1e-8)
    assert r.status == "converged"
    np.testing.assert_allclose(r.x, [1.0, 1.0], atol=1e-6)


def test_certificate_tight_at_optimum():
    Q = np.diag([2.0, 4.0])
    c = np.array([-1.0, 6.0])
    l = np.zeros(2)
    u = np.full(2, 10.0)
    x_star = np.clip(-c / np.diag(Q), l, u)
    cert = compute_certificate_qp(Q, c, x_star, l, u)
    assert cert.kkt_residual == pytest.approx(0.0, abs=1e-10)
    assert cert.relative_gap == pytest.approx(0.0, abs=1e-10)


def test_unconstrained_qp_reduces_to_linear_solve():
    Q = np.array([[3.0, 1.0], [1.0, 2.0]])
    c = np.array([1.0, -1.0])
    r = solve_qp_box(Q, c, certificate_stride=1, kkt_tol=1e-8, gap_tol=1e-8)
    x_star = np.linalg.solve(Q, -c)
    assert r.status == "converged"
    np.testing.assert_allclose(r.x, x_star, atol=1e-6)


def test_rejects_bad_bounds():
    with pytest.raises(ValueError):
        solve_qp_box(np.eye(2), np.zeros(2), l=np.array([1.0, 0.0]), u=np.array([0.0, 1.0]))
