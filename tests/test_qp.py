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


def test_unbounded_box_certificate_is_a_true_lower_bound():
    # f(x)=5e-13*x^2-1e-6*x has x*=1e6 and f*=-1/2.
    cert = compute_certificate_qp([[1e-12]], [-1e-6], [0.0], None, None)
    assert cert.dual <= -0.5 + 1e-12
    assert cert.relative_gap >= 0.5 - 1e-12


def test_small_gradient_does_not_certify_far_from_qp_optimum():
    result = solve_qp_box(np.array([[1e-12]]), np.array([-1e-6]),
                          certificate_stride=1)
    assert result.status == "converged"
    assert result.iterations > 0
    np.testing.assert_allclose(result.x, [1e6], rtol=0.02)
    assert result.certificate.primal == pytest.approx(-0.5, abs=1e-4)


def test_linear_unbounded_qp_has_no_finite_lower_bound():
    cert = compute_certificate_qp([[0.0]], [-1.0], [0.0], None, None)
    assert cert.dual == -np.inf
    assert cert.relative_gap == np.inf


@pytest.mark.parametrize("l,u", [(None, None), ([-np.inf, 0.0], [0.0, np.inf])])
def test_roundoff_does_not_turn_singular_q_into_positive_curvature(l, u):
    # Q=[1,3].T@[1,3], so d=(-3,1) is an exact null direction.
    # c.T@d=-10 makes this objective unbounded below on both boxes.
    cert = compute_certificate_qp([[1.0, 3.0], [3.0, 9.0]],
                                  [3.0, -1.0], [0.0, 0.0], l, u)
    assert cert.dual == -np.inf
    assert cert.relative_gap == np.inf


@pytest.mark.parametrize("l,u", [([np.nan], [1.0]), ([0.0], [np.nan]),
                                ([np.inf], [np.inf]), ([-np.inf], [-np.inf])])
def test_certificate_rejects_invalid_bounds(l, u):
    with pytest.raises(ValueError):
        compute_certificate_qp([[1.0]], [0.0], [0.0], l, u)
