import numpy as np
import pytest

from optimization_math.certificates_signed import compute_certificate_signed
from optimization_math.solvers._common import _soft_threshold_prox
from optimization_math.solvers.fista_signed import (
    lambda_max_signed,
    solve_fista_signed,
)


def test_soft_threshold_prox_recovers_signed_soft_threshold():
    z = np.array([-2.0, 0.5, 2.0])
    # step*lam = 0.5*0.5 = 0.25; S_0.25(0.5) = 0.25.
    np.testing.assert_allclose(
        _soft_threshold_prox(z, step=0.5, lam=0.5, mu=0.0),
        [-1.75, 0.25, 1.75],
    )
    np.testing.assert_allclose(
        _soft_threshold_prox(z, step=0.5, lam=0.5, mu=2.0),
        np.array([-1.75, 0.25, 1.75]) / 2.0,
    )


def test_lambda_max_signed_is_dual_norm_of_aty():
    A = np.diag([1.0, 2.0])
    y = np.array([3.0, 1.0])
    assert lambda_max_signed(A, y) == pytest.approx(3.0)


def test_signed_lasso_recovers_negative_signed_solution():
    # Hand-derived: x* = sign(y) * max(0, |y| - lam) for A = [[1]].
    result = solve_fista_signed(
        np.array([[1.0]]), np.array([-2.0]), 0.5,
        certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [-1.5], atol=1e-6)
    assert result.certificate.kkt_residual <= 1e-7
    assert result.certificate.relative_gap <= 1e-8


def test_signed_lasso_recovers_positive_solution_consistent_with_nonnegative():
    result = solve_fista_signed(
        np.array([[1.0]]), np.array([2.0]), 0.5,
        certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [1.5], atol=1e-6)


def test_signed_elastic_net_solution():
    result = solve_fista_signed(
        np.array([[1.0]]), np.array([2.0]), 0.5, mu=1.0,
        certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [0.75], atol=1e-6)


def test_signed_lasso_recovers_mixed_sign_diagonal_solution():
    result = solve_fista_signed(
        np.diag([1.0, 2.0]), np.array([3.0, -1.0]), 0.5,
        lipschitz=4.0, certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [2.5, -0.375], atol=1e-6)


def test_signed_lasso_returns_zero_at_lambda_max():
    result = solve_fista_signed(
        np.diag([1.0, 2.0]), np.array([3.0, 1.0]), 3.0,
        lipschitz=4.0, certificate_stride=1,
    )
    assert result.status == "converged"
    assert result.iterations == 0
    np.testing.assert_array_equal(result.x, np.zeros(2))


def test_signed_lasso_recovers_signed_sparse_support_including_negative_weights():
    rng = np.random.default_rng(7)
    m, n, k = 40, 100, 10
    A = rng.standard_normal((m, n))
    x_true = np.zeros(n)
    support = rng.choice(n, size=k, replace=False)
    x_true[support] = rng.standard_normal(k)  # includes negative entries
    y = A @ x_true + 0.01 * rng.standard_normal(m)
    lam = 0.05 * np.linalg.norm(A.T @ y, ord=np.inf)

    result = solve_fista_signed(
        A, y, lam, lipschitz=float(np.linalg.norm(A, 2) ** 2 * 1.2),
        certificate_stride=1, kkt_tol=1e-5, gap_tol=1e-5, max_iter=50000,
    )
    assert result.status == "converged"
    assert result.certificate.kkt_residual <= 1e-5
    assert result.certificate.relative_gap <= 1e-5
    # The recovered support must overlap the true negative-weight coordinates,
    # which the nonnegative solver would have zeroed out.
    negative_true = support[x_true[support] < 0]
    recovered_negative = np.where(result.x < 0)[0]
    assert set(negative_true) & set(recovered_negative)


def test_certificate_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        compute_certificate_signed(
            np.array([[1.0]]), np.array([1.0, 2.0]), np.array([0.5]), 0.5
        )


def test_certificate_scalar_signed_optimum_has_zero_kkt_and_gap():
    cert = compute_certificate_signed(
        np.array([[1.0]]), np.array([-2.0]), np.array([-1.5]), 0.5
    )
    assert cert.kkt_residual == pytest.approx(0.0, abs=1e-12)
    assert cert.primal == pytest.approx(0.875)
    assert cert.dual == pytest.approx(0.875)
    assert cert.relative_gap == pytest.approx(0.0, abs=1e-12)
