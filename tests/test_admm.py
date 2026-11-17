import numpy as np
import pytest

from optimization_math.solvers.admm import _cg_ata, _nonnegative_l1_z_step


class DiagonalOperator:
    shape = (2, 2)

    def matvec(self, x):
        return np.array([x[0], 2.0 * x[1]])

    def rmatvec(self, y):
        return np.array([y[0], 2.0 * y[1]])


def test_admm_z_step_is_nonnegative_soft_threshold_without_elastic_net_denominator():
    x = np.array([-2.0, 0.5, 2.0])
    u = np.array([0.5, 0.5, 0.5])
    np.testing.assert_allclose(
        _nonnegative_l1_z_step(x, u, lam=0.5, rho=2.0),
        [0.0, 0.75, 2.25],
    )


def test_cg_ata_solves_diagonal_normal_equations_without_dense_access():
    rhs = np.array([3.0, 1.0])
    x, iterations, rel = _cg_ata(
        DiagonalOperator(), rhs, shift=0.5, x0=np.zeros(2), tol=1e-12, max_iter=10
    )
    np.testing.assert_allclose(x, [2.0, 1.0 / 4.5], rtol=1e-10, atol=1e-12)
    assert iterations <= 2
    assert rel <= 1e-12


def test_cg_ata_zero_rhs_returns_zero_without_iterations():
    x, iterations, rel = _cg_ata(
        DiagonalOperator(), np.zeros(2), shift=1.0, x0=np.ones(2), tol=1e-12, max_iter=10
    )
    np.testing.assert_array_equal(x, np.zeros(2))
    assert iterations == 0
    assert rel == 0.0


from optimization_math.certificates import compute_certificate
from optimization_math.solvers.admm import solve_admm
from optimization_math.solvers.fista import lambda_max


def test_admm_recovers_hand_derived_scalar_lasso_solution():
    result = solve_admm(
        np.array([[1.0]]), np.array([2.0]), 0.5,
        rho=1.0, cg_tol=1e-12, residual_tol=1e-6,
        certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8, max_iter=200,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [1.5], atol=1e-6)
    assert result.certificate.kkt_residual <= 1e-7
    assert result.certificate.relative_gap <= 1e-8


def test_admm_recovers_scalar_elastic_net_solution():
    result = solve_admm(
        np.array([[1.0]]), np.array([2.0]), 0.5, mu=1.0,
        rho=1.0, cg_tol=1e-12, residual_tol=1e-6,
        certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8, max_iter=200,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [0.75], atol=1e-6)


def test_admm_applies_known_background_before_fitting():
    result = solve_admm(
        np.array([[1.0]]), np.array([2.0]), 0.5, b=np.array([1.0]),
        rho=1.0, cg_tol=1e-12, residual_tol=1e-6,
        certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8, max_iter=200,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [0.5], atol=1e-6)


def test_admm_returns_zero_at_lambda_max_without_false_iteration_work():
    A = np.diag([1.0, 2.0])
    y = np.array([3.0, 1.0])
    result = solve_admm(A, y, lambda_max(A, y), rho=1.0)
    assert result.status == "converged"
    assert result.iterations == 0
    np.testing.assert_array_equal(result.x, np.zeros(2))


def test_admm_iteration_limit_does_not_claim_convergence():
    result = solve_admm(
        np.array([[1.0]]), np.array([2.0]), 0.1,
        rho=1.0, max_iter=1, certificate_stride=20,
        residual_tol=1e-12, kkt_tol=1e-12, gap_tol=1e-12,
    )
    assert result.status == "max_iter"
    assert result.iterations == 1
    assert np.isfinite(result.certificate.relative_gap)
    assert len(result.history) == 1


def test_admm_uses_black_box_operator_without_dense_matrix_access():
    result = solve_admm(
        DiagonalOperator(), np.array([3.0, 1.0]), 0.5,
        rho=1.0, cg_tol=1e-12, residual_tol=1e-6,
        certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8, max_iter=500,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [2.5, 0.375], atol=1e-6)


@pytest.mark.parametrize(
    "overrides",
    [{"lam": 0.0}, {"lam": 0.1, "mu": -1.0}, {"rho": 0.0},
     {"max_iter": 0}, {"cg_max_iter": 0}, {"certificate_stride": 0},
     {"residual_tol": 0.0}, {"kkt_tol": 0.0}, {"gap_tol": 0.0},
     {"y": np.array([1.0, 2.0])}, {"y": np.array([np.nan])}],
)
def test_admm_rejects_invalid_shapes_nonfinite_data_and_controls(overrides):
    kwargs = dict(A=np.array([[1.0]]), y=np.array([1.0]), lam=0.1)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        solve_admm(**kwargs)


def test_admm_rejects_negative_warm_start():
    with pytest.raises(ValueError, match="nonnegative"):
        solve_admm(np.array([[1.0]]), np.array([1.0]), 0.1, x0=np.array([-1.0]))


def test_admm_convergence_helper_requires_residuals_and_both_certificates():
    from optimization_math.solvers.admm import _admm_has_converged
    cert = compute_certificate(np.array([[1.0]]), np.array([2.0]), np.array([1.5]), 0.5)
    assert _admm_has_converged(0.0, 0.0, cert, 1e-6, 1e-6, 1e-6)
    assert not _admm_has_converged(1.0, 0.0, cert, 1e-6, 1e-6, 1e-6)
    assert not _admm_has_converged(0.0, 1.0, cert, 1e-6, 1e-6, 1e-6)
    assert not _admm_has_converged(0.0, 0.0, cert, 1e-6, 1e-6, -1.0)
    # A certificate at a non-optimal point has kkt_residual above tolerance even with
    # zero ADMM residuals; the KKT gate must still block convergence.
    bad_kkt = compute_certificate(np.array([[1.0]]), np.array([2.0]), np.array([0.01]), 0.5)
    assert bad_kkt.kkt_residual > 1e-6
    assert not _admm_has_converged(0.0, 0.0, bad_kkt, 1e-6, 1e-6, 1e-6)
    # A certificate with a material positive gap must also block convergence.
    bad_gap = compute_certificate(np.array([[1.0]]), np.array([2.0]), np.array([1.0]), 0.5)
    assert bad_gap.relative_gap > 1e-8
    assert not _admm_has_converged(0.0, 0.0, bad_gap, 1e-6, 1e-6, 1e-8)


def test_balance_rho_preserves_unscaled_dual_when_not_clamped():
    from optimization_math.solvers.admm import _balance_rho
    u = np.array([0.5, -2.0])
    # primal (20.0) strictly exceeds rho_balance * dual (10.0 * 1.0), so rho doubles.
    new_rho, new_u = _balance_rho(
        1.0, u, 20.0, 1.0,
        rho_min=1e-3, rho_max=1e3, rho_balance=10.0, rho_scale=2.0,
    )
    assert new_rho == 2.0
    np.testing.assert_allclose(new_rho * new_u, 1.0 * u)


def test_balance_rho_respects_bounds():
    from optimization_math.solvers.admm import _balance_rho
    u = np.ones(2)
    high_rho, high_u = _balance_rho(
        1e3, u, 100.0, 1.0,
        rho_min=1e-3, rho_max=1e3, rho_balance=10.0, rho_scale=2.0,
    )
    low_rho, low_u = _balance_rho(
        1e-3, u, 1.0, 100.0,
        rho_min=1e-3, rho_max=1e3, rho_balance=10.0, rho_scale=2.0,
    )
    assert high_rho == 1e3
    assert low_rho == 1e-3
    np.testing.assert_allclose(high_rho * high_u, 1e3 * u)
    np.testing.assert_allclose(low_rho * low_u, 1e-3 * u)


def test_adaptive_rho_stays_bounded_and_converges():
    result = solve_admm(
        np.array([[1.0]]), np.array([2.0]), 0.5,
        rho=0.25, adaptive_rho=True, rho_min=0.1, rho_max=4.0,
        rho_balance=2.0, rho_scale=2.0, cg_tol=1e-12,
        residual_tol=1e-6, certificate_stride=1,
        kkt_tol=1e-7, gap_tol=1e-8, max_iter=300,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [1.5], atol=1e-6)
    assert all(0.1 <= item.rho <= 4.0 for item in result.history)
    assert 0.1 <= result.rho <= 4.0


def test_adaptive_rho_converges_on_scalar_lasso():
    # Fail-first anchor: Task 3 always raises NotImplementedError when adaptive_rho=True,
    # so this test fails (propagates the error) until Task 4 removes the guard and
    # adaptive rho converges to the scalar optimum.
    result = solve_admm(np.array([[1.0]]), np.array([2.0]), 0.5, adaptive_rho=True)
    assert result.status == "converged"


@pytest.mark.parametrize(
    "overrides",
    [{"adaptive_rho": "yes"}, {"rho_min": 0.0}, {"rho_max": 0.0},
     {"rho_min": 2.0, "rho_max": 1.0}, {"rho_balance": 0.5},
     {"rho_scale": 1.0}],
)
def test_admm_rejects_invalid_adaptive_rho_controls(overrides):
    kwargs = dict(A=np.array([[1.0]]), y=np.array([1.0]), lam=0.1)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        solve_admm(**kwargs)
