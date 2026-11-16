import numpy as np
import pytest

from optimization_math.certificates import compute_certificate
from optimization_math.solvers.fista import (
    _nonnegative_elastic_net_prox,
    lambda_max,
    solve_fista,
)


def test_nonnegative_elastic_net_prox_uses_soft_threshold_and_l2_denominator():
    z = np.array([-2.0, 0.5, 2.0])
    np.testing.assert_allclose(
        _nonnegative_elastic_net_prox(z, step=0.5, lam=0.5, mu=0.0),
        [0.0, 0.25, 1.75],
    )
    np.testing.assert_allclose(
        _nonnegative_elastic_net_prox(z, step=0.5, lam=0.5, mu=2.0),
        [0.0, 0.125, 0.875],
    )


def test_lambda_max_is_threshold_for_zero_stationarity():
    A = np.diag([1.0, 2.0])
    y = np.array([3.0, 1.0])
    assert lambda_max(A, y) == pytest.approx(3.0)


def test_fista_recovers_hand_derived_scalar_lasso_solution():
    result = solve_fista(
        np.array([[1.0]]), np.array([2.0]), 0.5,
        certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [1.5], atol=1e-6)
    assert result.certificate.kkt_residual <= 1e-7
    assert result.certificate.relative_gap <= 1e-8


def test_fista_recovers_scalar_elastic_net_solution():
    result = solve_fista(
        np.array([[1.0]]), np.array([2.0]), 0.5, mu=1.0,
        certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [0.75], atol=1e-6)


def test_fista_returns_zero_at_lambda_max_without_false_iteration_work():
    result = solve_fista(np.diag([1.0, 2.0]), np.array([3.0, 1.0]), 3.0)
    assert result.status == "converged"
    assert result.iterations == 0
    np.testing.assert_array_equal(result.x, np.zeros(2))


def test_fista_iteration_limit_does_not_claim_convergence():
    result = solve_fista(
        np.array([[1.0]]), np.array([2.0]), 0.1,
        max_iter=1, certificate_stride=20, kkt_tol=1e-12, gap_tol=1e-12,
        lipschitz=2.0,
    )
    assert result.status == "max_iter"
    assert result.iterations == 1
    assert np.isfinite(result.certificate.relative_gap)


@pytest.mark.parametrize("kwargs", [{"lam": 0.0}, {"lam": 0.1, "mu": -1.0}])
def test_fista_rejects_invalid_regularization(kwargs):
    with pytest.raises(ValueError):
        solve_fista(np.array([[1.0]]), np.array([1.0]), **kwargs)


def test_fista_rejects_negative_warm_start():
    with pytest.raises(ValueError, match="nonnegative"):
        solve_fista(
            np.array([[1.0]]), np.array([1.0]), 0.1, x0=np.array([-1.0])
        )


def test_fista_uses_black_box_operator_without_dense_matrix_access():
    class DiagonalOperator:
        shape = (2, 2)

        def matvec(self, x):
            return np.array([x[0], 2.0 * x[1]])

        def rmatvec(self, y):
            return np.array([y[0], 2.0 * y[1]])

    result = solve_fista(
        DiagonalOperator(), np.array([3.0, 1.0]), 0.5,
        lipschitz=4.0, certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [2.5, 0.375], atol=1e-6)


def test_fista_applies_known_background_before_fitting():
    result = solve_fista(
        np.array([[1.0]]), np.array([2.0]), 0.5, b=np.array([1.0]),
        certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8,
    )
    np.testing.assert_allclose(result.x, [0.5], atol=1e-6)


@pytest.mark.parametrize(
    "overrides",
    [{"y": np.array([1.0, 2.0])}, {"y": np.array([np.nan])},
     {"max_iter": 0}, {"certificate_stride": 0}, {"lipschitz": 0.0}],
)
def test_fista_rejects_invalid_shapes_nonfinite_data_and_controls(overrides):
    kwargs = dict(A=np.array([[1.0]]), y=np.array([1.0]), lam=0.1)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        solve_fista(**kwargs)


def test_fista_momentum_accelerates_beyond_plain_ista():
    rng = np.random.default_rng(11)
    u, _ = np.linalg.qr(rng.standard_normal((40, 20)))
    v, _ = np.linalg.qr(rng.standard_normal((20, 20)))
    s = np.logspace(3, 0, 20)
    A = (u * s) @ v.T
    y = A @ rng.random(20) + 0.05 * rng.standard_normal(40)
    lam = 0.05 * np.linalg.norm(A.T @ y, ord=np.inf)
    lipschitz = float(s[0] ** 2)
    kkt_tol, gap_tol = 1e-6, 1e-7

    def plain_ista_iterations():
        x = np.zeros(20)
        for k in range(1, 20000):
            grad = A.T @ (A @ x - y)
            x = _nonnegative_elastic_net_prox(
                x - grad / lipschitz, step=1.0 / lipschitz, lam=lam, mu=0.0
            )
            cert = compute_certificate(A, y, x, lam, stage=0, iteration=k)
            if cert.kkt_residual <= kkt_tol and cert.relative_gap <= gap_tol:
                return k
        raise AssertionError("plain ISTA reference did not converge")

    result = solve_fista(
        A, y, lam, lipschitz=lipschitz, restart=False, backtracking=False,
        certificate_stride=1, kkt_tol=kkt_tol, gap_tol=gap_tol, max_iter=20000,
    )
    assert result.status == "converged"
    assert result.iterations < 0.5 * plain_ista_iterations()


@pytest.mark.parametrize(
    "kwargs", [{"lam": None}, {"lam": [0.5]}, {"lam": 0.1, "mu": "auto"}]
)
def test_fista_rejects_nonnumeric_regularization_with_value_error(kwargs):
    with pytest.raises(ValueError):
        solve_fista(np.array([[1.0]]), np.array([1.0]), **kwargs)


def test_backtracking_recovers_from_underestimated_lipschitz():
    result = solve_fista(
        np.eye(2), np.array([2.0, -1.0]), 0.5,
        lipschitz=1e-8, backtracking=True,
        certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [1.5, 0.0], atol=1e-6)
    assert result.lipschitz >= 0.5


def test_gradient_restart_predicate_matches_sign_rule():
    from optimization_math.solvers.fista import _gradient_restart_needed
    assert _gradient_restart_needed(np.array([1.0]), np.array([1.0]), np.array([0.0]))
    assert not _gradient_restart_needed(np.array([1.0]), np.array([-1.0]), np.array([0.0]))


def test_restarted_fista_converges_on_correlated_operator():
    A = np.array([[1.0, 1.0], [1.0, 1.01]])
    result = solve_fista(
        A, np.array([1.0, 2.0]), 0.1, restart=True,
        certificate_stride=1, kkt_tol=1e-6, gap_tol=1e-6, max_iter=3000,
    )
    assert result.status == "converged"
    assert np.all(result.x >= 0)


def test_lambda_path_warm_starts_each_stage_and_finishes_at_target():
    result = solve_fista(
        np.array([[1.0]]), np.array([2.0]), 0.5,
        lambda_path=[2.0, 1.0, 0.5], certificate_stride=1,
        kkt_tol=1e-7, gap_tol=1e-8,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [1.5], atol=1e-6)
    assert result.certificate.lam == 0.5
    assert {cert.stage for cert in result.history} == {0, 1, 2}
    assert [cert.iteration for cert in result.history if cert.stage == 1][0] == 0
    assert result.total_iterations >= result.iterations


def test_lambda_path_must_be_strictly_decreasing_and_end_at_target():
    for path in ([2.0, 1.0, 1.0, 0.5], [2.0, 1.0]):
        with pytest.raises(ValueError, match="lambda_path"):
            solve_fista(np.array([[1.0]]), np.array([2.0]), 0.5, lambda_path=path)
