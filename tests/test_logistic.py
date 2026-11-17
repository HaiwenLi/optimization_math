import numpy as np
import pytest

from optimization_math.certificates_logistic import (
    _phi_conjugate,
    compute_certificate_logistic,
)
from optimization_math.solvers.fista_logistic import solve_fista_logistic


def test_single_sample_analytic_solution():
    # N=1, p=1, (x=1, y=+1): optimal w* = -log(lam/(1-lam)) for lam < 1/2,
    # else w* = 0 (gradient at 0 is -1/2, within [-lam, lam]).
    for lam, expected in [(0.1, -np.log(0.1 / 0.9)), (0.3, -np.log(0.3 / 0.7))]:
        res = solve_fista_logistic(
            np.array([[1.0]]), np.array([1.0]), lam,
            certificate_stride=1, kkt_tol=1e-8, gap_tol=1e-9,
        )
        assert res.status == "converged"
        np.testing.assert_allclose(res.w, [expected], atol=1e-5)


def test_single_sample_zero_at_high_lambda():
    res = solve_fista_logistic(
        np.array([[1.0]]), np.array([1.0]), 0.5,
        certificate_stride=1, kkt_tol=1e-8, gap_tol=1e-9,
    )
    assert res.status == "converged"
    assert res.iterations == 0
    np.testing.assert_allclose(res.w, [0.0], atol=1e-8)


def test_multisample_certificate_is_tight_at_zero_optimum():
    cert = compute_certificate_logistic(
        np.ones((2, 1)), np.ones(2), np.zeros(1), 1.0
    )
    assert cert.primal == pytest.approx(np.log(2.0))
    assert cert.dual == pytest.approx(np.log(2.0))
    assert cert.kkt_residual == pytest.approx(0.0, abs=1e-12)
    assert cert.relative_gap == pytest.approx(0.0, abs=1e-12)


def test_multisample_zero_optimum_converges_without_iterations():
    res = solve_fista_logistic(
        np.ones((2, 1)), np.ones(2), 1.0, max_iter=20,
        kkt_tol=1e-8, gap_tol=1e-9,
    )
    assert res.status == "converged"
    assert res.iterations == 0
    np.testing.assert_array_equal(res.w, [0.0])
    assert res.certificate.relative_gap <= 1e-9


def test_zero_design_converges_from_nonzero_warm_start():
    with np.errstate(divide="raise", invalid="raise"):
        res = solve_fista_logistic(
            np.zeros((1, 1)), np.ones(1), 1.0, w0=np.ones(1),
            max_iter=20, certificate_stride=1, kkt_tol=1e-8, gap_tol=1e-9,
        )
    assert res.status == "converged"
    assert np.isfinite(res.lipschitz) and res.lipschitz > 0.0
    np.testing.assert_array_equal(res.w, [0.0])
    assert res.certificate.primal == pytest.approx(np.log(2.0))
    assert res.certificate.relative_gap <= 1e-9


@pytest.mark.parametrize("shape", [(0, 1), (1, 0), (0, 0)])
@pytest.mark.parametrize("certificate", [False, True])
def test_logistic_rejects_empty_design_dimensions(shape, certificate):
    X = np.zeros(shape)
    y = np.ones(shape[0])
    with pytest.raises(ValueError, match="positive dimensions"):
        if certificate:
            compute_certificate_logistic(X, y, np.zeros(shape[1]), 1.0)
        else:
            solve_fista_logistic(X, y, 1.0)


def test_certificate_is_tight_at_analytic_optimum():
    lam = 0.2
    w_star = -np.log(lam / (1 - lam))
    cert = compute_certificate_logistic(
        np.array([[1.0]]), np.array([1.0]), np.array([w_star]), lam
    )
    assert cert.kkt_residual == pytest.approx(0.0, abs=1e-12)
    assert cert.relative_gap == pytest.approx(0.0, abs=1e-12)


def test_phi_conjugate_domain_and_endpoints():
    assert _phi_conjugate(np.array([0.0]))[0] == pytest.approx(0.0)
    assert _phi_conjugate(np.array([-1.0]))[0] == pytest.approx(0.0)
    mid = _phi_conjugate(np.array([-0.5]))[0]
    assert mid == pytest.approx(np.log(0.5))  # phi*(-0.5) = log(0.5) < 0
    # Out-of-domain input is clamped (defensive against float roundoff) rather
    # than rejected: phi*(0.5) clamps to phi*(0) = 0.
    assert _phi_conjugate(np.array([0.5]))[0] == pytest.approx(0.0)


def test_sparse_logistic_recovers_sparse_truth_and_classifies():
    rng = np.random.default_rng(3)
    N, p, k = 200, 20, 5
    X = rng.standard_normal((N, p))
    w_true = np.zeros(p)
    support = rng.choice(p, size=k, replace=False)
    w_true[support] = rng.standard_normal(k)
    # Separable-ish labels with a small label-flip fraction.
    logits = X @ w_true
    y = np.where(logits + 0.3 * rng.standard_normal(N) > 0, 1.0, -1.0)

    lam = 0.02
    res = solve_fista_logistic(
        X, y, lam, certificate_stride=1, kkt_tol=1e-5, gap_tol=1e-5, max_iter=20000,
    )
    assert res.status == "converged"
    assert res.certificate.kkt_residual <= 1e-5
    assert res.certificate.relative_gap <= 1e-5

    # Classification accuracy meaningfully above chance.
    pred = np.sign(X @ res.w)
    acc = np.mean(pred == y)
    assert acc > 0.7


def test_certificate_rejects_bad_shapes():
    with pytest.raises(ValueError):
        compute_certificate_logistic(
            np.array([[1.0], [2.0]]), np.array([1.0]), np.array([0.5]), 0.1
        )
    with pytest.raises(ValueError):
        compute_certificate_logistic(
            np.array([[1.0]]), np.array([1.0]), np.array([np.nan]), 0.1
        )


def test_logistic_rejects_zero_one_labels():
    with pytest.raises(ValueError, match="\\+1 and -1"):
        compute_certificate_logistic(
            np.array([[1.0]]), np.array([0.0]), np.array([0.5]), 0.1
        )
    with pytest.raises(ValueError, match="\\+1 and -1"):
        solve_fista_logistic(np.array([[1.0]]), np.array([0.0]), 0.1)
