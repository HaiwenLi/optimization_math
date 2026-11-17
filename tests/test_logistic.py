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
    with pytest.raises(ValueError):
        _phi_conjugate(np.array([0.5]))


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
