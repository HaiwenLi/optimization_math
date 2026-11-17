import numpy as np
import pytest

from optimization_math.operators_difference import first_difference_operator
from optimization_math.solvers.chambolle_tv import solve_tv_chambolle


def test_constant_signal_is_fixed_point():
    y = np.full(30, 3.0)
    r = solve_tv_chambolle(y, 0.5, tol=1e-8)
    assert r.status == "converged"
    np.testing.assert_allclose(r.x, y, atol=1e-6)
    assert r.certificate.relative_gap <= 1e-8


def test_tv_denoise_recovers_piecewise_constant():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 200)
    x_true = np.where(t < 0.3, 0.0, np.where(t < 0.7, 1.0, 0.5))
    y = x_true + 0.1 * rng.standard_normal(200)
    r = solve_tv_chambolle(y, 0.3, tol=1e-4, max_iter=20000)
    assert r.status == "converged"
    # TV denoising must reduce error vs. the raw noisy observation.
    assert np.linalg.norm(r.x - x_true) < np.linalg.norm(y - x_true)
    assert r.certificate.relative_gap <= 1e-4


def test_larger_lam_tv_smooths_more():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 1, 100)
    x_true = np.where(t < 0.5, 0.0, 1.0)
    y = x_true + 0.1 * rng.standard_normal(100)
    x_small = solve_tv_chambolle(y, 0.05, tol=1e-6).x
    x_large = solve_tv_chambolle(y, 1.0, tol=1e-6).x
    tv_small = np.sum(np.abs(first_difference_operator(100).matvec(x_small)))
    tv_large = np.sum(np.abs(first_difference_operator(100).matvec(x_large)))
    assert tv_large < tv_small  # larger lambda -> less total variation


def test_certificate_is_tight_at_denoised_solution():
    rng = np.random.default_rng(2)
    y = np.concatenate([np.zeros(20), np.ones(20)]) + 0.05 * rng.standard_normal(40)
    r = solve_tv_chambolle(y, 0.3, tol=1e-8)
    assert r.certificate.relative_gap <= 1e-8


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        solve_tv_chambolle(np.array([1.0, 2.0, np.nan]), 0.5)
    with pytest.raises(ValueError):
        solve_tv_chambolle(np.array([1.0, 2.0]), -0.1)
