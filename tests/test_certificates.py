import numpy as np
import pytest

from optimization_math.certificates import compute_certificate


def test_scalar_lasso_optimum_has_zero_kkt_and_dual_gap():
    cert = compute_certificate(
        np.array([[1.0]]), np.array([2.0]), np.array([1.5]), 0.5
    )
    assert cert.kkt_residual == pytest.approx(0.0, abs=1e-12)
    assert cert.primal == pytest.approx(0.875)
    assert cert.dual == pytest.approx(0.875)
    assert cert.relative_gap == pytest.approx(0.0, abs=1e-12)


def test_scalar_elastic_net_optimum_has_strong_duality():
    # x=0.75 solves x-2+x+0.5=0; p=d=1.4375.
    cert = compute_certificate(
        np.array([[1.0]]), np.array([2.0]), np.array([0.75]), 0.5, mu=1.0
    )
    assert cert.kkt_residual == pytest.approx(0.0, abs=1e-12)
    assert cert.primal == pytest.approx(1.4375)
    assert cert.dual == pytest.approx(1.4375)
    assert cert.relative_gap == pytest.approx(0.0, abs=1e-12)


def test_certificate_rejects_negative_primal_iterate():
    with pytest.raises(ValueError, match="nonnegative"):
        compute_certificate(
            np.array([[1.0]]), np.array([1.0]), np.array([-0.1]), 0.5
        )


def test_nontrivial_iterate_has_feasible_dual_lower_bound():
    cert = compute_certificate(
        np.array([[1.0, 0.5], [0.2, -1.0]]),
        np.array([1.0, -0.5]),
        np.array([0.2, 0.0]),
        0.4,
        mu=0.3,
    )
    assert cert.primal >= cert.dual
    assert cert.relative_gap >= 0
