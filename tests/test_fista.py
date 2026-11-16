import numpy as np
import pytest

from optimization_math.solvers.fista import (
    _nonnegative_elastic_net_prox,
    lambda_max,
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
