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
