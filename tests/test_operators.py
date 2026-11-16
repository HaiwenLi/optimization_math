import numpy as np
import pytest

from optimization_math.operators import as_linear_operator


def test_dense_operator_matches_matrix_products_and_adjoint_identity():
    A = np.array([[1.0, 2.0, -1.0], [0.0, 3.0, 4.0]])
    op = as_linear_operator(A)
    x = np.array([2.0, -1.0, 0.5])
    y = np.array([1.5, -2.0])

    np.testing.assert_allclose(op.matvec(x), A @ x)
    np.testing.assert_allclose(op.rmatvec(y), A.T @ y)
    assert np.vdot(op.matvec(x), y) == pytest.approx(np.vdot(x, op.rmatvec(y)))


def test_operator_rejects_wrong_input_and_output_shapes():
    A = np.ones((2, 3))
    op = as_linear_operator(A)
    with pytest.raises(ValueError, match="input"):
        op.matvec(np.ones(2))

    class BadOutput:
        shape = (2, 3)

        def matvec(self, x):
            return np.ones(3)

        def rmatvec(self, y):
            return np.ones(2)

    with pytest.raises(ValueError, match="output"):
        as_linear_operator(BadOutput()).matvec(np.ones(3))
