import numpy as np

from optimization_math.operators import as_linear_operator
from optimization_math.certificates import _vector


def _nonnegative_elastic_net_prox(z, *, step, lam, mu):
    return np.maximum(0.0, z - step * lam) / (1.0 + step * mu)


def lambda_max(A, y, *, b=None):
    op = as_linear_operator(A)
    y = _vector(y, "y")
    b_vec = np.zeros_like(y) if b is None else _vector(b, "b")
    if y.shape != (op.shape[0],) or b_vec.shape != y.shape:
        raise ValueError("y and b must match the operator output dimension")
    g0 = op.rmatvec(b_vec - y)
    return float(max(0.0, -np.min(g0)))
