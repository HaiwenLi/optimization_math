"""Forward-difference operator D: (Dx)_i = x_{i+1} - x_i, shape (n-1, n).

Implemented as an implicit LinearOperator (matvec/rmatvec only, no dense
matrix), consistent with the ``optimization_math.operators`` protocol.
"""
import numpy as np


class FirstDifferenceOperator:
    def __init__(self, n):
        n = int(n)
        if n < 2:
            raise ValueError("first difference requires n >= 2")
        self._n = n

    @property
    def shape(self):
        return (self._n - 1, self._n)

    def matvec(self, x):
        x = np.asarray(x, dtype=np.float64)
        return x[1:] - x[:-1]

    def rmatvec(self, y):
        y = np.asarray(y, dtype=np.float64)
        out = np.zeros(self._n, dtype=np.float64)
        out[:-1] -= y
        out[1:] += y
        return out


def first_difference_operator(n):
    return FirstDifferenceOperator(n)
