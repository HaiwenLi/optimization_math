from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class LinearOperator(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...
    def matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def rmatvec(self, y: NDArray[np.float64]) -> NDArray[np.float64]: ...


class _DenseOperator:
    def __init__(self, matrix: NDArray):
        if matrix.ndim != 2:
            raise ValueError("A dense operator must be a two-dimensional matrix")
        self._matrix = matrix

    @property
    def shape(self) -> tuple[int, int]:
        return self._matrix.shape

    def matvec(self, x):
        return self._matrix @ x

    def rmatvec(self, y):
        return self._matrix.T @ y


class _CheckedOperator:
    def __init__(self, op):
        self._op = op
        m, n = op.shape
        if m <= 0 or n <= 0:
            raise ValueError("Operator dimensions must be positive")
        self._shape = (int(m), int(n))

    @property
    def shape(self):
        return self._shape

    def matvec(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.shape[1],):
            raise ValueError(f"matvec input must have shape {(self.shape[1],)}")
        out = np.asarray(self._op.matvec(x), dtype=np.float64)
        if out.shape != (self.shape[0],):
            raise ValueError(f"matvec output must have shape {(self.shape[0],)}")
        return out

    def rmatvec(self, y):
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (self.shape[0],):
            raise ValueError(f"rmatvec input must have shape {(self.shape[0],)}")
        out = np.asarray(self._op.rmatvec(y), dtype=np.float64)
        if out.shape != (self.shape[1],):
            raise ValueError(f"rmatvec output must have shape {(self.shape[1],)}")
        return out


def as_linear_operator(A) -> LinearOperator:
    if isinstance(A, np.ndarray):
        A = _DenseOperator(A)
    if not isinstance(A, LinearOperator):
        raise TypeError("A must be a dense matrix or implement LinearOperator")
    return _CheckedOperator(A)
