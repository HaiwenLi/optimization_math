from dataclasses import dataclass
from typing import Literal

import numpy as np

from optimization_math.certificates import Certificate, _vector, compute_certificate
from optimization_math.operators import as_linear_operator
from optimization_math.solvers._common import (
    _positive_float,
    _positive_int,
    _validate_common,
)


@dataclass(frozen=True)
class AdmmIteration:
    iteration: int
    primal_residual: float
    dual_residual: float
    rho: float
    cg_iterations: int


@dataclass(frozen=True)
class AdmmResult:
    x: np.ndarray
    status: Literal["converged", "max_iter"]
    iterations: int
    rho: float
    cg_iterations: int
    certificate: Certificate
    history: tuple[AdmmIteration, ...]
    certificates: tuple[Certificate, ...]


def _nonnegative_l1_z_step(x, u, *, lam, rho):
    return np.maximum(0.0, x + u - lam / rho)


def _normal_operator_matvec(op, v, shift):
    return op.rmatvec(op.matvec(v)) + shift * v


def _cg_ata(op, rhs, *, shift, x0, tol, max_iter):
    rhs = _vector(rhs, "rhs")
    x = _vector(x0, "x0").copy()
    if rhs.shape != (op.shape[1],) or x.shape != (op.shape[1],):
        raise ValueError("CG vectors must match the operator input dimension")
    rhs_norm = np.linalg.norm(rhs)
    if rhs_norm == 0.0:
        return np.zeros_like(rhs), 0, 0.0
    scale = max(1.0, rhs_norm)
    r = rhs - _normal_operator_matvec(op, x, shift)
    p = r.copy()
    rs = float(np.dot(r, r))
    relative = np.sqrt(rs) / scale
    if relative <= tol:
        return x, 0, float(relative)
    for iteration in range(1, max_iter + 1):
        np_p = _normal_operator_matvec(op, p, shift)
        denominator = float(np.dot(p, np_p))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise FloatingPointError("CG encountered a non-SPD normal operator")
        alpha = rs / denominator
        x = x + alpha * p
        r = r - alpha * np_p
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(r)):
            raise FloatingPointError("CG produced a non-finite iterate")
        rs_new = float(np.dot(r, r))
        relative = np.sqrt(rs_new) / scale
        if relative <= tol:
            return x, iteration, float(relative)
        beta = rs_new / rs
        p = r + beta * p
        rs = rs_new
    return x, max_iter, float(relative)
