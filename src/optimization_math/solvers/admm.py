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


def _normalized_residuals(x, z, z_previous, u, rho):
    primal = np.linalg.norm(x - z) / max(1.0, np.linalg.norm(x), np.linalg.norm(z))
    dual = rho * np.linalg.norm(z - z_previous) / max(1.0, rho * np.linalg.norm(u))
    return float(primal), float(dual)


def _admm_has_converged(primal_residual, dual_residual, certificate,
                        residual_tol, kkt_tol, gap_tol):
    return (
        primal_residual <= residual_tol
        and dual_residual <= residual_tol
        and certificate.kkt_residual <= kkt_tol
        and certificate.relative_gap <= gap_tol
    )


def _result(x, status, iterations, rho, cg_iterations, certificate, history, certificates):
    return AdmmResult(
        x.copy(), status, iterations, float(rho), int(cg_iterations),
        certificate, tuple(history), tuple(certificates)
    )


def _balance_rho(rho, u, primal_residual, dual_residual, *,
                 rho_min, rho_max, rho_balance, rho_scale):
    new_rho = float(rho)
    if primal_residual > rho_balance * dual_residual:
        new_rho = min(rho_max, new_rho * rho_scale)
    elif dual_residual > rho_balance * primal_residual:
        new_rho = max(rho_min, new_rho / rho_scale)
    if new_rho == rho:
        return new_rho, u
    return new_rho, u * (rho / new_rho)


def solve_admm(A, y, lam, *, mu=0.0, b=None, x0=None, rho=1.0,
               adaptive_rho=False, rho_min=1e-3, rho_max=1e3,
               rho_balance=10.0, rho_scale=2.0, max_iter=200,
               cg_max_iter=None, cg_tol=1e-8, residual_tol=1e-4,
               kkt_tol=1e-5, gap_tol=1e-4, certificate_stride=10):
    """Nonnegative LASSO / elastic net via ADMM with CG inner solves.

    There is no Lipschitz parameter (steps are governed by the ADMM penalty
    ``rho``); ``mu`` and the output offset ``b`` are supported. Inner CG
    should run tighter than the outer certificate tolerances.
    """
    max_iter = _positive_int(max_iter, "max_iter")
    certificate_stride = _positive_int(certificate_stride, "certificate_stride")
    cg_tol = _positive_float(cg_tol, "cg_tol")
    residual_tol = _positive_float(residual_tol, "residual_tol")
    kkt_tol = _positive_float(kkt_tol, "kkt_tol")
    gap_tol = _positive_float(gap_tol, "gap_tol")
    rho = _positive_float(rho, "rho")
    rho_min = _positive_float(rho_min, "rho_min")
    rho_max = _positive_float(rho_max, "rho_max")
    rho_balance = _positive_float(rho_balance, "rho_balance")
    rho_scale = _positive_float(rho_scale, "rho_scale")
    if rho_max < rho_min or rho_balance < 1.0 or rho_scale <= 1.0:
        raise ValueError("invalid adaptive rho bounds or gains")
    if not isinstance(adaptive_rho, (bool, np.bool_)):
        raise ValueError("adaptive_rho must be a boolean")
    if cg_max_iter is not None:
        cg_max_iter = _positive_int(cg_max_iter, "cg_max_iter")

    op = as_linear_operator(A)
    y, b_vec, x, lam, mu = _validate_common(op, y, lam, mu, b, x0)
    cg_limit = op.shape[1] if cg_max_iter is None else cg_max_iter
    rhs_const = op.rmatvec(y - b_vec)

    z = x.copy()
    u = np.zeros_like(x)
    z_previous = z.copy()
    history = []
    certificates = [compute_certificate(op, y, z, lam, mu=mu, b=b_vec, stage=0, iteration=0)]
    total_cg_iterations = 0
    if _admm_has_converged(0.0, 0.0, certificates[-1], residual_tol, kkt_tol, gap_tol):
        return _result(z, "converged", 0, rho, 0, certificates[-1], history, certificates)

    for iteration in range(1, max_iter + 1):
        rhs = rhs_const + rho * (z - u)
        x, cg_iterations, cg_relative = _cg_ata(
            op, rhs, shift=mu + rho, x0=x, tol=cg_tol, max_iter=cg_limit
        )
        total_cg_iterations += cg_iterations
        z_previous = z
        z = _nonnegative_l1_z_step(x, u, lam=lam, rho=rho)
        u = u + x - z
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(z)) and np.all(np.isfinite(u))):
            raise FloatingPointError("ADMM produced a non-finite iterate")

        primal, dual = _normalized_residuals(x, z, z_previous, u, rho)
        history.append(AdmmIteration(iteration, primal, dual, float(rho), int(cg_iterations)))
        residual_ready = primal <= residual_tol and dual <= residual_tol
        needs_certificate = residual_ready or iteration % certificate_stride == 0 or iteration == max_iter
        if needs_certificate:
            certificates.append(compute_certificate(
                op, y, z, lam, mu=mu, b=b_vec, stage=0, iteration=iteration
            ))
        if residual_ready and _admm_has_converged(
            primal, dual, certificates[-1], residual_tol, kkt_tol, gap_tol
        ):
            return _result(
                z, "converged", iteration, rho, total_cg_iterations,
                certificates[-1], history, certificates
            )
        if adaptive_rho and iteration < max_iter:
            rho, u = _balance_rho(
                rho, u, primal, dual,
                rho_min=rho_min, rho_max=rho_max,
                rho_balance=rho_balance, rho_scale=rho_scale,
            )

    return _result(
        z, "max_iter", max_iter, rho, total_cg_iterations,
        certificates[-1], history, certificates
    )
