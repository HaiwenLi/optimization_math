from dataclasses import dataclass
from typing import Literal

import numpy as np

from optimization_math.certificates import Certificate
from optimization_math.certificates_group import compute_certificate_group
from optimization_math.operators import as_linear_operator
from optimization_math.solvers._common import _estimate_lipschitz, _positive_float, _positive_int


@dataclass(frozen=True)
class FistaGroupResult:
    x: np.ndarray
    status: Literal["converged", "max_iter"]
    iterations: int
    lipschitz: float
    certificate: Certificate
    history: tuple[Certificate, ...]


def _group_soft_threshold_prox(z, groups, step, lam):
    x = z.copy()
    for s in groups:
        nrm = np.linalg.norm(z[s])
        if nrm > 0:
            shrink = max(0.0, 1.0 - step * lam / nrm)
            x[s] = z[s] * shrink
    return x


def solve_fista_group(A, y, lam, groups, *, mu=0.0, b=None, x0=None, lipschitz=None,
                      max_iter=1000, kkt_tol=1e-5, gap_tol=1e-4,
                      certificate_stride=10, power_iter=50, power_tol=1e-6,
                      restart=True):
    max_iter = _positive_int(max_iter, "max_iter")
    certificate_stride = _positive_int(certificate_stride, "certificate_stride")
    power_iter = _positive_int(power_iter, "power_iter")
    kkt_tol = _positive_float(kkt_tol, "kkt_tol")
    gap_tol = _positive_float(gap_tol, "gap_tol")
    power_tol = _positive_float(power_tol, "power_tol")
    if lipschitz is not None:
        lipschitz = _positive_float(lipschitz, "lipschitz")

    op = as_linear_operator(A)
    y = np.asarray(y, dtype=np.float64)
    b_vec = np.zeros_like(y) if b is None else np.asarray(b, dtype=np.float64)
    if y.ndim != 1 or y.shape != (op.shape[0],) or b_vec.shape != y.shape:
        raise ValueError("y and b must match the operator output dimension")
    waves = _normalize_groups(groups, op.shape[1])
    n = op.shape[1]

    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    if x.shape != (n,):
        raise ValueError("x0 must match the operator input dimension")
    lam = float(lam)
    mu = float(mu)

    L = float(lipschitz) if lipschitz is not None else _estimate_lipschitz(
        op, power_iter=power_iter, power_tol=power_tol
    )

    z = x.copy()
    momentum = 1.0
    history = [compute_certificate_group(op, y, x, lam, waves, mu=mu, b=b_vec, stage=0, iteration=0)]
    status = "max_iter"
    iterations = 0

    if history[-1].kkt_residual <= kkt_tol and history[-1].relative_gap <= gap_tol:
        status = "converged"
        return FistaGroupResult(x, status, iterations, L, history[-1], tuple(history))

    for k in range(1, max_iter + 1):
        r_z = op.matvec(z) + b_vec - y
        grad = op.rmatvec(r_z) + mu * z
        x_new = _group_soft_threshold_prox(z - grad / L, waves, 1.0 / L, lam)
        if not np.all(np.isfinite(x_new)):
            raise FloatingPointError("FISTA produced a non-finite iterate")

        if restart and np.dot(grad, x_new - x) > 0.0:
            r_x = op.matvec(x) + b_vec - y
            grad_x = op.rmatvec(r_x) + mu * x
            x_new = _group_soft_threshold_prox(x - grad_x / L, waves, 1.0 / L, lam)
            if not np.all(np.isfinite(x_new)):
                raise FloatingPointError("FISTA produced a non-finite iterate")
            momentum = 1.0

        if k % certificate_stride == 0 or k == max_iter:
            cert = compute_certificate_group(op, y, x_new, lam, waves, mu=mu, b=b_vec,
                                             stage=0, iteration=k)
            history.append(cert)
            if cert.kkt_residual <= kkt_tol and cert.relative_gap <= gap_tol:
                x = x_new
                iterations = k
                status = "converged"
                break

        next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        z = x_new + ((momentum - 1.0) / next_momentum) * (x_new - x)
        x = x_new
        momentum = next_momentum
        iterations = k

    return FistaGroupResult(x, status, iterations, L, history[-1], tuple(history))


def _normalize_groups(groups, n):
    out = [np.asarray(g, dtype=np.intp) for g in groups]
    if any(g.ndim != 1 or g.size == 0 for g in out):
        raise ValueError("groups must be a list of nonempty index arrays")
    all_idx = np.concatenate(out) if out else np.array([], dtype=np.intp)
    if all_idx.size and (all_idx.min() < 0 or all_idx.max() >= n):
        raise ValueError("group indices out of range")
    return out
