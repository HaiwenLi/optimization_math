"""GAP-safe screening for sparse-regularized least squares.

Follows Fercoq, Gramfort & Salmon (2015) ``Mind the duality gap'' and
Ndiaye et al. (2017) ``Gap safe screening rules for sparsity enforcing
penalties''. A coordinate ``i`` is *safely* removed (its optimum is
guaranteed to be zero) when it satisfies

    |a_i^T nu| + ||a_i|| * r_ball < lam,

where ``nu`` is a dual-feasible point built from the residual, ``r_ball`` is
the radius of a sphere centered at ``nu`` that provably contains the optimum
dual solution, and ``a_i`` is the i-th column of ``A``.

The radius is derived from the primal-dual gap: for the dual-feasible scaled
residual ``nu = theta * r`` (with ``theta = min(1, lam / ||q||_inf)``), the dual
``D(nu) = -0.5 ||nu||^2 + <b - y, nu>`` is 1-strongly concave (Hessian = -I), so
the gap ``P(x) - D(nu)`` upper-bounds ``0.5 ||nu* - nu||^2`` and

    r_ball = sqrt(2 * gap).

Note there is NO theta factor: ``nu = theta * r`` is itself the dual-feasible
point, and the strong-concavity bound holds for any dual-feasible ``nu``.

Because the screened coordinate is *provably* zero, screening never changes
the optimal solution -- it only accelerates the solver by shrinking the
effective dimension.

This module is solver-agnostic: it exposes only the screening *decision*
(the safe (inactive) support) and leaves the actual solve to any solver. It
builds the same dual-feasible point the signed-LASSO certificate uses, so it
shares exactly the same optimality semantics as the rest of the framework.
"""
from __future__ import annotations

import numpy as np

from optimization_math.certificates_signed import compute_certificate_signed
from optimization_math.operators import as_linear_operator
from optimization_math.solvers._common import _positive_float


def column_norms_squared(op):
    """Squared ``||a_i||_2^2`` of every column of the operator ``A``.

    Computed exactly (and operator-agnostically) via ``A^T A e_i``, i.e. one
    ``rmatvec(matvec(.))`` per column. Returns a 1D array of length ``n``.
    """
    op = as_linear_operator(op)
    n = op.shape[1]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        e = np.zeros(n, dtype=np.float64)
        e[i] = 1.0
        out[i] = float(op.rmatvec(op.matvec(e))[i])
    return out


def compute_safe_screen(A, y, x, lam, *, mu=0.0, b=None, gap=None, theta=None):
    """Return the boolean mask of coordinates that are *safely* zero.

    Parameters
    ----------
    A : dense matrix or LinearOperator
        Design / forward operator (m x n).
    y : array
        Observations (length m).
    x : array
        Current primal iterate (length n); only used to build the dual point
        via the signed certificate.
    lam : float
        L1 penalty weight (must be > 0).
    mu : float
        Elastic-net quadratic weight. Must be 0.0: the GAP-safe radius is
        derived for plain LASSO (mu=0) only; mu>0 raises NotImplementedError.
    b : array or None
        Offset term (default zeros).
    gap : float or None
        Precomputed primal-dual gap ``P(x) - D(nu)`` for the same ``x`` and
        ``theta``. The caller must supply a valid upper bound, not a normalized
        relative gap. If None it is computed using the actual dual point.
    theta : float or None
        Dual scaling in (0, 1] satisfying ``||A.T @ (theta*r)||_inf <= lam``.
        If None, use the same scaling as the signed certificate.

    Returns
    -------
    numpy.ndarray of bool, length ``n``: True where the coordinate is safely
    provable to be zero at the LASSO optimum.
    """
    op = as_linear_operator(A)
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    b_vec = np.zeros_like(y) if b is None else np.asarray(b, dtype=np.float64)
    if y.ndim != 1 or x.ndim != 1 or b_vec.ndim != 1:
        raise ValueError("y, x, b must be 1D")
    if y.shape != (op.shape[0],) or x.shape != (op.shape[1],) or b_vec.shape != y.shape:
        raise ValueError("A, y, x, b shapes do not match")
    lam = _positive_float(lam, "lam")
    try:
        mu = float(mu)
    except (TypeError, ValueError):
        raise ValueError("mu must be a nonnegative number") from None
    if not np.isfinite(mu) or mu < 0:
        raise ValueError("mu must be nonnegative")
    if mu != 0.0:
        # The GAP-safe radius sqrt(2*gap) is derived for the plain LASSO dual
        # (D is 1-strongly concave, D'' = -I). The elastic-net quadratic
        # distorts the dual so that bound no longer holds; safe screening for
        # elastic net requires a separate, mu-corrected rule that is out of
        # scope here. Reject rather than silently over-claim safety.
        raise NotImplementedError(
            "GAP-safe screening is currently derived for mu=0 (plain LASSO) only"
        )
    if not all(np.all(np.isfinite(v)) for v in (y, x, b_vec)):
        raise ValueError("y, x and b must be finite")

    r = op.matvec(x) + b_vec - y
    q_inf = float(np.linalg.norm(op.rmatvec(r), ord=np.inf))
    supplied_theta = theta is not None
    if theta is None:
        theta = 1.0 if q_inf <= lam else lam / q_inf
    theta = _positive_float(theta, "theta")
    if theta > 1.0 or theta * q_inf > lam * (1.0 + 1e-12):
        raise ValueError("theta must be <= 1 and produce a dual-feasible point")
    nu = theta * r

    if gap is None:
        cert = compute_certificate_signed(op, y, x, lam, mu=mu, b=b_vec)
        if supplied_theta:
            # The certificate's dual uses its OWN theta, not the supplied one.
            dual = -0.5 * np.dot(nu, nu) + np.dot(b_vec - y, nu)
            gap = max(cert.primal - dual, 0.0)
        else:
            gap = max(cert.primal - cert.dual, 0.0)

    try:
        gap = float(gap)
    except (TypeError, ValueError):
        raise ValueError("gap must be a finite nonnegative number") from None
    if not np.isfinite(gap) or gap < 0.0:
        raise ValueError("gap must be a finite nonnegative number")

    # nu = theta * r (dual feasible), r_ball = sqrt(2 * gap).
    r_ball = float(np.sqrt(max(2.0 * gap, 0.0)))

    # |a_i^T nu| + ||a_i|| * r_ball < lam  <=>  coordinate i is safely zero.
    aTnu = op.rmatvec(nu)
    col_norms = np.sqrt(column_norms_squared(op))
    margin = np.abs(aTnu) + col_norms * r_ball
    return margin < lam
