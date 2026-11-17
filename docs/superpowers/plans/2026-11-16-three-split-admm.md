# Three-Split ADMM Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved core three-split ADMM / split-Bregman-family solver for nonnegative LASSO/elastic-net with known background `b`, black-box CG inner solves, and certificate-gated convergence.

**Architecture:** Add `solvers/admm.py` beside `solvers/fista.py`; extract shared validation helpers into private `solvers/_common.py` without changing FISTA behavior; reuse `operators.py` and `certificates.py` unchanged. `x` carries the quadratic data/elastic-net terms, `z` carries nonnegativity + L1, and `u` is the scaled dual for constraint `x=z`.

**Tech Stack:** Python 3.12, NumPy runtime only, pytest tests, git local commits only.

**Spec:** `docs/superpowers/specs/2026-11-16-three-split-admm-design.md`

## Global Constraints

- Runtime dependency: NumPy only; no SciPy, no dense `AᵀA` assembly, no FFT operator in this plan.
- All solver state is one-dimensional `float64`; `lam` finite `>0`, `mu` finite `>=0`, `rho` finite `>0`.
- `solve_admm` output `result.x` is the nonnegative `z` split; certificates are computed on `z` via `compute_certificate(A, y, z, lam, mu=mu, b=b)`.
- Convergence requires BOTH normalized ADMM residuals (`<= residual_tol`) AND `kkt_residual <= kkt_tol` AND `relative_gap <= gap_tol`; residual shrinkage, CG convergence, or `max_iter` never claim convergence alone.
- `b` is known and constant; do NOT estimate background or implement a `B c` block in this plan.
- Do not implement FFT imaging operators, 285,696-voxel benchmarks, EF-Spin integration, external ground-truth solvers, working columns, support-restricted NNLS debiasing, or generalized mixed-regularization certificates.
- Commits are local only; do not push. Repository text must not contain machine absolute paths, personal emails, tokens, or keys.

---

### Task 1: Shared Validation Helper Extraction

**Files:**
- Create: `src/optimization_math/solvers/_common.py`
- Modify: `src/optimization_math/solvers/fista.py`
- Test: `tests/test_fista.py` (no assertion changes)

**Interfaces:**
- Consumes: existing `optimization_math.certificates._vector` and existing FISTA validation behavior.
- Produces: `_positive_int(value, name)`, `_positive_float(value, name)`, `_validate_common(op, y, lam, mu, b, x0)` for reuse by `admm.py`; FISTA keeps identical public behavior.

- [ ] **Step 1: Record the pre-refactor characterization**

Run: `python -m pytest -q`
Expected: `44 passed`

- [ ] **Step 2: Create the shared helper module**

Create `src/optimization_math/solvers/_common.py`:

```python
import numpy as np

from optimization_math.certificates import _vector


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value, name):
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive finite number") from None
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return out


def _validate_common(op, y, lam, mu, b, x0):
    y = _vector(y, "y")
    b_vec = np.zeros_like(y) if b is None else _vector(b, "b")
    if y.shape != (op.shape[0],) or b_vec.shape != y.shape:
        raise ValueError("y and b must match the operator output dimension")
    x = np.zeros(op.shape[1], dtype=np.float64) if x0 is None else _vector(x0, "x0").copy()
    if x.shape != (op.shape[1],):
        raise ValueError("x0 must match the operator input dimension")
    if np.any(x < 0):
        raise ValueError("x0 must be nonnegative")
    try:
        lam = float(lam)
        mu = float(mu)
    except (TypeError, ValueError):
        raise ValueError("lam must be positive and mu must be nonnegative") from None
    if not np.isfinite(lam) or not np.isfinite(mu) or lam <= 0 or mu < 0:
        raise ValueError("lam must be positive and mu must be nonnegative")
    return y, b_vec, x, lam, mu
```

- [ ] **Step 3: Update FISTA imports and remove duplicate helpers**

In `src/optimization_math/solvers/fista.py`, replace the local `_positive_int`, `_positive_float`, and `_validate_common` definitions with:

```python
from optimization_math.solvers._common import (
    _positive_float,
    _positive_int,
    _validate_common,
)
```

Keep `_nonnegative_elastic_net_prox`, `lambda_max`, `_estimate_lipschitz`, `_gradient_restart_needed`, `_backtracked_prox`, `_run_stage`, and `solve_fista` otherwise unchanged.

- [ ] **Step 4: Run tests to verify no behavior change**

Run: `python -m pytest -q`
Expected: `44 passed`

- [ ] **Step 5: Commit**

```bash
git add src/optimization_math/solvers/_common.py src/optimization_math/solvers/fista.py
git commit -m "refactor: share solver validation helpers"
```

---

### Task 2: ADMM Result Types, z-Step, and Black-Box CG

**Files:**
- Create: `src/optimization_math/solvers/admm.py`
- Test: `tests/test_admm.py`

**Interfaces:**
- Consumes: `as_linear_operator`, `Certificate`, `_vector` from existing modules; `_positive_int` is not needed yet.
- Produces: `AdmmIteration`, `AdmmResult`, `_nonnegative_l1_z_step(x, u, lam, rho)`, `_cg_ata(op, rhs, *, shift, x0, tol, max_iter)` returning `(x, iterations, relative_residual)`.

- [ ] **Step 1: Write failing helper tests**

Create `tests/test_admm.py`:

```python
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
```

Run: `python -m pytest tests/test_admm.py -q`
Expected: FAIL during collection with `ModuleNotFoundError: optimization_math.solvers.admm`

- [ ] **Step 2: Implement minimal ADMM helpers**

Create `src/optimization_math/solvers/admm.py`:

```python
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
```

Run: `python -m pytest tests/test_admm.py -q`
Expected: `3 passed`

Run: `python -m pytest -q`
Expected: `47 passed`

- [ ] **Step 3: Commit**

```bash
git add src/optimization_math/solvers/admm.py tests/test_admm.py
git commit -m "feat: add ADMM z-step and black-box CG helpers"
```

---

### Task 3: Fixed-ρ solve_admm Core With Certificate-Gated Stopping

**Files:**
- Modify: `src/optimization_math/solvers/admm.py`
- Test: `tests/test_admm.py`

**Interfaces:**
- Consumes: Task 1 `_validate_common`, `_positive_int`, `_positive_float`; Task 2 `_cg_ata`, `_nonnegative_l1_z_step`, result types.
- Produces public `solve_admm(...)` with fixed `rho` behavior; adaptive ρ remains Task 4 even if parameters are validated now.

- [ ] **Step 1: Write failing fixed-ρ solver tests**

Append to `tests/test_admm.py`:

```python
from optimization_math.certificates import compute_certificate
from optimization_math.solvers.admm import solve_admm
from optimization_math.solvers.fista import lambda_max


def test_admm_recovers_hand_derived_scalar_lasso_solution():
    result = solve_admm(
        np.array([[1.0]]), np.array([2.0]), 0.5,
        rho=1.0, cg_tol=1e-12, residual_tol=1e-6,
        certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8, max_iter=200,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [1.5], atol=1e-6)
    assert result.certificate.kkt_residual <= 1e-7
    assert result.certificate.relative_gap <= 1e-8


def test_admm_recovers_scalar_elastic_net_solution():
    result = solve_admm(
        np.array([[1.0]]), np.array([2.0]), 0.5, mu=1.0,
        rho=1.0, cg_tol=1e-12, residual_tol=1e-6,
        certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8, max_iter=200,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [0.75], atol=1e-6)


def test_admm_applies_known_background_before_fitting():
    result = solve_admm(
        np.array([[1.0]]), np.array([2.0]), 0.5, b=np.array([1.0]),
        rho=1.0, cg_tol=1e-12, residual_tol=1e-6,
        certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8, max_iter=200,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [0.5], atol=1e-6)


def test_admm_returns_zero_at_lambda_max_without_false_iteration_work():
    A = np.diag([1.0, 2.0])
    y = np.array([3.0, 1.0])
    result = solve_admm(A, y, lambda_max(A, y), rho=1.0)
    assert result.status == "converged"
    assert result.iterations == 0
    np.testing.assert_array_equal(result.x, np.zeros(2))


def test_admm_iteration_limit_does_not_claim_convergence():
    result = solve_admm(
        np.array([[1.0]]), np.array([2.0]), 0.1,
        rho=1.0, max_iter=1, certificate_stride=20,
        residual_tol=1e-12, kkt_tol=1e-12, gap_tol=1e-12,
    )
    assert result.status == "max_iter"
    assert result.iterations == 1
    assert np.isfinite(result.certificate.relative_gap)
    assert len(result.history) == 1


def test_admm_uses_black_box_operator_without_dense_matrix_access():
    result = solve_admm(
        DiagonalOperator(), np.array([3.0, 1.0]), 0.5,
        rho=1.0, cg_tol=1e-12, residual_tol=1e-6,
        certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8, max_iter=500,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [2.5, 0.375], atol=1e-6)


@pytest.mark.parametrize(
    "overrides",
    [{"lam": 0.0}, {"lam": 0.1, "mu": -1.0}, {"rho": 0.0},
     {"max_iter": 0}, {"cg_max_iter": 0}, {"certificate_stride": 0},
     {"residual_tol": 0.0}, {"kkt_tol": 0.0}, {"gap_tol": 0.0},
     {"y": np.array([1.0, 2.0])}, {"y": np.array([np.nan])}],
)
def test_admm_rejects_invalid_shapes_nonfinite_data_and_controls(overrides):
    kwargs = dict(A=np.array([[1.0]]), y=np.array([1.0]), lam=0.1)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        solve_admm(**kwargs)


def test_admm_rejects_negative_warm_start():
    with pytest.raises(ValueError, match="nonnegative"):
        solve_admm(np.array([[1.0]]), np.array([1.0]), 0.1, x0=np.array([-1.0]))


def test_admm_convergence_helper_requires_residuals_and_both_certificates():
    from optimization_math.solvers.admm import _admm_has_converged
    cert = compute_certificate(np.array([[1.0]]), np.array([2.0]), np.array([1.5]), 0.5)
    assert _admm_has_converged(0.0, 0.0, cert, 1e-6, 1e-6, 1e-6)
    assert not _admm_has_converged(1.0, 0.0, cert, 1e-6, 1e-6, 1e-6)
    assert not _admm_has_converged(0.0, 1.0, cert, 1e-6, 1e-6, 1e-6)
    assert not _admm_has_converged(0.0, 0.0, cert, 1e-6, 1e-6, -1.0)
    # A certificate at a non-optimal point has kkt_residual above tolerance even with
    # zero ADMM residuals; the KKT gate must still block convergence.
    bad_kkt = compute_certificate(np.array([[1.0]]), np.array([2.0]), np.array([0.01]), 0.5)
    assert bad_kkt.kkt_residual > 1e-6
    assert not _admm_has_converged(0.0, 0.0, bad_kkt, 1e-6, 1e-6, 1e-6)
    # A certificate with a material positive gap must also block convergence.
    bad_gap = compute_certificate(np.array([[1.0]]), np.array([2.0]), np.array([1.0]), 0.5)
    assert bad_gap.relative_gap > 1e-8
    assert not _admm_has_converged(0.0, 0.0, bad_gap, 1e-6, 1e-6, 1e-8)
```

Run: `python -m pytest tests/test_admm.py -q`
Expected: FAIL with `ImportError: cannot import name 'solve_admm'`

- [ ] **Step 2: Implement fixed-ρ solve_admm**

Append to `src/optimization_math/solvers/admm.py`:

```python
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


def solve_admm(A, y, lam, *, mu=0.0, b=None, x0=None, rho=1.0,
               adaptive_rho=False, rho_min=1e-3, rho_max=1e3,
               rho_balance=10.0, rho_scale=2.0, max_iter=200,
               cg_max_iter=None, cg_tol=1e-8, residual_tol=1e-4,
               kkt_tol=1e-5, gap_tol=1e-4, certificate_stride=10):
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
    if adaptive_rho:
        raise NotImplementedError("adaptive rho is implemented in Task 4")
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
```

Run: `python -m pytest tests/test_admm.py -q`
Expected: all `test_admm.py` tests pass

Run: `python -m pytest -q`
Expected: `66 passed`

- [ ] **Step 3: Commit**

```bash
git add src/optimization_math/solvers/admm.py tests/test_admm.py
git commit -m "feat: implement fixed-rho three-split ADMM"
```

---

### Task 4: Bounded Adaptive ρ With Unscaled-Dual Continuity

**Files:**
- Modify: `src/optimization_math/solvers/admm.py`
- Test: `tests/test_admm.py`

**Interfaces:**
- Consumes: fixed-ρ `solve_admm` from Task 3.
- Produces: `_balance_rho(rho, u, primal_residual, dual_residual, *, rho_min, rho_max, rho_balance, rho_scale)` returning `(new_rho, new_u)`; `solve_admm(..., adaptive_rho=True)` works.

- [ ] **Step 1: Write failing adaptive-ρ tests**

Append to `tests/test_admm.py`:

```python
def test_balance_rho_preserves_unscaled_dual_when_not_clamped():
    from optimization_math.solvers.admm import _balance_rho
    u = np.array([0.5, -2.0])
    # primal (20.0) strictly exceeds rho_balance * dual (10.0 * 1.0), so rho doubles.
    new_rho, new_u = _balance_rho(
        1.0, u, 20.0, 1.0,
        rho_min=1e-3, rho_max=1e3, rho_balance=10.0, rho_scale=2.0,
    )
    assert new_rho == 2.0
    np.testing.assert_allclose(new_rho * new_u, 1.0 * u)


def test_balance_rho_respects_bounds():
    from optimization_math.solvers.admm import _balance_rho
    u = np.ones(2)
    high_rho, high_u = _balance_rho(
        1e3, u, 100.0, 1.0,
        rho_min=1e-3, rho_max=1e3, rho_balance=10.0, rho_scale=2.0,
    )
    low_rho, low_u = _balance_rho(
        1e-3, u, 1.0, 100.0,
        rho_min=1e-3, rho_max=1e3, rho_balance=10.0, rho_scale=2.0,
    )
    assert high_rho == 1e3
    assert low_rho == 1e-3
    np.testing.assert_allclose(high_rho * high_u, 1e3 * u)
    np.testing.assert_allclose(low_rho * low_u, 1e-3 * u)


def test_adaptive_rho_stays_bounded_and_converges():
    result = solve_admm(
        np.array([[1.0]]), np.array([2.0]), 0.5,
        rho=0.25, adaptive_rho=True, rho_min=0.1, rho_max=4.0,
        rho_balance=2.0, rho_scale=2.0, cg_tol=1e-12,
        residual_tol=1e-6, certificate_stride=1,
        kkt_tol=1e-7, gap_tol=1e-8, max_iter=300,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [1.5], atol=1e-6)
    assert all(0.1 <= item.rho <= 4.0 for item in result.history)
    assert 0.1 <= result.rho <= 4.0


def test_adaptive_rho_converges_on_scalar_lasso():
    # Fail-first anchor: Task 3 always raises NotImplementedError when adaptive_rho=True,
    # so this test fails (propagates the error) until Task 4 removes the guard and
    # adaptive rho converges to the scalar optimum.
    result = solve_admm(np.array([[1.0]]), np.array([2.0]), 0.5, adaptive_rho=True)
    assert result.status == "converged"


@pytest.mark.parametrize(
    "overrides",
    [{"adaptive_rho": "yes"}, {"rho_min": 0.0}, {"rho_max": 0.0},
     {"rho_min": 2.0, "rho_max": 1.0}, {"rho_balance": 0.5},
     {"rho_scale": 1.0}],
)
def test_admm_rejects_invalid_adaptive_rho_controls(overrides):
    kwargs = dict(A=np.array([[1.0]]), y=np.array([1.0]), lam=0.1)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        solve_admm(**kwargs)
```

Run: `python -m pytest tests/test_admm.py -q`
Expected: FAIL — `test_adaptive_rho_converges_on_scalar_lasso` propagates `NotImplementedError`, and the two `_balance_rho` unit tests fail collection/import (`cannot import name '_balance_rho'`). The parametrized `test_admm_rejects_invalid_adaptive_rho_controls` already passes at end of Task 3 (its invalid values are rejected by Task 3's validation, before the adaptive branch is reached); it is a regression test, not a fail-first anchor.

- [ ] **Step 2: Implement bounded adaptive ρ**

In `src/optimization_math/solvers/admm.py`, remove the `if adaptive_rho: raise NotImplementedError(...)` block and add:

```python
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
```

Inside the `solve_admm` loop, the `_balance_rho` update must run **after** the convergence check on the current iteration's residuals and **before** moving to the next iteration. This ordering is essential: the convergence return at Task 3 uses the `rho` that produced the current residuals (matching `history[-1].rho`), while the balance update only changes `rho`/`u` for the *next* iteration. The full Task 3 code block above already shows this ordering (balance call placed after the converged return and gated by `iteration < max_iter`); no further edit is needed beyond the Task 3 block already reflecting it.

Keep the convergence check using the `primal`, `dual`, and `rho` values that produced the current iteration; the updated `rho/u` apply to the next iteration only.

Run: `python -m pytest tests/test_admm.py -q`
Expected: all `test_admm.py` tests pass

Run: `python -m pytest -q`
Expected: `76 passed`

- [ ] **Step 3: Commit**

```bash
git add src/optimization_math/solvers/admm.py tests/test_admm.py
git commit -m "feat: add bounded adaptive rho to ADMM"
```

---

### Task 5: Cross-Check, Documentation, and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/survey/2026-11-13-L1-L2-稀疏求解算法调研.md`
- Modify: `docs/superpowers/specs/2026-11-16-three-split-admm-design.md`
- Modify: `docs/superpowers/plans/2026-11-16-three-split-admm.md`
- Test: `tests/test_admm.py`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: public usage documentation, corrected roadmap status, and final verification evidence.

- [ ] **Step 1: Write the FISTA cross-check test**

Append to `tests/test_admm.py`:

```python
def test_admm_matches_fista_objective_on_deterministic_small_problem():
    from optimization_math.solvers.fista import solve_fista
    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    y = np.array([2.0, 1.0, 1.5])
    lam, mu = 0.2, 0.1
    admm = solve_admm(
        A, y, lam, mu=mu, rho=1.0, cg_tol=1e-12, residual_tol=1e-7,
        certificate_stride=1, kkt_tol=1e-8, gap_tol=1e-9, max_iter=1000,
    )
    fista = solve_fista(
        A, y, lam, mu=mu, certificate_stride=1,
        kkt_tol=1e-8, gap_tol=1e-9, max_iter=5000,
    )
    assert admm.status == fista.status == "converged"
    assert admm.certificate.relative_gap <= 1e-9
    assert fista.certificate.relative_gap <= 1e-9
    assert abs(admm.certificate.primal - fista.certificate.primal) <= 1e-8 * max(
        1.0, abs(fista.certificate.primal)
    )
```

Run: `python -m pytest tests/test_admm.py::test_admm_matches_fista_objective_on_deterministic_small_problem -q`
Expected: PASS

- [ ] **Step 2: Update README with verified ADMM usage**

Add a short ADMM subsection after the FISTA quick start. The example must be run for real before documentation is committed:

```python
import numpy as np
from optimization_math.solvers.admm import solve_admm

A = np.array([[1.0, 1.0], [0.0, 1.0]])
y = np.array([2.0, 1.0])
result = solve_admm(A, y, 0.1, rho=1.0, certificate_stride=10)
assert result.status == "converged"
assert result.certificate.kkt_residual <= 1e-5
assert result.certificate.relative_gap <= 1e-4
print(result.status)
print(result.x)
```

Keep the existing platform-variation caveat pattern: numerical digits may vary slightly across NumPy/BLAS platforms; tolerance assertions are the acceptance criterion. Update the roadmap row so ADMM core is marked as implemented for known `b`, while background-block joint estimation remains pending.

- [ ] **Step 3: Update survey and spec status to match implementation**

In `docs/survey/2026-11-13-L1-L2-稀疏求解算法调研.md`, adjust the P0 ADMM wording so it does not claim background-block joint estimation is already delivered; state that the merged core assumes known `b` and that `B c` joint estimation is a separate follow-up. In `docs/superpowers/specs/2026-11-16-three-split-admm-design.md`, change the status line to record user approval and implementation completion date. Update this plan's checkboxes only for steps whose commands actually ran.

- [ ] **Step 4: Run final verification**

Run: `python -m pytest -q`
Expected: `77 passed`

Run the README ADMM example exactly as written and confirm it prints `converged` with both certificate tolerances satisfied.

Run: `git diff --check`
Expected: no output

Run: `git status --short --branch`
Expected: only intended files modified before commit; after commit, clean.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/survey/2026-11-13-L1-L2-稀疏求解算法调研.md docs/superpowers/specs/2026-11-16-three-split-admm-design.md docs/superpowers/plans/2026-11-16-three-split-admm.md tests/test_admm.py
git commit -m "docs: document verified core ADMM usage"
```

---

## Final Explicit Non-Claims

- No background block `B c` or unknown-`b` estimation is implemented.
- No FFT imaging operator, 285,696-voxel benchmark, EF-Spin integration, or external solver ground truth is validated.
- No generalized mixed-regularization certificate is added; ADMM reuses the existing nonnegative LASSO/elastic-net certificate on `z`.
- Nothing is pushed to GitHub by this plan.
