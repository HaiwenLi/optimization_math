# Nonnegative FISTA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested NumPy nonnegative FISTA solver for LASSO and elastic-net problems with adaptive restart, backtracking, lambda continuation, and algorithm-independent optimality certificates.

**Architecture:** A checked `matvec/rmatvec` operator adapter isolates solvers from matrix storage. Pure certificate functions construct a feasible augmented dual point, while the FISTA module owns iteration, continuation, and result history. Each behavior is introduced by a failing test and committed only after its focused test and the existing suite pass.

**Tech Stack:** Python 3.12, NumPy 2.x, pytest 9, setuptools src layout.

**Spec:** `docs/superpowers/specs/2026-11-15-nonnegative-fista-design.md`

## Global Constraints

- Runtime dependency is NumPy only; pytest is a test dependency; do not add SciPy.
- Solve `min_{x>=0} 0.5*||Ax+b-y||² + lam*||x||₁ + 0.5*mu*||x||²` with `lam > 0`, `mu >= 0`.
- Solver states and outputs are one-dimensional `float64` arrays.
- Convergence requires both `kkt_residual <= kkt_tol` and `relative_gap <= gap_tol`.
- The elastic-net augmented dual uses `nu=theta*r`, `eta=theta*sqrt(mu)*x`, and `q=A.T@r+mu*x`.
- Certificate evaluation at an accepted FISTA iterate costs an additional `A/Aᵀ` pair; do not describe it as free.
- Do not implement ADMM, split Bregman, working columns, FFT imaging operators, or external ground-truth solvers in this plan.
- Use test-first development: observe each new behavior fail for the expected reason before implementing it.
- Keep all repository paths relative and do not add personal information or secrets.
- Make local commits only; do not push without separate user approval.

---

### Task 1: Package Skeleton and Checked Linear Operators

**Files:**
- Create: `pyproject.toml`
- Create: `src/optimization_math/__init__.py`
- Create: `src/optimization_math/operators.py`
- Test: `tests/test_operators.py`

**Interfaces:**
- Consumes: Existing NumPy/pytest environment.
- Produces: `LinearOperator`, `as_linear_operator(A)`, checked `shape`, `matvec`, and `rmatvec` used by every later task.

- [ ] **Step 1: Add package configuration and a failing dense-operator test**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "optimization-math"
version = "0.1.0"
description = "Sparse inverse problems with pluggable solvers and optimality certificates"
readme = "README.md"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
test = ["pytest>=8"]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

Create an empty `src/optimization_math/__init__.py`, then create `tests/test_operators.py`:

```python
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
```

- [ ] **Step 2: Install the local package and verify RED**

Run:

```powershell
python -m pip install -e .
python -m pytest tests/test_operators.py -v
```

Expected: collection fails because `optimization_math.operators` does not exist. Do not proceed until the failure is caused by the missing operator module.

- [ ] **Step 3: Implement the minimal checked operator adapter**

Create `src/optimization_math/operators.py`:

```python
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
```

- [ ] **Step 4: Verify GREEN and commit**

Run `python -m pytest tests/test_operators.py -v`; expect both tests to pass.

```powershell
git add pyproject.toml src tests/test_operators.py
git commit -m "feat: add checked linear operator interface"
```

---

### Task 2: Proximal Map, Lambda Threshold, and Dual Certificates

**Files:**
- Create: `src/optimization_math/certificates.py`
- Create: `src/optimization_math/solvers/__init__.py`
- Create: `src/optimization_math/solvers/fista.py`
- Test: `tests/test_certificates.py`
- Test: `tests/test_fista.py`

**Interfaces:**
- Consumes: `as_linear_operator` from Task 1.
- Produces: `_nonnegative_elastic_net_prox`, `lambda_max`, frozen `Certificate`, and `compute_certificate` consumed by the solver.

- [ ] **Step 1: Write hand-derived failing math tests**

Create `tests/test_certificates.py`:

```python
import numpy as np
import pytest

from optimization_math.certificates import compute_certificate


def test_scalar_lasso_optimum_has_zero_kkt_and_dual_gap():
    cert = compute_certificate(
        np.array([[1.0]]), np.array([2.0]), np.array([1.5]), 0.5
    )
    assert cert.kkt_residual == pytest.approx(0.0, abs=1e-12)
    assert cert.primal == pytest.approx(0.875)
    assert cert.dual == pytest.approx(0.875)
    assert cert.relative_gap == pytest.approx(0.0, abs=1e-12)


def test_scalar_elastic_net_optimum_has_strong_duality():
    # x=0.75 solves x-2+x+0.5=0; p=d=1.4375.
    cert = compute_certificate(
        np.array([[1.0]]), np.array([2.0]), np.array([0.75]), 0.5, mu=1.0
    )
    assert cert.kkt_residual == pytest.approx(0.0, abs=1e-12)
    assert cert.primal == pytest.approx(1.4375)
    assert cert.dual == pytest.approx(1.4375)
    assert cert.relative_gap == pytest.approx(0.0, abs=1e-12)


def test_certificate_rejects_negative_primal_iterate():
    with pytest.raises(ValueError, match="nonnegative"):
        compute_certificate(
            np.array([[1.0]]), np.array([1.0]), np.array([-0.1]), 0.5
        )


def test_nontrivial_iterate_has_feasible_dual_lower_bound():
    cert = compute_certificate(
        np.array([[1.0, 0.5], [0.2, -1.0]]),
        np.array([1.0, -0.5]),
        np.array([0.2, 0.0]),
        0.4,
        mu=0.3,
    )
    assert cert.primal >= cert.dual
    assert cert.relative_gap >= 0
```

Create `tests/test_fista.py`:

```python
import numpy as np
import pytest

from optimization_math.solvers.fista import (
    _nonnegative_elastic_net_prox,
    lambda_max,
)


def test_nonnegative_elastic_net_prox_uses_soft_threshold_and_l2_denominator():
    z = np.array([-2.0, 0.5, 2.0])
    np.testing.assert_allclose(
        _nonnegative_elastic_net_prox(z, step=0.5, lam=0.5, mu=0.0),
        [0.0, 0.0, 1.75],
    )
    np.testing.assert_allclose(
        _nonnegative_elastic_net_prox(z, step=0.5, lam=0.5, mu=2.0),
        [0.0, 0.125, 0.875],
    )


def test_lambda_max_is_threshold_for_zero_stationarity():
    A = np.diag([1.0, 2.0])
    y = np.array([3.0, 1.0])
    assert lambda_max(A, y) == pytest.approx(3.0)
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/test_certificates.py tests/test_fista.py -v`.
Expected: collection fails for the missing certificate and FISTA modules.

- [ ] **Step 3: Implement certificates with the corrected augmented dual**

Create `src/optimization_math/certificates.py`:

```python
from dataclasses import dataclass

import numpy as np

from optimization_math.operators import as_linear_operator


@dataclass(frozen=True)
class Certificate:
    stage: int
    iteration: int
    lam: float
    primal: float
    dual: float
    kkt_residual: float
    relative_gap: float


def _vector(value, name):
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 1 or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return out


def compute_certificate(A, y, x, lam, *, mu=0.0, b=None, stage=0, iteration=0):
    op = as_linear_operator(A)
    y = _vector(y, "y")
    x = _vector(x, "x")
    b_vec = np.zeros_like(y) if b is None else _vector(b, "b")
    if y.shape != (op.shape[0],) or b_vec.shape != y.shape or x.shape != (op.shape[1],):
        raise ValueError("Certificate vector shapes do not match the operator")
    if not np.isfinite(lam) or not np.isfinite(mu) or lam <= 0 or mu < 0:
        raise ValueError("lam must be positive and mu must be nonnegative")
    if np.any(x < 0):
        raise ValueError("x must be nonnegative")

    r = op.matvec(x) + b_vec - y
    q = op.rmatvec(r) + mu * x
    primal = 0.5 * np.dot(r, r) + lam * np.sum(x) + 0.5 * mu * np.dot(x, x)

    positive = x > 0
    active_violation = np.max(np.abs(q[positive] + lam), initial=0.0)
    inactive_violation = np.max(np.maximum(-lam - q[~positive], 0.0), initial=0.0)
    kkt = max(active_violation, inactive_violation) / max(lam, np.linalg.norm(q, ord=np.inf))

    negative = q < 0
    theta = min(1.0, float(np.min(lam / -q[negative], initial=1.0))) if np.any(negative) else 1.0
    dual = (
        -0.5 * theta**2 * (np.dot(r, r) + mu * np.dot(x, x))
        + theta * np.dot(b_vec - y, r)
    )
    gap = (primal - dual) / max(1.0, abs(primal), abs(dual))
    if -1e-12 <= gap < 0:
        gap = 0.0
    elif gap < 0:
        raise FloatingPointError("Material negative dual gap indicates a certificate error")

    return Certificate(stage, iteration, float(lam), float(primal), float(dual), float(kkt), float(gap))
```

- [ ] **Step 4: Implement prox and lambda_max, then verify GREEN**

Create an empty `src/optimization_math/solvers/__init__.py` and add to `src/optimization_math/solvers/fista.py`:

```python
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
```

Run `python -m pytest tests/test_certificates.py tests/test_fista.py -v`; expect all six tests to pass. The elastic-net scalar test must pass with `q=A.T@r+mu*x`; changing it to a minus sign must fail.

- [ ] **Step 5: Run the suite and commit**

Run `python -m pytest -v`; expect all tests to pass.

```powershell
git add src/optimization_math/certificates.py src/optimization_math/solvers tests
git commit -m "feat: add nonnegative optimality certificates"
```

---

### Task 3: Deterministic Base FISTA Solver

**Files:**
- Modify: `src/optimization_math/solvers/fista.py`
- Test: `tests/test_fista.py`

**Interfaces:**
- Consumes: Task 1 operator and Task 2 prox/certificate functions.
- Produces: `FistaResult` and `solve_fista` with base momentum, power iteration, elastic-net support, final certificates, and honest `max_iter` status.

- [ ] **Step 1: Add failing scalar, zero-solution, and stopping tests**

Append to `tests/test_fista.py`:

```python
from optimization_math.solvers.fista import solve_fista


def test_fista_recovers_hand_derived_scalar_lasso_solution():
    result = solve_fista(
        np.array([[1.0]]), np.array([2.0]), 0.5,
        certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [1.5], atol=1e-6)
    assert result.certificate.kkt_residual <= 1e-7
    assert result.certificate.relative_gap <= 1e-8


def test_fista_recovers_scalar_elastic_net_solution():
    result = solve_fista(
        np.array([[1.0]]), np.array([2.0]), 0.5, mu=1.0,
        certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [0.75], atol=1e-6)


def test_fista_returns_zero_at_lambda_max_without_false_iteration_work():
    result = solve_fista(np.diag([1.0, 2.0]), np.array([3.0, 1.0]), 3.0)
    assert result.status == "converged"
    assert result.iterations == 0
    np.testing.assert_array_equal(result.x, np.zeros(2))


def test_fista_iteration_limit_does_not_claim_convergence():
    result = solve_fista(
        np.array([[1.0]]), np.array([2.0]), 0.1,
        max_iter=1, certificate_stride=20, kkt_tol=1e-12, gap_tol=1e-12,
    )
    assert result.status == "max_iter"
    assert result.iterations == 1
    assert np.isfinite(result.certificate.relative_gap)


@pytest.mark.parametrize("kwargs", [{"lam": 0.0}, {"lam": 0.1, "mu": -1.0}])
def test_fista_rejects_invalid_regularization(kwargs):
    with pytest.raises(ValueError):
        solve_fista(np.array([[1.0]]), np.array([1.0]), **kwargs)


def test_fista_rejects_negative_warm_start():
    with pytest.raises(ValueError, match="nonnegative"):
        solve_fista(
            np.array([[1.0]]), np.array([1.0]), 0.1, x0=np.array([-1.0])
        )


def test_fista_uses_black_box_operator_without_dense_matrix_access():
    class DiagonalOperator:
        shape = (2, 2)

        def matvec(self, x):
            return np.array([x[0], 2.0 * x[1]])

        def rmatvec(self, y):
            return np.array([y[0], 2.0 * y[1]])

    result = solve_fista(
        DiagonalOperator(), np.array([3.0, 1.0]), 0.5,
        lipschitz=4.0, certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [2.5, 0.25], atol=1e-6)


def test_fista_applies_known_background_before_fitting():
    result = solve_fista(
        np.array([[1.0]]), np.array([2.0]), 0.5, b=np.array([1.0]),
        certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8,
    )
    np.testing.assert_allclose(result.x, [0.5], atol=1e-6)


@pytest.mark.parametrize(
    "overrides",
    [{"y": np.array([1.0, 2.0])}, {"y": np.array([np.nan])},
     {"max_iter": 0}, {"certificate_stride": 0}, {"lipschitz": 0.0}],
)
def test_fista_rejects_invalid_shapes_nonfinite_data_and_controls(overrides):
    kwargs = dict(A=np.array([[1.0]]), y=np.array([1.0]), lam=0.1)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        solve_fista(**kwargs)
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/test_fista.py -k "fista" -v`.
Expected: collection fails because `solve_fista` and `FistaResult` are not implemented.

- [ ] **Step 3: Implement result types, validation, and power iteration**

Add to `fista.py`:

```python
from dataclasses import dataclass
from typing import Literal

from optimization_math.certificates import Certificate, compute_certificate


@dataclass(frozen=True)
class FistaResult:
    x: np.ndarray
    status: Literal["converged", "max_iter"]
    iterations: int
    total_iterations: int
    lipschitz: float
    restart_count: int
    certificate: Certificate
    history: tuple[Certificate, ...]


def _estimate_lipschitz(op, *, power_iter, power_tol):
    rng = np.random.default_rng(0)
    v = rng.standard_normal(op.shape[1])
    v /= np.linalg.norm(v)
    estimate = 0.0
    for _ in range(power_iter):
        w = op.rmatvec(op.matvec(v))
        norm = np.linalg.norm(w)
        if norm == 0:
            return 1.0
        w /= norm
        new_estimate = float(np.dot(w, op.rmatvec(op.matvec(w))))
        if estimate > 0 and abs(new_estimate - estimate) <= power_tol * max(1.0, abs(new_estimate)):
            estimate = new_estimate
            break
        estimate = new_estimate
        v = w
    return max(estimate, np.finfo(np.float64).tiny)


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
    if not np.isfinite(lam) or not np.isfinite(mu) or lam <= 0 or mu < 0:
        raise ValueError("lam must be positive and mu must be nonnegative")
    return y, b_vec, x
```

Validate positive integer `max_iter`, `certificate_stride`, and `power_iter`; positive finite tolerances and user-supplied `lipschitz`; raise `ValueError` on violation.

- [ ] **Step 4: Implement the base iteration and certificate stopping**

Implement the initial public signature without continuation/restart options yet:

```python
def solve_fista(A, y, lam, *, mu=0.0, b=None, x0=None, lipschitz=None,
                max_iter=1000, kkt_tol=1e-5, gap_tol=1e-4,
                certificate_stride=10, power_iter=50, power_tol=1e-6):
```

Use this exact iteration skeleton:

```python
op = as_linear_operator(A)
y, b_vec, x = _validate_common(op, y, lam, mu, b, x0)
L = float(lipschitz) if lipschitz is not None else _estimate_lipschitz(
    op, power_iter=power_iter, power_tol=power_tol
)
x_previous = x.copy()
z = x.copy()
momentum = 1.0
history = [compute_certificate(op, y, x, lam, mu=mu, b=b_vec, stage=0, iteration=0)]
status = "max_iter"
iterations = 0

if history[-1].kkt_residual <= kkt_tol and history[-1].relative_gap <= gap_tol:
    status = "converged"
else:
    for k in range(1, max_iter + 1):
        r_z = op.matvec(z) + b_vec - y
        grad = op.rmatvec(r_z)
        x_new = _nonnegative_elastic_net_prox(z - grad / L, step=1.0 / L, lam=lam, mu=mu)
        if not np.all(np.isfinite(x_new)):
            raise FloatingPointError("FISTA produced a non-finite iterate")

        if k % certificate_stride == 0 or k == max_iter:
            cert = compute_certificate(op, y, x_new, lam, mu=mu, b=b_vec, stage=0, iteration=k)
            history.append(cert)
            if cert.kkt_residual <= kkt_tol and cert.relative_gap <= gap_tol:
                x = x_new
                iterations = k
                status = "converged"
                break

        next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        z = x_new + ((momentum - 1.0) / next_momentum) * (x_new - x)
        x_previous, x = x, x_new
        momentum = next_momentum
        iterations = k

return FistaResult(x, status, iterations, iterations, L, 0, history[-1], tuple(history))
```

The executor may remove the unused `x_previous` variable during the green refactor without changing behavior. Do not add restart or continuation in this task.

- [ ] **Step 5: Verify GREEN, run all tests, and commit**

Run `python -m pytest tests/test_fista.py -v`, then `python -m pytest -v`.
Expected: all tests pass, including exact zero-iteration convergence at `lambda_max`.

```powershell
git add src/optimization_math/solvers/fista.py tests/test_fista.py
git commit -m "feat: implement base nonnegative FISTA solver"
```

---

### Task 4: Backtracking, Adaptive Restart, and Lambda Continuation

**Files:**
- Modify: `src/optimization_math/solvers/fista.py`
- Test: `tests/test_fista.py`

**Interfaces:**
- Consumes: Base solver from Task 3.
- Produces: Final `solve_fista` signature with `restart`, `backtracking`, `lambda_path`, and `continuation_tol_scale`; stage-local histories and aggregate restart/iteration counts.

- [ ] **Step 1: Add failing backtracking and restart tests**

Append:

```python
def test_backtracking_recovers_from_underestimated_lipschitz():
    result = solve_fista(
        np.eye(2), np.array([2.0, -1.0]), 0.5,
        lipschitz=1e-8, backtracking=True,
        certificate_stride=1, kkt_tol=1e-7, gap_tol=1e-8,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [1.5, 0.0], atol=1e-6)
    assert result.lipschitz >= 0.5


def test_gradient_restart_predicate_matches_sign_rule():
    from optimization_math.solvers.fista import _gradient_restart_needed
    assert _gradient_restart_needed(np.array([1.0]), np.array([1.0]), np.array([0.0]))
    assert not _gradient_restart_needed(np.array([1.0]), np.array([-1.0]), np.array([0.0]))


def test_restarted_fista_converges_on_correlated_operator():
    A = np.array([[1.0, 1.0], [1.0, 1.01]])
    result = solve_fista(
        A, np.array([1.0, 2.0]), 0.1, restart=True,
        certificate_stride=1, kkt_tol=1e-6, gap_tol=1e-6, max_iter=3000,
    )
    assert result.status == "converged"
    assert np.all(result.x >= 0)
```

- [ ] **Step 2: Verify the new options are missing, then implement them**

Run `python -m pytest tests/test_fista.py -k "backtracking or restart" -v`.
Expected: failures for unexpected keyword arguments and the missing restart predicate.

Extend the signature with `restart=True`, `backtracking=True`. Add:

```python
def _gradient_restart_needed(grad, x_new, x_anchor):
    return bool(np.dot(grad, x_new - x_anchor) > 0.0)
```

Implement a helper that evaluates the proximal-gradient descent condition and doubles `L` until the condition holds:

```python
def _backtracked_prox(op, y, b_vec, z, grad, L, lam, mu, enabled):
    f_z = 0.5 * np.dot(op.matvec(z) + b_vec - y, op.matvec(z) + b_vec - y)
    for _ in range(60):
        x_new = _nonnegative_elastic_net_prox(z - grad / L, step=1.0 / L, lam=lam, mu=mu)
        if not enabled:
            return x_new, L
        delta = x_new - z
        r_new = op.matvec(x_new) + b_vec - y
        f_new = 0.5 * np.dot(r_new, r_new)
        upper = f_z + np.dot(grad, delta) + 0.5 * L * np.dot(delta, delta)
        tolerance = 1e-12 * max(1.0, abs(f_z), abs(upper))
        if f_new <= upper + tolerance:
            return x_new, L
        L *= 2.0
    raise FloatingPointError("Backtracking failed to find a finite stable step")
```

In the executor, compute `r_z` once and pass its squared norm into this helper instead of evaluating `matvec(z)` twice; preserve the formula. If `_gradient_restart_needed(grad, x_new, x)` is true, recompute from `z=x`, reset `momentum=1.0`, count the restart, and use the non-extrapolated accepted step. Keep the increased `L` for later iterations.

- [ ] **Step 3: Run focused tests and commit the robustness increment**

Run `python -m pytest tests/test_fista.py -k "backtracking or restart" -v`, then `python -m pytest -v`.
Expected: all tests pass.

```powershell
git add src/optimization_math/solvers/fista.py tests/test_fista.py
git commit -m "feat: safeguard FISTA with backtracking and restart"
```

- [ ] **Step 4: Add failing continuation tests**

Append:

```python
def test_lambda_path_warm_starts_each_stage_and_finishes_at_target():
    result = solve_fista(
        np.array([[1.0]]), np.array([2.0]), 0.5,
        lambda_path=[2.0, 1.0, 0.5], certificate_stride=1,
        kkt_tol=1e-7, gap_tol=1e-8,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [1.5], atol=1e-6)
    assert result.certificate.lam == 0.5
    assert {cert.stage for cert in result.history} == {0, 1, 2}
    assert [cert.iteration for cert in result.history if cert.stage == 1][0] == 0
    assert result.total_iterations >= result.iterations


def test_lambda_path_must_be_strictly_decreasing_and_end_at_target():
    for path in ([2.0, 1.0, 1.0, 0.5], [2.0, 1.0]):
        with pytest.raises(ValueError, match="lambda_path"):
            solve_fista(np.array([[1.0]]), np.array([2.0]), 0.5, lambda_path=path)
```

- [ ] **Step 5: Verify RED and implement stage-wise continuation**

Run `python -m pytest tests/test_fista.py -k "lambda_path" -v`; expect unexpected-keyword failures.

Extend the signature with `lambda_path=None`, `continuation_tol_scale=10.0`. Validate:

```python
path = [float(lam)] if lambda_path is None else [float(v) for v in lambda_path]
if (not path or path[-1] != lam or any(not np.isfinite(v) or v <= 0 for v in path)
        or any(path[i] <= path[i + 1] for i in range(len(path) - 1))):
    raise ValueError("lambda_path must be positive, strictly decreasing, and end at lam")
if continuation_tol_scale < 1.0:
    raise ValueError("continuation_tol_scale must be at least one")
```

Refactor the base iteration into a private stage runner returning `(x, status, iterations, L, restart_count, certificate, history)`. For stage `s`, use stage-local iteration numbers starting with an initial certificate at zero, warm-start from the previous stage's `x`, and use:

```python
scale = 1.0 if s == len(path) - 1 else continuation_tol_scale
stage_kkt_tol = kkt_tol * scale
stage_gap_tol = gap_tol * scale
```

Intermediate stages proceed after either dual-certificate convergence or their own `max_iter`; only the final stage determines public `status`. Return final-stage `iterations`, summed `total_iterations`, summed `restart_count`, final `L`, and concatenated stage histories.

- [ ] **Step 6: Verify GREEN, run all tests, and commit**

Run `python -m pytest tests/test_fista.py -k "lambda_path" -v`, then `python -m pytest -v`.
Expected: all tests pass.

```powershell
git add src/optimization_math/solvers/fista.py tests/test_fista.py
git commit -m "feat: add FISTA lambda continuation"
```

---

### Task 5: Public Usage Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/survey/2026-11-13-L1-L2-稀疏求解算法调研.md`
- Modify: `docs/superpowers/specs/2026-11-15-nonnegative-fista-design.md`
- Modify: `docs/superpowers/plans/2026-11-15-nonnegative-fista.md`

**Interfaces:**
- Consumes: All public functions and result fields from Tasks 1–4.
- Produces: A minimal runnable example and documentation consistent with the implemented certificate cost and augmented dual.

- [ ] **Step 1: Add the minimal README example**

Document this runnable API, not an aspirational interface:

```python
import numpy as np
from optimization_math.solvers.fista import solve_fista

A = np.array([[1.0, 1.0], [0.0, 1.0]])
y = np.array([2.0, 1.0])
result = solve_fista(
    A, y, 0.1,
    lambda_path=[1.0, 0.5, 0.1],
    certificate_stride=10,
)
assert result.status == "converged"
print(result.x)
print(result.certificate.kkt_residual, result.certificate.relative_gap)
```

Add installation commands `python -m pip install -e ".[test]"` and `python -m pytest -v`. Update the directory tree to show `operators.py`, `certificates.py`, and `solvers/fista.py`.

- [ ] **Step 2: Correct certificate-cost and elastic-net dual prose**

In README, replace “每步免费产出证书” with a statement that certificates are evaluated at a configurable interval and at the final iterate.

In the survey:

- State that ISTA can reuse the current-point gradient, whereas accelerated FISTA needs an extra `A/Aᵀ` pair for an accepted-iterate certificate.
- Add the elastic-net augmented dual definitions `nu=theta*r`, `eta=theta*sqrt(mu)*x`, `q=A.T@r+mu*x`, with `theta(q)>=-lambda` feasibility.
- Preserve the existing LASSO formulas and roadmap; do not rewrite unrelated survey content.

Update the specification status to record user approval and the corrected dual sign. Mark completed plan checkboxes only after their commands have actually passed.

- [ ] **Step 3: Run the README example and full verification**

Run:

```powershell
python -m pip install -e ".[test]"
python -m pytest -v
git diff --check
git status --short --branch
```

Expected: all tests pass without warnings; `git diff --check` has no output. Run the README example with Python and confirm it prints `converged`-quality final certificates. Scan changed text for absolute user paths, email addresses, tokens, and private keys; remove any accidental additions without altering the already-public historical author metadata.

- [ ] **Step 4: Commit documentation and report limits**

```powershell
git add README.md docs
git commit -m "docs: document verified FISTA usage and certificates"
git status --short --branch
```

Final report must list test count, local commit IDs, and explicit non-claims: no real FFT operator test, no 285,696-voxel benchmark, no EF-Spin integration, and no ADMM implementation. Do not push.
