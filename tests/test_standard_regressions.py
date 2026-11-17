"""Deterministic regressions for structured-solver certificate failures."""

import numpy as np
import pytest

from optimization_math.certificates_group import compute_certificate_group
from optimization_math.operators_difference import first_difference_operator
from optimization_math.solvers.chambolle_tv import (
    certificate_tv_chambolle,
    solve_tv_chambolle,
)
from optimization_math.solvers.fista_group import solve_fista_group


def _group_entrypoint(entrypoint, groups, *, lam=1.0, mu=0.0):
    A = np.eye(2)
    y = np.array([0.0, 2.0])
    if entrypoint == "solver":
        return solve_fista_group(A, y, lam, groups, mu=mu, lipschitz=1.0)
    return compute_certificate_group(A, y, np.zeros(2), lam, groups, mu=mu)


@pytest.mark.parametrize("entrypoint", ["solver", "certificate"])
@pytest.mark.parametrize(
    "groups",
    [
        pytest.param([[0]], id="missing-coordinate-false-convergence"),
        pytest.param([[0, 1], [1]], id="overlap"),
        pytest.param([[0, 0], [1]], id="repeated-within-group"),
        pytest.param([[0], [0]], id="repeated-and-missing"),
        pytest.param([[0.0], [1.0]], id="float-dtype"),
        pytest.param([[0.5], [1]], id="fractional"),
        pytest.param([[False, True]], id="bool-dtype"),
        pytest.param([[False, 1]], id="mixed-bool"),
        pytest.param([[-1], [1]], id="negative-index"),
        pytest.param([[0], [2]], id="out-of-bounds"),
        pytest.param([np.array([0, 2**64 - 1], dtype=np.uint64)], id="unsigned-bounds"),
        pytest.param([], id="empty-layout"),
        pytest.param([[], [0, 1]], id="empty-group"),
        pytest.param([[[0, 1]]], id="nonvector-group"),
    ],
)
def test_group_entrypoints_reject_nonpartitions(entrypoint, groups):
    with pytest.raises(ValueError):
        _group_entrypoint(entrypoint, groups)


@pytest.mark.parametrize("entrypoint", ["solver", "certificate"])
@pytest.mark.parametrize(
    "groups",
    [
        [[2, 0], [3, 1]],
        [np.array([2, 0], dtype=np.int32), np.array([3, 1], dtype=np.uint64)],
    ],
    ids=["python-ints", "numpy-ints"],
)
def test_group_entrypoints_accept_permuted_partition(entrypoint, groups):
    # Each group has data norm 5, so lambda=1 shrinks each by 4/5.
    y = np.array([3.0, 0.0, 4.0, 5.0])
    expected = np.array([2.4, 0.0, 3.2, 4.0])
    if entrypoint == "solver":
        result = solve_fista_group(
            np.eye(4), y, 1.0, groups, lipschitz=1.0,
            certificate_stride=1, kkt_tol=1e-10, gap_tol=1e-10,
        )
        assert result.status == "converged"
        np.testing.assert_allclose(result.x, expected, rtol=0, atol=1e-14)
        cert = result.certificate
    else:
        cert = compute_certificate_group(np.eye(4), y, expected, 1.0, groups)
    assert cert.primal == pytest.approx(9.0, abs=1e-14)
    assert cert.dual == pytest.approx(9.0, abs=1e-14)
    assert cert.kkt_residual <= 1e-14
    assert cert.relative_gap <= 1e-14


@pytest.mark.parametrize("entrypoint", ["solver", "certificate"])
@pytest.mark.parametrize("parameter", ["lam", "mu"])
@pytest.mark.parametrize("value", [None, "not-a-number"])
def test_group_parameter_conversion_raises_valueerror(entrypoint, parameter, value):
    with pytest.raises(ValueError, match="lam.*mu"):
        _group_entrypoint(entrypoint, [[0], [1]], **{parameter: value})


@pytest.mark.parametrize("offset", [0.0, 1e4, 1e8, 1e12])
def test_tv_solver_certificate_is_invariant_to_dc_offset(offset):
    result = solve_tv_chambolle(
        np.array([0.0, 2.0]) + offset, 0.25, tol=1e-10, max_iter=1000,
    )
    assert result.status == "converged"
    assert result.iterations < 1000
    # Absolute comparison: relative tolerance would hide errors at large offsets.
    np.testing.assert_allclose(
        result.x - offset, [0.25, 1.75], rtol=0, atol=1e-10,
    )
    assert result.certificate.primal == pytest.approx(0.4375, abs=1e-10)
    assert result.certificate.dual == pytest.approx(0.4375, abs=1e-10)
    assert result.certificate.relative_gap <= 1e-10
    assert result.certificate.kkt_residual <= 1e-10


@pytest.mark.parametrize("offset", [0.0, 1e4, 1e8, 1e12])
def test_tv_public_certificate_is_tight_at_exact_shifted_optimum(offset):
    cert = certificate_tv_chambolle(
        first_difference_operator(2), np.array([0.0, 2.0]) + offset,
        np.array([0.25, 1.75]) + offset, 0.25, np.array([1.0]),
    )
    assert cert.primal == 0.4375
    assert cert.dual == 0.4375
    assert cert.relative_gap == 0.0
    assert cert.kkt_residual == 0.0


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("x", np.array([np.nan, 1.75])),
        ("x", np.array([0.25, np.inf])),
        ("y", np.array([np.nan, 2.0])),
        ("y", np.array([0.0, -np.inf])),
        ("p", np.array([np.nan])),
        ("p", np.array([np.inf])),
        ("x", np.array(0.25)),
        ("x", np.array([[0.25, 1.75]])),
        ("x", np.array([0.25])),
        ("x", np.array([0.25, 1.75, 2.0])),
        ("y", np.array(0.0)),
        ("y", np.array([[0.0, 2.0]])),
        ("y", np.array([0.0])),
        ("y", np.array([0.0, 2.0, 3.0])),
        ("p", np.array(1.0)),
        ("p", np.array([[1.0]])),
        ("p", np.array([])),
        ("p", np.array([1.0, 1.0])),
        ("p", np.array([1.0 + 1e-8])),
        ("p", np.array([-1.0 - 1e-8])),
        ("lam_tv", -0.25),
        ("lam_tv", np.nan),
        ("lam_tv", np.inf),
        ("lam_tv", None),
        ("lam_tv", "not-a-number"),
    ],
)
def test_tv_public_certificate_rejects_invalid_input(parameter, value):
    kwargs = dict(
        y=np.array([0.0, 2.0]), x=np.array([0.25, 1.75]),
        lam_tv=0.25, p=np.array([1.0]),
    )
    kwargs[parameter] = value
    with pytest.raises(ValueError):
        certificate_tv_chambolle(first_difference_operator(2), **kwargs)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_tv_certificate_clips_roundoff_to_a_feasible_dual(sign):
    cert = certificate_tv_chambolle(
        first_difference_operator(2), sign * np.array([0.0, 2.0]),
        sign * np.array([0.25, 1.75]), 0.25,
        np.array([sign * (1.0 + 5e-15)]),
    )
    assert cert.dual <= cert.primal
    assert cert.dual == 0.4375
    assert cert.relative_gap == 0.0
