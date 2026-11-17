"""Independent mathematical anchors and end-to-end standard comparisons."""
from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest

from benchmarks.standard_problems import (
    lasso_optimum, logistic_optimum, make_standard_cases, tv_optimum,
)
from benchmarks.bench_standard import CountingOperator, _assert_benchmark, benchmark_standard
from benchmarks import bench_standard as standard


@pytest.mark.parametrize("nonnegative,mu,expected", [
    (False, 0., [-.95, 0., .45]),
    (True, 0., [0., 0., .45]),
    (False, .2, [-1.9 / 2.2, 0., .9 / 2.2]),
    (True, .2, [0., 0., .9 / 2.2]),
])
def test_literal_lasso(nonnegative, mu, expected):
    np.testing.assert_allclose(
        lasso_optimum(np.array([-2., .05, 1.]), np.full(3, 2.), .1, mu, nonnegative),
        expected,
    )


@pytest.mark.parametrize("lam,expected", [(.25, [.25, 1.75]), (2., [1., 1.])])
def test_literal_two_point_tv(lam, expected):
    np.testing.assert_allclose(tv_optimum(np.array([0., 2.]), lam), expected)


def test_logistic_zero_and_nonzero_literal():
    np.testing.assert_array_equal(logistic_optimum(np.ones(2), np.array([1., -1.]), .3), 0.)
    np.testing.assert_allclose(
        logistic_optimum(np.ones(2), np.array([1., -1.]), .1),
        [np.log(4.), -np.log(4.)],
    )


def test_cases_are_immutable_and_oracles_satisfy_independent_kkt():
    for case in make_standard_cases(seeds=(0,), conditions=(1., 100.)):
        with pytest.raises(FrozenInstanceError):
            case.mu = 5.
        with pytest.raises(ValueError):
            case.x_opt.flat[0] = 2.
        assert np.isfinite(case.objective(case.x_opt))
        if case.problem in ("signed_lasso", "nonnegative_lasso"):
            g = case.A.T @ (case.A @ case.x_opt + case.b - case.y) + case.mu * case.x_opt
            active = case.x_opt != 0
            np.testing.assert_allclose(g[active] + case.lam * np.sign(case.x_opt[active]), 0, atol=1e-10)
            if case.problem == "signed_lasso":
                assert np.all(np.abs(g[~active]) <= case.lam + 1e-10)
            else:
                assert np.all(g[~active] >= -case.lam - 1e-10)
        if case.reference_type == "manufactured_kkt":
            assert case.truth is None


def test_counting_operator_counts_every_call():
    op = CountingOperator(np.eye(2))
    for _ in range(3):
        op.matvec(np.ones(2))
    op.rmatvec(np.ones(2))
    assert (op.forward_calls, op.adjoint_calls) == (3, 1)


def test_empty_matrix_rejected():
    with pytest.raises(AssertionError, match="empty"):
        _assert_benchmark([])


def test_standard_matrix():
    rows = benchmark_standard(seeds=(0, 1), conditions=(1., 100.))
    _assert_benchmark(rows)
    assert {r.problem for r in rows} == {
        "signed_lasso", "nonnegative_lasso", "box_qp", "nnls",
        "logistic", "tv", "group", "nuclear",
    }
    for row in rows:
        json.dumps(row.to_dict())
        assert row.history
        if row.algorithm not in ("chambolle_tv", "scipy_lbfgsb"):
            assert dict(row.configuration)["certificate_stride"] == 5
        if row.algorithm == "admm":
            assert dict(row.configuration)["cg_tol"] == 1e-12
            assert dict(row.configuration)["residual_tol"] == 1e-6
        if row.forward_calls is not None:
            assert row.forward_calls > row.iterations
            assert row.adjoint_calls > row.iterations
    # Every paired algorithm sees a single immutable case/reference, not regenerated data.
    grouped = {}
    for row in rows:
        grouped.setdefault(row.case, []).append(row)
    for group in grouped.values():
        assert len({r.reference_objective for r in group}) == 1
        assert len({r.input_digest for r in group}) == 1


def test_acceptance_rejects_false_convergence_and_loose_exact_prox():
    rows = benchmark_standard(seeds=(0,), conditions=(1.,))
    row = rows[0]
    for bad in (replace(row, status="max_iter"),
                replace(row, solution_error=.01),
                replace(row, normalized_objective_error=1e-3),
                replace(row, kkt_residual=1e-3),
                replace(row, objective=float("nan"))):
        with pytest.raises(AssertionError):
            _assert_benchmark([bad])
    nuclear = next(r for r in rows if r.problem == "nuclear")
    with pytest.raises(AssertionError, match="solution error"):
        _assert_benchmark([replace(nuclear, solution_error=1e-5)])


def test_scipy_initial_certificate_skips_optimizer(monkeypatch):
    case = next(c for c in make_standard_cases((0,), (100.,))
                if c.problem == "logistic" and np.all(c.x_opt == 0))

    def forbidden(*args, **kwargs):
        pytest.fail("optimizer must not run when the initial certificate passes")

    monkeypatch.setattr(standard, "minimize", forbidden, raising=False)
    row = standard._run(case, "scipy_lbfgsb")
    assert row.status == "converged"
    assert row.iterations == 0
    assert [p.iteration for p in row.history] == [0]


@pytest.mark.parametrize("stride", [3, 5])
def test_scipy_stops_at_first_scheduled_common_certificate(monkeypatch, stride):
    monkeypatch.setitem(standard.OPTIONS, "certificate_stride", stride)
    for case in make_standard_cases((0,), (100.,)):
        if case.problem not in ("box_qp", "nnls", "logistic"):
            continue
        row = standard._run(case, "scipy_lbfgsb")
        _assert_benchmark([row])
        config = dict(row.configuration)
        assert config["certificate_stride"] == stride
        assert config["kkt_tol"] == config["gap_tol"] == standard.CERT_TOL
        assert row.history[-1].iteration == row.iterations
        assert all(p.iteration % stride == 0 for p in row.history[:-1])
        assert len({p.iteration for p in row.history}) == len(row.history)
        assert all(p.kkt_residual > standard.CERT_TOL or p.relative_gap > standard.CERT_TOL
                   for p in row.history[:-1])


def test_scipy_import_is_outside_run_timer(monkeypatch):
    import builtins
    original_import = builtins.__import__
    measured = False

    def clock():
        nonlocal measured
        measured = True
        return 1.

    def guarded_import(name, *args, **kwargs):
        if measured and name == "scipy.optimize":
            pytest.fail("dependency import inside measured solver run")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(standard, "perf_counter", clock)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    case = next(c for c in make_standard_cases((0,), (1.,)) if c.problem == "box_qp")
    _assert_benchmark([standard._run(case, "scipy_lbfgsb")])


def test_scipy_does_not_swallow_optimizer_errors(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("unrelated optimizer error")

    monkeypatch.setattr(standard, "minimize", broken, raising=False)
    case = next(c for c in make_standard_cases((0,), (1.,)) if c.problem == "box_qp")
    with pytest.raises(RuntimeError, match="unrelated optimizer error"):
        standard._run(case, "scipy_lbfgsb")


def test_scipy_native_success_does_not_replace_certificate_gate(monkeypatch):
    from types import SimpleNamespace

    def false_success(fun, start, **kwargs):
        return SimpleNamespace(x=start, nit=0, success=True, message="native success")

    monkeypatch.setattr(standard, "minimize", false_success)
    case = next(c for c in make_standard_cases((0,), (100.,)) if c.problem == "box_qp")
    row = standard._run(case, "scipy_lbfgsb")
    assert row.status == "failed: native success"
    assert row.kkt_residual > standard.CERT_TOL


def test_scipy_certificate_work_matches_recorded_checkpoints(monkeypatch):
    checked = []
    original = standard.compute_certificate_qp

    def counted(*args, **kwargs):
        checked.append(kwargs["iteration"])
        return original(*args, **kwargs)

    monkeypatch.setattr(standard, "compute_certificate_qp", counted)
    case = next(c for c in make_standard_cases((0,), (100.,)) if c.problem == "nnls")
    row = standard._run(case, "scipy_lbfgsb")
    assert checked == [p.iteration for p in row.history]
    assert checked == list(range(0, row.iterations + 1, 5))
    assert row.status == "converged"
