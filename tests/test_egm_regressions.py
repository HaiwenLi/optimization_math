"""EGM Goldfish Round 1 regressions: each test pins a confirmed review finding."""
from dataclasses import replace
import os

import numpy as np
import pytest

from benchmarks.bench_standard import OPTIONS, _assert_benchmark, benchmark_standard, main
from benchmarks.standard_report import write_standard_report
from optimization_math.certificates_nuclear import compute_certificate_nuclear
from optimization_math.certificates_qp import compute_certificate_qp
from optimization_math.solvers.fista_group import solve_fista_group
from optimization_math.solvers.fista_nuclear import solve_fista_nuclear
from optimization_math.solvers.qp_box import solve_qp_box


class IdentityMatrixOp:
    def __init__(self, shape):
        self.shape = shape

    def matvec(self, x):
        return x.ravel()

    def rmatvec(self, y):
        return y.reshape(self.shape)


# --- F1: QP certificate must reject indefinite / asymmetric Q ----------------

@pytest.mark.parametrize("Q", [
    np.diag([1.0, -1.0]),          # unbounded below, saddle certified optimal
    np.diag([1.0, -0.5]),          # bounded nonconvex, wrong minimum certified
    np.array([[0.0, 1.0], [-2.0, 0.0]]),  # strongly asymmetric
])
def test_qp_rejects_indefinite_or_asymmetric_hessian(Q):
    n = Q.shape[0]
    with pytest.raises(ValueError):
        solve_qp_box(Q, np.zeros(n), -np.ones(n), np.ones(n))
    with pytest.raises(ValueError):
        compute_certificate_qp(Q, np.zeros(n), np.zeros(n), None, None)


@pytest.mark.parametrize("Q", [
    np.diag([1.0, 0.0]),                       # singular PSD stays supported
    np.diag([1.0, -3.5e-15]),                  # just inside the 8*eps*n allowance
    np.array([[2.0, 1.0 + 1e-15], [1.0, 2.0]]),  # roundoff-level asymmetry
])
def test_qp_accepts_psd_up_to_roundoff(Q):
    n = Q.shape[0]
    solve_qp_box(Q, np.ones(n), -np.ones(n), np.ones(n))  # must not raise
    compute_certificate_qp(Q, np.ones(n), np.zeros(n), None, None)


def test_qp_rejects_just_past_roundoff_allowance():
    # Allowance is 8*eps*n*max|eig| = 3.55e-15 here; -1e-14 must be rejected.
    with pytest.raises(ValueError):
        solve_qp_box(np.diag([1.0, -1e-14]), np.zeros(2), None, None)


def test_qp_asymmetric_certificate_had_fake_finite_dual():
    # Previously returned dual=-0.25 on an unbounded problem.
    with pytest.raises(ValueError):
        compute_certificate_qp(np.array([[2.0, 4.0], [0.0, 2.0]]),
                               np.array([0.0, -1.0]), np.zeros(2), None, None)


# --- F2: solve_fista_group lipschitz uses the sibling convention ||A||^2 ----

def test_group_lipschitz_is_squared_norm_mu_added_internally():
    A = np.eye(2)
    y = np.array([1.0, 2.0])
    lam, mu = 0.5, 20.0
    result = solve_fista_group(A, y, lam, [np.array([0]), np.array([1])],
                               mu=mu, lipschitz=1.0)  # ||A||^2 = 1, sibling units
    assert result.status == "converged"
    assert result.lipschitz == pytest.approx(1.0 + mu)
    expected = y * np.maximum(1.0 - lam / np.abs(y), 0.0) / (1.0 + mu)
    np.testing.assert_allclose(result.x, expected, atol=1e-8)


# --- F3: benchmark problem knowledge must fail loud on unknown problems ------

def test_assert_benchmark_unknown_problem_fails_loud():
    row = replace(benchmark_standard(seeds=(0,), conditions=(1.,))[0],
                  problem="future_problem")
    with pytest.raises(AssertionError, match="_SOLUTION_LIMITS"):
        _assert_benchmark([row])


def test_certificate_semantics_mapping_is_exhaustive():
    from benchmarks import bench_standard
    known = {"signed_lasso", "nonnegative_lasso", "box_qp", "nnls",
             "logistic", "tv", "group", "nuclear"}
    assert set(bench_standard._CERTIFICATE_SEMANTICS) == known
    with pytest.raises(KeyError):
        bench_standard._CERTIFICATE_SEMANTICS["future_problem"]


# --- F5: --out-dir guard must hold on Windows and protect the repo root ------

@pytest.mark.parametrize("bad", ["/abs", "/x/y", "..", "a/../b", ".", "",
                                 "./", "foo/..", " ", "foo ", "foo."])
def test_out_dir_rejects_absolute_parent_and_dot(bad, monkeypatch):
    monkeypatch.setattr("benchmarks.bench_standard.benchmark_standard", lambda *a, **k: [])
    monkeypatch.setattr("benchmarks.bench_standard._assert_benchmark", lambda *a, **k: None)
    monkeypatch.setattr("benchmarks.standard_report.write_standard_report",
                        lambda *a, **k: None)
    with pytest.raises(SystemExit):
        main(["--out-dir", bad])


@pytest.mark.skipif(os.name != "nt", reason="drive-relative paths are Windows-specific")
@pytest.mark.parametrize("bad", ["C:foo", "C:\\abs", "\\\\share\\x"])
def test_out_dir_rejects_windows_drive_paths(bad, monkeypatch):
    monkeypatch.setattr("benchmarks.bench_standard.benchmark_standard", lambda *a, **k: [])
    monkeypatch.setattr("benchmarks.bench_standard._assert_benchmark", lambda *a, **k: None)
    monkeypatch.setattr("benchmarks.standard_report.write_standard_report",
                        lambda *a, **k: None)
    with pytest.raises(SystemExit):
        main(["--out-dir", bad])


def test_out_dir_accepts_relative_subdirectory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("benchmarks.bench_standard.benchmark_standard", lambda *a, **k: [])
    monkeypatch.setattr("benchmarks.bench_standard._assert_benchmark", lambda *a, **k: None)
    monkeypatch.setattr("benchmarks.standard_report.write_standard_report",
                        lambda *a, **k: None)
    main(["--out-dir", "benchmarks/out/nested"])


# --- F7: nuclear entry points validate matrix shapes --------------------------

def test_nuclear_solver_rejects_mismatched_x0_shape():
    y = np.ones(6)
    with pytest.raises(ValueError, match="must match"):
        solve_fista_nuclear(IdentityMatrixOp((6, 6)), y, 1.0, X0=np.zeros((1, 6)))


def test_nuclear_certificate_rejects_mismatched_shape():
    y = np.ones(6)
    with pytest.raises(ValueError, match="must match"):
        compute_certificate_nuclear(IdentityMatrixOp((6, 6)), y, np.zeros((1, 6)), 1.0)


def test_nuclear_certificate_rejects_mismatched_y_length():
    # Must fail loud by design, not because a downstream dot() happens to
    # size-check.
    with pytest.raises(ValueError, match="matching A's output size"):
        compute_certificate_nuclear(IdentityMatrixOp((4, 4)), np.array([0.5]),
                                    np.zeros((4, 4)), 0.3)


# --- F11: configuration serializes as a JSON object ---------------------------

def test_configuration_serializes_as_object():
    row = benchmark_standard(seeds=(0,), conditions=(1.,))[0]
    config = row.to_dict()["configuration"]
    assert isinstance(config, dict)
    assert config["max_iter"] == OPTIONS["max_iter"]
    assert config["certificate_stride"] == OPTIONS["certificate_stride"]


# --- F12: mixed counted/uncounted groups do not crash the report --------------

@pytest.fixture(scope="module")
def rows():
    return benchmark_standard(seeds=(0,), conditions=(1.,))


@pytest.mark.parametrize("order", ["counted-first", "counted-last"])
def test_report_handles_mixed_count_availability(rows, tmp_path, order):
    pytest.importorskip("matplotlib")
    counted = rows[0]
    uncounted = replace(counted, forward_calls=None, adjoint_calls=None)
    group = [counted, uncounted] if order == "counted-first" else [uncounted, counted]
    paths = write_standard_report(group, tmp_path)
    text = paths["markdown"].read_text(encoding="utf-8")
    assert "mixed (n/a)" in text
