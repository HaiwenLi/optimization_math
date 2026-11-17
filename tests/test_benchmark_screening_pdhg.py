"""Automatic-pass tests for the screening and PDHG benchmarks."""
import numpy as np
import pytest

from benchmarks.bench_screening import benchmark_screening, _assert_benchmark as _assert_screening
from benchmarks.bench_pdhg import benchmark_pdhg, _assert_benchmark as _assert_pdhg


def test_screening_benchmark_meets_thresholds():
    rows = benchmark_screening(seeds=(0, 1))
    _assert_screening(rows)


def test_screening_is_safe_and_effective():
    rows = benchmark_screening(seeds=(2,))
    r = rows[0]
    # Safety: never removes an optimum-support coordinate.
    assert r["opt_support_kept"]
    # Effectiveness: prunes a large fraction of coordinates.
    assert r["screened_frac"] >= 0.5


def test_pdhg_benchmark_meets_thresholds():
    rows = benchmark_pdhg(seeds=(0, 1))
    _assert_pdhg(rows)


def test_pdhg_matches_fista_recovery():
    rows = benchmark_pdhg(seeds=(3,))
    pdhg = rows["pdhg"][0]
    fista = rows["fista-signed"][0]
    assert np.allclose(pdhg.recovery.relative_l2, fista.recovery.relative_l2, atol=1e-4)
    assert pdhg.recovery.support_f1 == fista.recovery.support_f1


def test_screening_plot_uses_absolute_gap_and_correct_radius(monkeypatch, tmp_path):
    from types import SimpleNamespace
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes
    from benchmarks import bench_screening as bench

    # Supply a known iterate: theta=1/2, absolute gap=1/2. The plot must
    # not sweep smaller (unsafe) gaps, and must screen the second coordinate
    # at the first point (radius=1, not the old radius=2).
    monkeypatch.setattr(bench, "make_problem", lambda seed:
                        (np.diag([1.0, 0.75]), np.array([2.0, 0.0]),
                         np.array([1.0, 0.0]), 1.0, 1.0))
    monkeypatch.setattr(bench, "solve_fista_signed",
                        lambda *args, **kwargs: SimpleNamespace(x=np.zeros(2)))
    monkeypatch.setattr(bench, "OUT_DIR", tmp_path)
    captured = {}
    original_plot = Axes.semilogx

    def capture_plot(ax, x, y, *args, **kwargs):
        captured["gaps"] = np.asarray(x)
        captured["fractions"] = np.asarray(y)
        return original_plot(ax, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "semilogx", capture_plot)
    bench._plot_screening_radius()
    assert np.min(captured["gaps"]) >= 0.5
    assert captured["fractions"][0] == 0.5
    assert (tmp_path / "screening_radius.png").stat().st_size > 0
