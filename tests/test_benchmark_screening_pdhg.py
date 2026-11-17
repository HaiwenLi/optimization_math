"""Automatic-pass tests for the screening and PDHG benchmarks."""
import numpy as np

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
