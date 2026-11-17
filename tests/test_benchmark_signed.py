import numpy as np
import pytest

from benchmarks.bench_signed_lasso import benchmark_signed_lasso, _assert_benchmark


def test_signed_lasso_benchmark_meets_objective_thresholds():
    rows = benchmark_signed_lasso(seeds=(0, 1))
    _assert_benchmark(rows)


def test_signed_lasso_recovers_negative_weights_better_than_nonnegative():
    rows = benchmark_signed_lasso(seeds=(2,))
    signed = rows["fista-signed"][0]
    nonneg = rows["fista-nonneg"][0]
    assert signed.recovery.support_f1 > nonneg.recovery.support_f1
    # The nonnegative solver zeroes negative coordinates; signed keeps them.
    assert signed.recovery.relative_l2 < nonneg.recovery.relative_l2
