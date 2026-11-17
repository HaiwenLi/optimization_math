import numpy as np

from benchmarks.bench_logistic import benchmark_logistic


def test_logistic_benchmark_converges_and_certifies():
    rows = benchmark_logistic(seeds=(0,), lams=(0.05,))
    r = rows[0]
    assert r.status == "converged"
    assert r.kkt_residual <= 1e-5
    assert r.relative_gap <= 1e-5


def test_logistic_lambda_tradeoff_is_measured():
    rows = benchmark_logistic(seeds=(0,), lams=(0.05, 0.01))
    # Smaller lambda -> more iterations (weaker penalty, slower convergence).
    high = [r for r in rows if r.solver.endswith("0.05)")][0]
    low = [r for r in rows if r.solver.endswith("0.01)")][0]
    assert low.iterations > high.iterations
