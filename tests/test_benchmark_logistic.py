import numpy as np

from benchmarks.bench_logistic import benchmark_logistic


def test_logistic_benchmark_converges_and_certifies():
    pairs = benchmark_logistic(seeds=(0,), lams=(0.05,))
    lam, r = pairs[0]
    assert lam == 0.05
    assert r.status == "converged"
    assert r.kkt_residual <= 1e-5
    assert r.relative_gap <= 1e-5


def test_logistic_lambda_tradeoff_is_measured():
    pairs = benchmark_logistic(seeds=(0,), lams=(0.05, 0.01))
    by_lam = {lam: r for lam, r in pairs}
    # Smaller lambda -> more iterations (weaker penalty, slower convergence).
    assert by_lam[0.01].iterations > by_lam[0.05].iterations
