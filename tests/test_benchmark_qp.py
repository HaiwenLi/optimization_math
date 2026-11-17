from benchmarks.bench_qp import benchmark_qp


def test_qp_benchmark_converges_and_recovers():
    rows = benchmark_qp(seeds=(0,))
    assert all(r["status"] == "converged" for r in rows)
    assert all(r["rel_L2"] < 1e-4 for r in rows)
    assert all(r["gap"] < 1e-5 for r in rows)
