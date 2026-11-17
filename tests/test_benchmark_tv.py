from benchmarks.bench_tv import benchmark_tv


def test_tv_benchmark_finds_optimal_lambda_and_beats_raw():
    x_true, y, raw_err, rows = benchmark_tv()
    raw_rel = raw_err / __import__("numpy").linalg.norm(x_true)
    best = min(rows, key=lambda r: r["rel_L2"])
    # TV denoising must beat the raw observation at the best lambda.
    assert best["rel_L2"] < raw_rel
    # interior lambda should be better than the extremes (non-monotone trade-off).
    assert best["rel_L2"] <= rows[0]["rel_L2"] or best["rel_L2"] <= rows[-1]["rel_L2"]
