from benchmarks.bench_structured import benchmark_group, benchmark_nuclear


def test_group_benchmark_joint_selection():
    g = benchmark_group()
    assert g["group"]["coherence"] >= g["plain"]["coherence"]


def test_nuclear_benchmark_denoises_and_recovers_rank():
    nuc = benchmark_nuclear()
    assert nuc["best_err"] < nuc["noise"]
    assert nuc["rank_recovered"]
