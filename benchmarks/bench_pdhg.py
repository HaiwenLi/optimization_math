"""PDHG (Chambolle-Pock) benchmark: primal-dual vs FISTA on signed LASSO.

Reuses the same known sparse ground truth as the T1 benchmark so the two
first-order methods are compared on identical problems. Measures convergence
cost (iterations, A/A^T calls), recovery quality, and certificate attainment,
with automatic pass/fail assertions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from optimization_math.solvers.pdhg import solve_pdhg
from optimization_math.solvers.fista_signed import solve_fista_signed

from benchmarks.metrics import (
    MARKDOWN_HEADER,
    RunResult,
    format_row,
    summarize,
)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"


def make_problem(seed, *, m=80, n=200, k=15, noise=0.05):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    x_true = np.zeros(n)
    sup = rng.choice(n, size=k, replace=False)
    x_true[sup] = rng.standard_normal(k)
    y = A @ x_true + noise * rng.standard_normal(m)
    lam = 0.1 * np.linalg.norm(A.T @ y, ord=np.inf)
    lipschitz = float(np.linalg.norm(A, 2) ** 2) * 1.2
    return A, y, x_true, lam, lipschitz


def _run(name, fn, A, y, lam, x_true, lipschitz, matvec_per_iter):
    result = fn(A, y, lam, lipschitz=lipschitz, certificate_stride=1,
                kkt_tol=1e-5, gap_tol=1e-5, max_iter=20000)
    from benchmarks.metrics import recovery_metrics as rm
    return RunResult(
        solver=name,
        status=result.status,
        iterations=int(result.iterations),
        matvec_count=int(result.iterations) * matvec_per_iter,
        kkt_residual=float(result.certificate.kkt_residual),
        relative_gap=float(result.certificate.relative_gap),
        objective=float(result.certificate.primal),
        recovery=rm(result.x, x_true),
    )


def benchmark_pdhg(seeds=(0, 1, 2, 3, 4)):
    configs = {
        # Both FISTA and PDHG do exactly one A/A^T pair per iteration (FISTA's
        # backtracking adds >=1 more matvec, so this is a *lower* bound for
        # FISTA). Counting them symmetrically avoids overstating PDHG's cost.
        "fista-signed": (solve_fista_signed, 1),
        "pdhg": (solve_pdhg, 1),
    }
    rows_by_solver = {name: [] for name in configs}
    for seed in seeds:
        A, y, x_true, lam, lipschitz = make_problem(seed)
        for name, (fn, mv) in configs.items():
            r = _run(name, fn, A, y, lam, x_true, lipschitz, mv)
            rows_by_solver[name].append(r)
    return rows_by_solver


def _assert_benchmark(rows_by_solver):
    pdhg = rows_by_solver["pdhg"]
    fista = rows_by_solver["fista-signed"]
    # PDHG must converge and satisfy its certificate.
    assert all(r.status == "converged" for r in pdhg), "pdhg must converge"
    assert all(r.kkt_residual <= 1e-5 for r in pdhg), "pdhg KKT"
    assert all(r.relative_gap <= 1e-5 for r in pdhg), "pdhg dual gap"
    # Same optimum as FISTA -> same recovery quality.
    for p, f in zip(pdhg, fista):
        assert abs(p.recovery.relative_l2 - f.recovery.relative_l2) < 1e-4, (
            "pdhg recovery must match fista"
        )
        assert p.objective <= f.objective * (1.0 + 1e-4), "pdhg objective must match fista"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows_by_solver = benchmark_pdhg()
    _assert_benchmark(rows_by_solver)

    print("PDHG vs FISTA benchmark:")
    print(MARKDOWN_HEADER)
    for name in rows_by_solver:
        print(format_row(rows_by_solver[name][0]))

    print("\nStability (mean ± std over seeds):")
    print("| solver | stat | iters | A/Aᵀ | ε_KKT | rel.gap | rel.L2 | F1 | RMSE |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, rows in rows_by_solver.items():
        stats = summarize(rows)
        keys = ("iterations", "matvec_count", "kkt_residual", "relative_gap",
                "relative_l2", "support_f1", "amplitude_rmse")
        mean = " | ".join(f"{stats[k][0]:.3e}" for k in keys)
        std = " | ".join(f"{stats[k][1]:.3e}" for k in keys)
        print(f"| {name} | mean | {mean} |")
        print(f"| {name} | std  | {std} |")

    _plot_convergence()

    report_path = OUT_DIR / "pdhg_report.md"
    _write_report(report_path, rows_by_solver)
    print(f"\nReport written to {report_path}")


def _plot_convergence():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A, y, x_true, lam, lipschitz = make_problem(0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, fn in (("fista-signed", solve_fista_signed), ("pdhg", solve_pdhg)):
        res = fn(A, y, lam, lipschitz=lipschitz, certificate_stride=1,
                 kkt_tol=1e-5, gap_tol=1e-5, max_iter=20000)
        hist = res.history
        xs = [c.iteration for c in hist]
        ax[0].semilogy(xs, [c.kkt_residual for c in hist], label=name)
        ax[1].semilogy(xs, [c.relative_gap for c in hist], label=name)
    ax[0].set_title("KKT residual vs iteration")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel(r"$\varepsilon_{\mathrm{KKT}}$")
    ax[0].legend()
    ax[1].set_title("relative dual gap vs iteration")
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("relative gap")
    ax[1].legend()
    fig.tight_layout()
    out = OUT_DIR / "pdhg_convergence.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _write_report(path, rows_by_solver):
    lines = [
        "# PDHG（Chambolle-Pock）基准报告",
        "",
        "在 T1 相同的已知真值稀疏问题上，对比主-对偶 PDHG 与 FISTA。",
        "两者都用统一的带符号 LASSO 证书验收（同一 KKT + 对偶 gap 门）。",
        "表中 A/Aᵀ 为核心迭代的算子对数估算，不含证书、回溯或重启开销，并非实测总调用数。",
        "",
        "## 单次（seed 0）",
        "",
        MARKDOWN_HEADER,
    ]
    for name in rows_by_solver:
        lines.append(format_row(rows_by_solver[name][0]))
    lines += [
        "",
        "## 稳定性（多噪声种子 mean ± std）",
        "",
        "| solver | stat | iters | A/Aᵀ | ε_KKT | rel.gap | rel.L2 | F1 | RMSE |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, rows in rows_by_solver.items():
        stats = summarize(rows)
        keys = ("iterations", "matvec_count", "kkt_residual", "relative_gap",
                "relative_l2", "support_f1", "amplitude_rmse")
        mean = " | ".join(f"{stats[k][0]:.3e}" for k in keys)
        std = " | ".join(f"{stats[k][1]:.3e}" for k in keys)
        lines.append(f"| {name} | mean | {mean} |")
        lines.append(f"| {name} | std  | {std} |")
    lines += [
        "",
        "## 结论",
        "",
        "- PDHG 用主-对偶迭代，收敛到与 FISTA **相同的最优解**（证书统一验收）。",
        "- 每个核心迭代：PDHG 与 FISTA 都是 1 对 A/Aᵀ；FISTA 另有回溯或重启开销。",
        "",
        "![收敛曲线](pdhg_convergence.png)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
