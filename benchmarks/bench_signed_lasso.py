"""T1 benchmark: signed LASSO / elastic-net against a known sparse ground truth.

Generates synthetic Gaussian-design sparse recovery problems whose truth
(including *negative* weights) is known, then scores each solver on:
  - convergence cost (iterations, A/A^T calls)
  - recovery vs truth (relative L2, support precision/recall/F1, amplitude RMSE)
  - certificate accuracy (epsilon_KKT, relative dual gap)
  - stability (mean +/- std over noise seeds)

Run as a script (prints tables, writes convergence plot and a markdown report)
or import ``benchmark_signed_lasso`` from a test for repeatable assertions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from optimization_math.solvers.fista_signed import solve_fista_signed
from optimization_math.solvers.fista import solve_fista  # nonnegative reference

from benchmarks.metrics import (
    MARKDOWN_HEADER,
    RunResult,
    format_row,
    summarize,
)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"


def make_problem(seed, *, m=80, n=200, k=15, noise=0.05):
    """Gaussian-design sparse regression with mixed-sign truth (negatives kept)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    x_true = np.zeros(n)
    support = rng.choice(n, size=k, replace=False)
    x_true[support] = rng.standard_normal(k)  # mixed sign
    y = A @ x_true + noise * rng.standard_normal(m)
    lam = 0.1 * np.linalg.norm(A.T @ y, ord=np.inf)
    lipschitz = float(np.linalg.norm(A, 2) ** 2) * 1.2
    return A, y, x_true, lam, lipschitz


def run_solver(name, fn, A, y, lam, x_true, *, matvec_per_iter):
    """Run one solver and package a RunResult. ``matvec_per_iter`` gives the
    number of A/A^T pairs per iteration for the cost column."""
    result = fn(A, y, lam)
    from benchmarks.metrics import recovery_metrics as rm
    objective = result.certificate.primal
    return RunResult(
        solver=name,
        status=result.status,
        iterations=int(result.iterations),
        matvec_count=int(result.iterations) * matvec_per_iter,
        kkt_residual=float(result.certificate.kkt_residual),
        relative_gap=float(result.certificate.relative_gap),
        objective=float(objective),
        recovery=rm(result.x, x_true),
    )


def benchmark_signed_lasso(seeds=(0, 1, 2, 3, 4)):
    """Run the full T1 benchmark over several noise seeds and return results."""
    configs = {
        "fista-signed": (solve_fista_signed, 1),
        "fista-nonneg": (solve_fista, 1),
    }
    rows_by_solver = {name: [] for name in configs}
    for seed in seeds:
        A, y, x_true, lam, lipschitz = make_problem(seed)
        for name, (fn, matvec_per_iter) in configs.items():
            def call(A, y, lam, fn=fn, name=name, lipschitz=lipschitz):
                kwargs = {}
                if name in ("fista-signed", "fista-nonneg"):
                    kwargs["lipschitz"] = lipschitz
                return fn(A, y, lam, certificate_stride=1,
                          kkt_tol=1e-5, gap_tol=1e-5, max_iter=20000, **kwargs)
            row = run_solver(name, call, A, y, lam, x_true,
                             matvec_per_iter=matvec_per_iter)
            rows_by_solver[name].append(row)
    return rows_by_solver


def _assert_benchmark(rows_by_solver):
    """Objective pass/fail thresholds — no human judgment required.

    Signed FISTA must converge, satisfy its certificate, recover support with a
    meaningful average F1, and (the key contrast) recover negative weights that
    the nonnegative solver discards — so signed F1 must dominate nonnegative F1
    on *every* seed.
    """
    signed = rows_by_solver["fista-signed"]
    nonneg = rows_by_solver["fista-nonneg"]
    assert len(signed) == len(nonneg)
    assert all(r.status == "converged" for r in signed), "fista-signed must converge"
    assert all(r.kkt_residual <= 1e-5 for r in signed), "KKT certificate"
    assert all(r.relative_gap <= 1e-5 for r in signed), "dual gap certificate"
    mean_f1 = np.mean([r.recovery.support_f1 for r in signed])
    assert mean_f1 >= 0.7, f"mean support F1 too low: {mean_f1:.3f}"
    for s, n in zip(signed, nonneg):
        assert s.recovery.support_f1 > n.recovery.support_f1, (
            "signed solver must recover negative-weight support better than nonnegative"
        )


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows_by_solver = benchmark_signed_lasso()
    _assert_benchmark(rows_by_solver)

    print("T1 signed LASSO benchmark — single-seed rows (seed 0):")
    print(MARKDOWN_HEADER)
    for name in rows_by_solver:
        print(format_row(rows_by_solver[name][0]))

    print("\nStability (mean ± std over seeds):")
    print("| solver | stat | iters | A/Aᵀ | ε_KKT | rel.gap | rel.L2 | F1 | RMSE |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, rows in rows_by_solver.items():
        stats = summarize(rows)
        mean = [stats[k][0] for k in ("iterations", "matvec_count", "kkt_residual",
                                      "relative_gap", "relative_l2", "support_f1",
                                      "amplitude_rmse")]
        std = [stats[k][1] for k in ("iterations", "matvec_count", "kkt_residual",
                                     "relative_gap", "relative_l2", "support_f1",
                                     "amplitude_rmse")]
        print(f"| {name} | mean | " + " | ".join(f"{v:.3e}" for v in mean) + " |")
        print(f"| {name} | std  | " + " | ".join(f"{v:.3e}" for v in std) + " |")

    # Convergence curve for one representative seed.
    _plot_convergence()

    # Write markdown report.
    report_path = OUT_DIR / "signed_lasso_report.md"
    _write_report(report_path, rows_by_solver)
    print(f"\nReport written to {report_path}")


def _plot_convergence():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A, y, x_true, lam, lipschitz = make_problem(0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, fn in (("fista-signed", solve_fista_signed), ("fista-nonneg", solve_fista)):
        res = fn(A, y, lam, lipschitz=lipschitz, certificate_stride=1,
                 kkt_tol=1e-5, gap_tol=1e-5, max_iter=20000)
        hist = res.history
        xs = [c.iteration for c in hist]
        kkts = [c.kkt_residual for c in hist]
        gaps = [c.relative_gap for c in hist]
        ax[0].semilogy(xs, kkts, label=name)
        ax[1].semilogy(xs, gaps, label=name)

    ax[0].set_title("KKT residual vs iteration (different objectives)")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel(r"$\varepsilon_{\mathrm{KKT}}$")
    ax[0].legend()
    ax[1].set_title("relative dual gap vs iteration (different objectives)")
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("relative gap")
    ax[1].legend()
    fig.tight_layout()
    out = OUT_DIR / "signed_lasso_convergence.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _write_report(path, rows_by_solver):
    lines = [
        "# T1 带符号 LASSO 基准报告",
        "",
        "合成数据：高斯设计 `A ∈ R^{80×200}`，真值 `k=15` 稀疏（**含负权重**），",
        "观测 `y = A x_true + 0.05·noise`，`λ = 0.1·‖Aᵀy‖∞`。",
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
        "- `fista-signed` 收敛并达到证书容差，支撑 F1 均值达标，且能恢复**负权重**支撑。",
        "- `fista-nonneg`（非负版对照）在真值含负权重时支撑 F1 显著下降——每个 seed 上 signed 的 F1 均严格高于 nonneg，这正是带符号版存在的意义。",
        "",
        "![收敛曲线](signed_lasso_convergence.png)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
