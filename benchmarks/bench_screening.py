"""GAP-safe screening benchmark: quantify screening safety and effectiveness.

Answers two questions objectively at the converged iterate (where the dual gap
is smallest and the safe-screen radius is tightest):
  1. SAFETY -- the screened-out coordinates never include an optimum-nonzero
     coordinate (the hard guarantee).
  2. EFFECTIVENESS -- how many coordinates are provably pruned.

Note: the screened fraction is *measured at the converged iterate only*. The
separate "radius vs gap" curve in the report is illustrative (it sweeps the gap
analytically), not a per-iteration measurement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from optimization_math.solvers.fista_signed import solve_fista_signed
from optimization_math.screening import compute_safe_screen

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


def benchmark_screening(seeds=(0, 1, 2, 3, 4)):
    """Record final gap, screened fraction and optimum-support retention."""
    rows = []
    for seed in seeds:
        A, y, x_true, lam, lipschitz = make_problem(seed)
        res = solve_fista_signed(A, y, lam, lipschitz=lipschitz,
                                 kkt_tol=1e-8, gap_tol=1e-8,
                                 certificate_stride=1, max_iter=20000)
        opt_support = np.abs(res.x) > 1e-8
        # Safety + effectiveness measured at the converged iterate, where the
        # gap is small and the screening radius is tightest. NOTE: the safety
        # contract is against the OPTIMUM's support (not the generating truth's
        # support) -- screening guarantees a screened coordinate is zero at the
        # optimum, and LASSO legitimately zeroes weak/noise-dominated truth
        # coordinates that are nonzero in the generating ``x_true``.
        safe_mask = compute_safe_screen(A, y, res.x, lam)
        screened_frac = float(np.mean(safe_mask))
        # Safety: never remove a coordinate that is nonzero at the optimum.
        never_removes_opt = not np.any(safe_mask & opt_support)
        rows.append({
            "seed": seed,
            "status": res.status,
            "iterations": res.iterations,
            "final_gap": res.certificate.relative_gap,
            "screened_frac": screened_frac,
            "n_screened": int(np.sum(safe_mask)),
            "opt_support_kept": never_removes_opt,
        })
    return rows


def _assert_benchmark(rows):
    assert all(r["status"] == "converged" for r in rows), "FISTA must converge"
    # Safety is the hard guarantee: screening must never drop a coordinate that
    # is nonzero at the optimum.
    assert all(r["opt_support_kept"] for r in rows), "screening removed an optimal-support coordinate"
    # Effectiveness: on these sparse problems (k=15 of n=200) at convergence the
    # gap is tiny so screening should prune a large fraction of coordinates.
    fracs = [r["screened_frac"] for r in rows]
    assert np.mean(fracs) >= 0.5, f"screening too weak: mean screened fraction {np.mean(fracs):.3f}"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = benchmark_screening()
    _assert_benchmark(rows)

    print("GAP-safe screening benchmark:")
    print("| seed | status | iters | final rel.gap | screened frac | opt support kept |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['seed']} | {r['status']} | {r['iterations']} | {r['final_gap']:.2e} "
              f"| {r['screened_frac']:.3f} | {r['opt_support_kept']} |")

    _plot_screening_radius()

    report_path = OUT_DIR / "screening_report.md"
    _write_report(report_path, rows)
    print(f"\nReport written to {report_path}")


def _plot_screening_radius():
    """Show how the safe-screen radius (hence screened fraction) tightens as
    the gap shrinks, on one representative problem."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A, y, x_true, lam, lipschitz = make_problem(0)
    res = solve_fista_signed(A, y, lam, lipschitz=lipschitz,
                             kkt_tol=1e-8, gap_tol=1e-8,
                             certificate_stride=1, max_iter=20000)
    # Recompute the screen at a grid of *artificial* gap values to show the
    # monotone tightening: smaller gap -> smaller radius -> more screening.
    # Evaluate the dual point at the optimum; vary the gap to move the radius.
    from optimization_math.certificates_signed import compute_certificate_signed
    cert = compute_certificate_signed(A, y, res.x, lam)
    # Inflate the true absolute gap; smaller values need not be safe bounds
    # for this fixed dual point. Use the production rule, not a second formula.
    minimum_gap = max(cert.primal - cert.dual, 1e-8)
    gaps = np.geomspace(minimum_gap, max(1.0, 100.0 * minimum_gap), 40)
    fracs = [float(np.mean(compute_safe_screen(A, y, res.x, lam, gap=g)))
             for g in gaps]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(gaps, fracs, marker="o")
    ax.set_xlabel("absolute gap upper bound (illustrative sweep)")
    ax.set_ylabel("safely screened fraction")
    ax.set_title("GAP-safe screening effectiveness vs gap")
    fig.tight_layout()
    out = OUT_DIR / "screening_radius.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _write_report(path, rows):
    lines = [
        "# GAP-safe screening 基准报告",
        "",
        "合成稀疏问题（`n=200, k=15`）上，用 FISTA 收敛后得到的对偶可行点",
        "计算 GAP-safe 筛选半径，统计可被**安全剔除**（保证在最优解为 0）的坐标比例。",
        "",
        "**安全口径**：筛选保证的是「被剔除坐标在 LASSO **最优解**中为 0」——",
        "而非生成真值 `x_true` 中为 0（LASSO 会正确地把被噪声淹没的弱真值坐标压缩为 0）。",
        "",
        "| seed | status | iters | final rel.gap | screened frac | opt support kept |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['status']} | {r['iterations']} | {r['final_gap']:.2e} "
                     f"| {r['screened_frac']:.3f} | {r['opt_support_kept']} |")
    lines += [
        "",
        "## 结论",
        "",
        "- **安全性**：所有 seed 上，被筛选的坐标都不含任何**最优解非零**分量（hard 保证）。",
        "- **有效性**：收敛时 gap 极小时可安全剔除大部分坐标，降维显著。",
        "",
        "![筛选半径-筛选比例曲线](screening_radius.png)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
