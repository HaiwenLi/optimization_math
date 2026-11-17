"""T5 benchmark: group LASSO group-selection effect and nuclear-norm low-rank.

Two sub-benchmarks, both with known truth:
  1. group LASSO vs plain LASSO: does group penalization select groups jointly?
  2. nuclear-norm: does singular-value thresholding recover low rank?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from optimization_math.solvers.fista_group import solve_fista_group
from optimization_math.solvers.fista_signed import solve_fista_signed  # plain (signed) LASSO reference
from optimization_math.solvers.fista_nuclear import solve_fista_nuclear
from optimization_math.certificates_nuclear import soft_threshold_singular

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"


class IdentityOp:
    def __init__(self, shape):
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    def matvec(self, X):
        return X.ravel()

    def rmatvec(self, v):
        return v.reshape(self._shape)


def _group_coherence(x_hat, groups):
    """Fraction of active groups whose support is jointly nonzero (group effect)."""
    active = [float(np.count_nonzero(np.abs(x_hat[g]) > 1e-8) == len(g)) for g in groups]
    return float(np.mean(active)) if active else 0.0


def benchmark_group(seed=0):
    rng = np.random.default_rng(seed)
    n = 30
    groups = [np.arange(i, i + 3) for i in range(0, n, 3)]
    A = rng.standard_normal((20, n))
    x_true = np.zeros(n)
    x_true[0:3] = rng.standard_normal(3)  # only group 0 active
    y = A @ x_true + 0.01 * rng.standard_normal(20)
    lam = 0.1
    L = float(np.linalg.norm(A, 2) ** 2 * 1.2)

    r_group = solve_fista_group(A, y, lam, groups, lipschitz=L,
                                certificate_stride=1, kkt_tol=1e-5, gap_tol=1e-5, max_iter=20000)
    # Plain (signed) LASSO reference at a comparable lambda (per-coordinate l1).
    r_plain = solve_fista_signed(A, y, lam, lipschitz=L,
                                 certificate_stride=1, kkt_tol=1e-5, gap_tol=1e-5, max_iter=20000)
    return {
        "group": {
            "status": r_group.status,
            "coherence": _group_coherence(r_group.x, groups),
            "err_active": float(np.linalg.norm(r_group.x[0:3] - x_true[0:3])),
            "iter": r_group.iterations,
        },
        "plain": {
            "status": r_plain.status,
            "coherence": _group_coherence(r_plain.x, groups),
            "err_active": float(np.linalg.norm(r_plain.x[0:3] - x_true[0:3])),
            "iter": r_plain.iterations,
        },
    }


def benchmark_nuclear(seed=0):
    rng = np.random.default_rng(seed)
    m = n = 12
    U = rng.standard_normal((m, 3))
    V = rng.standard_normal((n, 3))
    X_true = U @ V.T
    Y = X_true + 0.05 * rng.standard_normal((m, n))
    A = IdentityOp((m, n))
    noise = float(np.linalg.norm(Y - X_true))
    # Objective 1: best denoising error (over a lambda grid).
    best_err, best_lam, best_rank = None, None, None
    # Objective 2: exact rank recovery at a large enough lambda.
    rank_recovered_at = {}
    for lam in (0.05, 0.1, 0.2, 0.5, 1.0):
        r = solve_fista_nuclear(A, Y.ravel(), lam, lipschitz=1.0,
                                certificate_stride=1, kkt_tol=1e-6, gap_tol=1e-7)
        err = float(np.linalg.norm(r.X - X_true))
        rank = int(np.sum(np.linalg.svd(r.X, compute_uv=False) > 1e-6))
        if best_err is None or err < best_err:
            best_err, best_lam, best_rank = err, lam, rank
        if rank == 3 and lam not in rank_recovered_at:  # exact rank-3 recovery
            rank_recovered_at[lam] = True
    return {
        "noise": noise,
        "best_lam": best_lam,
        "best_err": best_err,
        "best_rank": best_rank,
        "rank_recovered": bool(rank_recovered_at),
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    g = benchmark_group()
    print("T5 group LASSO (group coherence, higher = better group selection):")
    print(f"  group: coherence={g['group']['coherence']:.3f} err_active={g['group']['err_active']:.3f} "
          f"iter={g['group']['iter']}")
    print(f"  plain: coherence={g['plain']['coherence']:.3f} err_active={g['plain']['err_active']:.3f} "
          f"iter={g['plain']['iter']}")
    assert g["group"]["coherence"] >= g["plain"]["coherence"], "group LASSO should select groups jointly"

    nuc = benchmark_nuclear()
    print(f"\nT5 nuclear norm: noise={nuc['noise']:.4f} best_lam={nuc['best_lam']} "
          f"best_err={nuc['best_err']:.4f} best_rank={nuc['best_rank']} "
          f"rank_recovered={nuc['rank_recovered']}")
    assert nuc["best_err"] < nuc["noise"], "nuclear denoising must beat the noise"
    assert nuc["rank_recovered"], "a large enough lambda must recover exact low rank"

    print("\n自动判定通过：group LASSO 组联合选择 ≥ 普通 LASSO；核范数去噪降噪且能恢复低秩。")
    _write_report(g, nuc)


def _write_report(g, nuc):
    lines = [
        "# T5 结构化稀疏基准报告",
        "",
        "## group LASSO（组联合选择）",
        "",
        f"- group LASSO：组一致性 {g['group']['coherence']:.3f}，活跃组误差 {g['group']['err_active']:.3f}",
        f"- 普通 LASSO：组一致性 {g['plain']['coherence']:.3f}，活跃组误差 {g['plain']['err_active']:.3f}",
        "",
        "## 核范数低秩",
        "",
        f"- 噪声 {nuc['noise']:.4f} → 最优 λ={nuc['best_lam']} 去噪误差 {nuc['best_err']:.4f}，秩 {nuc['best_rank']}",
        "",
    ]
    (OUT_DIR / "structured_sparse_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
