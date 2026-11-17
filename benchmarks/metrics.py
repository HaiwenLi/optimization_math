"""Shared evaluation metrics for benchmarking solvers against a known ground truth.

All functions are pure NumPy and independent of any solver, so they can be
reused across every problem type (signed LASSO, logistic, TV, QP, structured
sparsity). The design principle of the benchmark suite: the ground truth is
*known* (synthetic data), so recovery quality is measured by exact comparison
rather than by a human's judgment.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RecoveryMetrics:
    """Recovery quality of a solution ``x_hat`` against a truth ``x_true``."""

    relative_l2: float
    support_precision: float
    support_recall: float
    support_f1: float
    amplitude_rmse: float


def relative_l2_error(x_hat, x_true, *, eps=1e-12):
    x_hat = np.asarray(x_hat, dtype=np.float64)
    x_true = np.asarray(x_true, dtype=np.float64)
    denom = max(np.linalg.norm(x_true), eps)
    return float(np.linalg.norm(x_hat - x_true) / denom)


def amplitude_rmse(x_hat, x_true):
    x_hat = np.asarray(x_hat, dtype=np.float64)
    x_true = np.asarray(x_true, dtype=np.float64)
    support = np.abs(x_true) > 0
    if not np.any(support):
        return float(np.linalg.norm(x_hat))
    k = int(np.count_nonzero(support))
    return float(np.linalg.norm(x_hat[support] - x_true[support]) / np.sqrt(k))


def support_metrics(x_hat, x_true, *, tol=1e-8):
    """Classify support recovery between predicted and true nonzero sets."""
    x_hat = np.asarray(x_hat, dtype=np.float64)
    x_true = np.asarray(x_true, dtype=np.float64)
    pred = np.abs(x_hat) > tol
    true = np.abs(x_true) > tol
    tp = int(np.count_nonzero(pred & true))
    fp = int(np.count_nonzero(pred & ~true))
    fn = int(np.count_nonzero(~pred & true))
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return float(precision), float(recall), float(f1)


def recovery_metrics(x_hat, x_true, *, tol=1e-8):
    """Bundle all recovery metrics into a single ``RecoveryMetrics``."""
    precision, recall, f1 = support_metrics(x_hat, x_true, tol=tol)
    return RecoveryMetrics(
        relative_l2=relative_l2_error(x_hat, x_true),
        support_precision=precision,
        support_recall=recall,
        support_f1=f1,
        amplitude_rmse=amplitude_rmse(x_hat, x_true),
    )


@dataclass(frozen=True)
class RunResult:
    """One solver run's quantitative outcome, ready for tabulation."""

    solver: str
    status: str
    iterations: int
    matvec_count: int
    kkt_residual: float
    relative_gap: float
    objective: float
    recovery: RecoveryMetrics


def format_row(result: RunResult) -> str:
    """Single-line markdown table row for a RunResult."""
    r = result.recovery
    return (
        f"| {result.solver} | {result.status} | {result.iterations} | "
        f"{result.matvec_count} | {result.kkt_residual:.2e} | {result.relative_gap:.2e} | "
        f"{result.objective:.6e} | {r.relative_l2:.2e} | {r.support_precision:.3f} | "
        f"{r.support_recall:.3f} | {r.support_f1:.3f} | {r.amplitude_rmse:.3e} |"
    )


MARKDOWN_HEADER = (
    "| solver | status | iters | A/Aᵀ calls | ε_KKT | rel.gap | objective | "
    "rel.L2 | supp.prec | supp.recall | supp.F1 | amp.RMSE |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def summarize(runs):
    """Mean ± std over repeated runs (stability/variance view).

    ``runs`` is an iterable of ``RunResult`` with the same solver name.
    Returns a dict of ``{metric: (mean, std)}`` for the scalar metrics.
    """
    if not runs:
        return {}
    its = np.array([r.iterations for r in runs], dtype=np.float64)
    mv = np.array([r.matvec_count for r in runs], dtype=np.float64)
    kkt = np.array([r.kkt_residual for r in runs], dtype=np.float64)
    gap = np.array([r.relative_gap for r in runs], dtype=np.float64)
    l2 = np.array([r.recovery.relative_l2 for r in runs], dtype=np.float64)
    f1 = np.array([r.recovery.support_f1 for r in runs], dtype=np.float64)
    rmse = np.array([r.recovery.amplitude_rmse for r in runs], dtype=np.float64)

    def pair(a):
        return (float(np.mean(a)), float(np.std(a)))

    return {
        "iterations": pair(its),
        "matvec_count": pair(mv),
        "kkt_residual": pair(kkt),
        "relative_gap": pair(gap),
        "relative_l2": pair(l2),
        "support_f1": pair(f1),
        "amplitude_rmse": pair(rmse),
    }


def format_summary(name, stats):
    """Render a ``summarize`` dict as two table rows (mean / std)."""
    mean_vals = [stats[k][0] for k in stats]
    std_vals = [stats[k][1] for k in stats]
    return (
        f"| {name} | mean | " + " | ".join(f"{v:.3e}" for v in mean_vals) + " |\n"
        f"| {name} | std  | " + " | ".join(f"{v:.3e}" for v in std_vals) + " |"
    )
