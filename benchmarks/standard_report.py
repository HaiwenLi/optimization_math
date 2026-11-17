"""Portable evidence artifacts for the independent-optimum benchmark matrix."""
from collections import defaultdict
import json
from pathlib import Path
import platform

import numpy as np

from benchmarks.bench_standard import (
    CERT_TOL, OBJECTIVE_TOL, SOLUTION_TOL, _assert_benchmark,
)

COLORS = {
    "fista": "#236b9b", "fista_signed": "#236b9b", "fista_logistic": "#236b9b",
    "qp_box": "#236b9b", "admm": "#2c8a59", "pdhg": "#bf641f",
    "scipy_lbfgsb": "#76508e",
}


def _groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row.problem, row.algorithm)].append(row)
    return dict(sorted(groups.items()))


def _json_safe(value):
    """Keep failed nonfinite measurements portable without hiding the verdict."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _summary_cell(values, aggregate):
    """Render measured values; nonfinite entries stay explicit, never fake zeros."""
    finite = [v for v in values if np.isfinite(v)]
    if len(finite) == len(values):
        return f"{aggregate(values):.3e}"
    if finite:
        return f"{aggregate(finite):.3e} + n/a (nonfinite)"
    return "n/a (nonfinite)"


def _plot_accuracy(rows, path, plt):
    groups = _groups(rows)
    panels = [
        [key for key in groups if key[0] in ("nonnegative_lasso", "signed_lasso",
                                            "box_qp", "nnls")],
        [key for key in groups if key[0] not in ("nonnegative_lasso", "signed_lasso",
                                                "box_qp", "nnls")],
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), layout="constrained")
    for ax, keys in zip(axes, panels):
        if not keys:
            ax.set_axis_off()
            continue
        y = np.arange(len(keys))
        plotted = False
        nonfinite = []
        for field, color, marker, label in (
            ("normalized_objective_error", "#236b9b", "o", "max objective error"),
            ("solution_error", "#bf641f", "s", "max solution error"),
        ):
            positions, values = [], []
            for index, key in zip(y, keys):
                field_values = [getattr(r, field) for r in groups[key]]
                finite = [v for v in field_values if np.isfinite(v)]
                if len(finite) != len(field_values):
                    nonfinite.append(key)
                if finite:
                    # Only finite measurements are drawn; nonfinite errors are
                    # never floored to 1e-16, which would look deceptively perfect.
                    positions.append(index)
                    values.append(max(finite))
            if values:
                ax.scatter(np.maximum(values, 1e-16), positions, color=color,
                           marker=marker, s=48, label=label)
                plotted = True
        ax.set_yticks(y, [f"{p} / {a}" for p, a in keys], fontsize=10)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.grid(axis="x", alpha=.2)
        ax.set_xlabel("Normalized error against independent optimum (floor 1e-16)")
        if plotted:
            ax.legend(loc="best", frameon=False)
        else:
            # Keep the log axis drawable with an explicit range instead of crashing.
            ax.set_xlim(1e-16, 1.)
            ax.text(.5, .5, "no finite data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="#666666")
        if nonfinite:
            names = sorted({f"{p} / {a}" for p, a in nonfinite})
            ax.text(.98, .04, "nonfinite error, not plotted: " + "; ".join(names),
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8, color="#8a3b3b")
    axes[0].set_title("Worst error across all seeds, conditioning levels and penalties",
                      fontsize=14, loc="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_convergence(rows, path, plt):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, problem in zip(axes.flat, ("nonnegative_lasso", "signed_lasso",
                                     "box_qp", "logistic")):
        candidates = [r for r in rows if r.problem == problem and r.mu == 0.]
        if not candidates:
            ax.set_axis_off()
            continue
        # The same case/digest is shared by every competing algorithm.
        exemplar = sorted(candidates, key=lambda r: (-r.conditioning, r.seed, r.lam))[0]
        plotted = False
        dropped_any = False
        for row in rows:
            if row.case != exemplar.case:
                continue
            points, dropped = [], 0
            for point in row.history:
                error = abs(point.objective - row.reference_objective) / \
                    max(1., abs(row.reference_objective))
                if np.isfinite(error):
                    points.append((point.iteration, max(error, 1e-16)))
                else:
                    dropped += 1
            dropped_any = dropped_any or dropped > 0
            if not points:
                continue
            ax.semilogy([p[0] for p in points], [p[1] for p in points],
                        label=row.algorithm, linewidth=1.7,
                        color=COLORS.get(row.algorithm))
            plotted = True
        ax.set_title(f"{problem}: seed {exemplar.seed}, scale {exemplar.conditioning:g}",
                     fontsize=12)
        ax.set_xlabel("Algorithm iteration (different work per iteration)")
        ax.set_ylabel("Normalized objective error (floor 1e-16)")
        ax.grid(alpha=.2)
        if plotted:
            ax.legend(frameon=False, fontsize=10)
        else:
            ax.set_ylim(1e-16, 1.)
            ax.text(.5, .5, "no finite data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="#666666")
        if dropped_any:
            ax.text(.98, .04, "nonfinite history points skipped",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8, color="#8a3b3b")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_calls(rows, path, plt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, problem in zip(axes, ("nonnegative_lasso", "signed_lasso")):
        selected = [r for r in rows if r.problem == problem and r.mu == 0.
                    and r.forward_calls is not None and r.adjoint_calls is not None]
        if not selected:
            ax.set_axis_off()
            continue
        for algorithm in sorted({r.algorithm for r in selected}):
            conditions = sorted({r.conditioning for r in selected})
            calls = [float(np.median([r.forward_calls + r.adjoint_calls
                                     for r in selected if r.algorithm == algorithm
                                     and r.conditioning == condition]))
                     for condition in conditions]
            ax.loglog(conditions, calls, "o-", label=algorithm, color=COLORS.get(algorithm))
        ax.set_title(f"{problem}: lambda=0.1, mu=0", fontsize=12)
        ax.set_xlabel("Unregularized Hessian condition target")
        ax.set_ylabel("Measured forward + adjoint calls (median over seeds)")
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_standard_report(rows, out_dir):
    """Write JSON, Markdown and PNGs; failed rows retain explicit failed verdicts."""
    if not rows:
        raise ValueError("cannot report an empty comparison matrix")
    import scipy
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    failures = []
    for row in rows:
        try:
            _assert_benchmark([row])
        except AssertionError as exc:
            failures.append(str(exc))
    summary = dict(total=len(rows), passed=len(rows)-len(failures), failed=len(failures))
    environment = dict(python=platform.python_version(), system=platform.system(),
                       machine=platform.machine(), numpy=np.__version__,
                       scipy=scipy.__version__, matplotlib=matplotlib.__version__)
    data = dict(schema_version=1, environment=environment, summary=summary,
                tolerances=dict(certificate=CERT_TOL, objective=OBJECTIVE_TOL,
                                solution=SOLUTION_TOL, exact_prox_solution=1e-9,
                                tv_solution=1e-5),
                seeds=sorted({r.seed for r in rows}),
                conditioning=sorted({r.conditioning for r in rows}),
                failures=failures, runs=[r.to_dict() for r in rows])
    json_path = out / "standard_results.json"
    json_path.write_text(json.dumps(_json_safe(data), indent=2, allow_nan=False), encoding="utf-8")
    compact = {key: value for key, value in data.items() if key != "runs"}
    compact["runs"] = [{key: value for key, value in run.items() if key != "history"}
                       for run in data["runs"]]
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(_json_safe(compact), indent=2, allow_nan=False),
                            encoding="utf-8")
    images = [assets / name for name in ("accuracy.png", "convergence.png", "operator_calls.png")]
    _plot_accuracy(rows, images[0], plt)
    _plot_convergence(rows, images[1], plt)
    _plot_calls(rows, images[2], plt)

    lines = [
        "# Standard optimization results",
        "",
        f"**{summary['passed']}/{summary['total']} comparisons passed**; "
        f"{summary['failed']} failed.",
        "",
        "References are independent analytic optima or manufactured KKT solutions, "
        "not another tested solver and not the generating signal.",
        "",
        "[Per-run measured metrics and configurations](metrics.json)",
        "",
        f"- Seeds: `{data['seeds']}`; conditioning targets: `{data['conditioning']}`.",
        "- Default fixture sizes: 8-variable vector problems, 4x4 nuclear "
        "denoising, and 2-point TV. This is a small correctness suite, not a "
        "large-scale performance benchmark.",
        f"- Certificate and normalized objective tolerances: `{CERT_TOL:g}`.",
        f"- Normalized solution tolerance: `{SOLUTION_TOL:g}`; "
        "identity group/nuclear cases `1e-9`; two-point TV `1e-5`.",
        "- Objective error denominator: `max(1, abs(f_star))`; solution error "
        "denominator: `max(1, norm(x_star))`.",
        "- Wall time includes solver certificate work and SciPy callback certificates, "
        "excludes fixture/reference construction and report generation. "
        "Tiny single-run timings are descriptive, not a statistically controlled speed ranking.",
        "- Operator counts are actual forward / adjoint invocations, including "
        "initialization, norm estimates, backtracking, CG and certificates. "
        "`n/a` means a dense-only API without operator counters, not zero work.",
        "- TV's stationarity field is an absolute primal-dual gap; nuclear uses "
        "a normalized Frobenius proximal fixed-point residual. Neither is a "
        "literal KKT norm. Logistic conditioning is a design scaling target, "
        "not its optimum Hessian condition.",
        "- Group/nuclear/TV rows have an exact reference but only one implemented "
        "algorithm; they do not establish comparative speed superiority.",
        "",
        "## Accuracy and measured cost",
        "",
        "| problem | algorithm | runs | max objective error | max solution error | "
        "median iters | median forward / adjoint | median ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (problem, algorithm), group in _groups(rows).items():
        calls = ("n/a" if group[0].forward_calls is None else
                 f"{np.median([r.forward_calls for r in group]):g} / "
                 f"{np.median([r.adjoint_calls for r in group]):g}")
        lines.append(f"| {problem} | {algorithm} | {len(group)} | "
                     f"{_summary_cell([r.normalized_objective_error for r in group], max)} | "
                     f"{_summary_cell([r.solution_error for r in group], max)} | "
                     f"{np.median([r.iterations for r in group]):g} | {calls} | "
                     f"{1000*np.median([r.time_sec for r in group]):.3f} |")
    lines += [
        "",
        "## Generating-signal recovery (separate from optimization error)",
        "",
        "| problem | algorithm | median recovery error |",
        "|---|---|---:|",
    ]
    for (problem, algorithm), group in _groups(rows).items():
        recovery = [r.recovery_error for r in group if r.recovery_error is not None]
        if recovery:
            lines.append(f"| {problem} | {algorithm} | "
                         f"{_summary_cell(recovery, np.median)} |")
    lines += [
        "",
        "## Figures",
        "",
        "![Worst errors against the independent optimum](assets/accuracy.png)",
        "",
        "![Same-instance objective convergence](assets/convergence.png)",
        "",
        "![Measured total operator calls](assets/operator_calls.png)",
        "",
        "## Environment and reproducibility",
        "",
        *(f"- {key}: `{value}`" for key, value in environment.items()),
        "",
        "From the repository root:",
        "",
        "```powershell",
        'python -m pip install -e ".[test,benchmark]"',
        "python -W error -m benchmarks.bench_standard --out-dir benchmarks/out/standard",
        "```",
        "",
        "The generated `standard_results.json` contains every result, input digest, "
        "history checkpoint and explicit optimizer configuration. Its call counts and "
        "timings should not be compared to legacy iteration-based cost estimates. "
        "The checked-in plots and summary are a snapshot of one machine run.",
    ]
    if failures:
        lines += ["", "## Failed checks", "", *(f"- {failure}" for failure in failures)]
    md_path = out / "README.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dict(json=json_path, metrics=metrics_path, markdown=md_path, images=tuple(images))
