"""The report renders real benchmark results and preserves failed verdicts."""
from dataclasses import replace
import json

import pytest

from benchmarks.bench_standard import benchmark_standard
from benchmarks.standard_report import write_standard_report
from benchmarks.standard_report import _plot_accuracy, _plot_convergence


@pytest.fixture(scope="module")
def rows():
    return benchmark_standard(seeds=(0,), conditions=(1.,))


def test_report_contains_results_metadata_and_real_pngs(rows, tmp_path):
    pytest.importorskip("matplotlib")
    paths = write_standard_report(rows, tmp_path)
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["summary"]["passed"] == len(rows)
    assert data["summary"]["failed"] == 0
    assert len(data["runs"]) == len(rows)
    assert data["environment"]["numpy"]
    assert data["environment"]["python"]
    compact = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert len(compact["runs"]) == len(rows)
    assert all("history" not in run for run in compact["runs"])
    assert compact["runs"][0]["input_digest"] == rows[0].input_digest
    text = paths["markdown"].read_text(encoding="utf-8")
    assert "generating" in text
    assert "forward" in text and "adjoint" in text
    for image in paths["images"]:
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert image.stat().st_size > 1000
        assert image.relative_to(tmp_path).as_posix() in text


def test_report_never_labels_failed_run_as_passed(rows, tmp_path):
    pytest.importorskip("matplotlib")
    bad = replace(rows[0], status="max_iter")
    paths = write_standard_report([bad], tmp_path)
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["summary"] == {"total": 1, "passed": 0, "failed": 1}
    assert len(data["failures"]) == 1
    assert bad.case in data["failures"][0]
    assert "0/1" in paths["markdown"].read_text(encoding="utf-8")


def test_report_preserves_nonfinite_failure_as_json_null(rows, tmp_path):
    pytest.importorskip("matplotlib")
    paths = write_standard_report([replace(rows[0], objective=float("nan"))], tmp_path)
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["summary"]["failed"] == 1
    assert data["runs"][0]["objective"] is None
    assert "nonfinite" in data["failures"][0]


@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_report_completes_for_nonfinite_errors(rows, tmp_path, value):
    pytest.importorskip("matplotlib")
    (tmp_path / "README.md").write_text("stale success", encoding="utf-8")
    bad = replace(rows[0], status="max_iter",
                  normalized_objective_error=value, solution_error=value)
    paths = write_standard_report([bad], tmp_path)
    for key in ("json", "metrics"):
        data = json.loads(paths[key].read_text(encoding="utf-8"))
        assert data["summary"] == {"total": 1, "passed": 0, "failed": 1}
        assert data["runs"][0]["normalized_objective_error"] is None
        assert data["runs"][0]["solution_error"] is None
        assert data["runs"][0]["status"] == "max_iter"
    text = paths["markdown"].read_text(encoding="utf-8")
    assert "stale success" not in text
    assert "0/1 comparisons passed" in text
    table_row = next(line for line in text.splitlines()
                     if line.startswith(f"| {bad.problem} | {bad.algorithm} | 1 |"))
    assert table_row.count("n/a (nonfinite)") == 2
    for image in paths["images"]:
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert image.stat().st_size > 1000


@pytest.fixture
def plot_figures(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    close = plt.close
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)
    yield plt
    close("all")


@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=["nan", "inf"])
@pytest.mark.parametrize("mixed", [False, True], ids=["all-invalid", "mixed"])
def test_accuracy_marks_nonfinite_without_perfect_error_points(
        rows, tmp_path, plot_figures, value, mixed):
    bad = replace(rows[0], normalized_objective_error=value, solution_error=value)
    selected = [bad, rows[0]] if mixed else [bad]
    _plot_accuracy(selected, tmp_path / "accuracy.png", plot_figures)
    ax = plot_figures.gcf().axes[0]
    assert f"{bad.problem} / {bad.algorithm}" in [
        label.get_text() for label in ax.get_yticklabels()]
    assert any("nonfinite" in text.get_text() for text in ax.texts)
    offsets = [point for collection in ax.collections for point in collection.get_offsets()]
    if mixed:
        assert sorted(float(point[0]) for point in offsets) == sorted([
            max(rows[0].normalized_objective_error, 1e-16),
            max(rows[0].solution_error, 1e-16)])
    else:
        assert not offsets
        assert any("no finite data" in text.get_text() for text in ax.texts)


@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=["nan", "inf"])
@pytest.mark.parametrize("history_kind", ["invalid", "mixed", "empty"])
def test_convergence_filters_invalid_history_and_marks_unavailable(
        rows, tmp_path, plot_figures, value, history_kind):
    row = rows[0]
    invalid = replace(row.history[0], objective=value)
    history = {"invalid": (invalid,), "mixed": (invalid, row.history[-1]),
               "empty": ()}[history_kind]
    _plot_convergence([replace(row, history=history)],
                      tmp_path / "convergence.png", plot_figures)
    # Panels are created in the fixed order the production function uses.
    panel = {"nonnegative_lasso": 0, "signed_lasso": 1,
             "box_qp": 2, "logistic": 3}[row.problem]
    ax = plot_figures.gcf().axes[panel]
    assert any("nonfinite" in text.get_text() or "no finite data" in text.get_text()
               for text in ax.texts)
    points = [value for line in ax.lines for value in line.get_ydata()]
    if history_kind == "mixed":
        assert points == [max(abs(row.history[-1].objective - row.reference_objective)
                              / max(1., abs(row.reference_objective)), 1e-16)]
    else:
        assert not points


def test_report_rejects_empty_results(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        write_standard_report([], tmp_path)
