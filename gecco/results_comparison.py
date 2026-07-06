"""Lightweight results comparison for GeCCo run directories.

This module provides discovery, DuckDB-based summarisation, CSV export,
static HTML report generation, and PNG/PDF figure export for comparing
output from multiple GeCCo run directories.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import duckdb

# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

_DIAGNOSTICS_PATTERNS = (
    "diagnostics_unified.duckdb",
    "diagnostics.duckdb",
    # "diagnostics_*.duckdb" is handled via glob
)


def discover_run_dirs(results_roots: list[str | Path]) -> list[dict[str, Any]]:
    """Recursively discover run directories below *results_roots*.

    Each discovered entry is a dict with keys:
        ``input_root``    – the parent results root that contained this run
        ``config_label``  – a short label derived from the directory tree
        ``results_dir``   – absolute path to the run directory
        ``db_path``       – path to the discovered DuckDB file
        ``run_id``        – the leaf directory name

    Parameters
    ----------
    results_roots:
        One or more top-level directories to search recursively.

    Returns
    -------
    list[dict]
        Discovered run entries, each containing ``input_root``, ``config_label``,
        ``results_dir``, ``db_path``, and ``run_id``.
    """
    discovered: list[dict[str, Any]] = []

    for root in results_roots:
        root_path = Path(root).resolve()
        if not root_path.is_dir():
            continue

        # Walk recursively looking for diagnostics DuckDB files
        for db_path in _find_duckdb_files(root_path):
            run_dir = db_path.parent
            # Derive a config_label from the relative path under input_root
            try:
                rel = run_dir.relative_to(root_path)
            except ValueError:
                rel = Path(run_dir.name)
            config_label = str(rel.parent) if rel.parent != Path(".") else run_dir.name

            discovered.append(
                {
                    "input_root": str(root_path),
                    "config_label": config_label,
                    "results_dir": str(run_dir),
                    "db_path": str(db_path),
                    "run_id": run_dir.name,
                }
            )

    return discovered


def _find_duckdb_files(root: Path) -> list[Path]:
    """Return all DuckDB diagnostic files under *root*, one per parent directory.

    Matches ``diagnostics.duckdb``, ``diagnostics_unified.duckdb``,
    and ``diagnostics_*.duckdb``.  When multiple matches exist in the same
    parent directory (e.g. ``diagnostics.duckdb`` and ``diagnostics_unified.duckdb``),
    only the first one found is kept.
    """
    seen_parents: set[Path] = set()
    results: list[Path] = []
    for pattern in _DIAGNOSTICS_PATTERNS + ("diagnostics_*.duckdb",):
        for path in root.rglob(pattern):
            parent = path.parent
            if parent not in seen_parents:
                seen_parents.add(parent)
                results.append(path)
    return results


# --------------------------------------------------------------------------- #
# Summarisation
# --------------------------------------------------------------------------- #

_COLUMN_ORDER = [
    "input_root",
    "config_label",
    "results_dir",
    "run_id",
    "best_train_metric",
    "best_val_metric",
    "best_test_metric",
    "best_test_nll",
    "best_test_mean_r2",
    "best_test_max_r2",
    "best_test_param",
    "best_model_name",
    "n_models",
    "n_failed_models",
    "has_test_eval",
    "has_individual_differences",
]


def summarise_run(entry: dict) -> dict[str, Any]:
    """Query a single run's DuckDB and return a summary dict.

    Parameters
    ----------
    entry:
        A discovery entry dict (as returned by :func:`discover_run_dirs`)
        with at least ``db_path``.

    Returns
    -------
    dict
        A one-row-per-run summary with all keys listed in ``_COLUMN_ORDER``.
    """
    db_path = entry["db_path"]
    result: dict[str, Any] = {
        "input_root": entry["input_root"],
        "config_label": entry["config_label"],
        "results_dir": entry["results_dir"],
        "run_id": entry["run_id"],
        "best_train_metric": None,
        "best_val_metric": None,
        "best_test_metric": None,
        "best_test_nll": None,
        "best_test_mean_r2": None,
        "best_test_max_r2": None,
        "best_test_param": None,
        "best_model_name": None,
        "n_models": 0,
        "n_failed_models": 0,
        "has_test_eval": False,
        "has_individual_differences": False,
    }

    try:
        conn = duckdb.connect(str(db_path), read_only=True)
    except Exception:
        return result

    try:
        # Check if tables exist
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        ]

        if "models" not in tables:
            return result

        # Count models and failed models
        count_row = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END) FROM models"
        ).fetchone()
        result["n_models"] = count_row[0] or 0
        result["n_failed_models"] = count_row[1] or 0

        # Best train model (lowest metric_value)
        train_row = conn.execute(
            "SELECT name, metric_value, mean_nll "
            "FROM models "
            "WHERE split='train' AND status='ok' AND metric_value IS NOT NULL "
            "ORDER BY metric_value ASC "
            "LIMIT 1"
        ).fetchone()
        if train_row:
            result["best_train_metric"] = float(train_row[1])
            result["best_model_name"] = train_row[0]

        # Best val model (lowest metric_value)
        val_row = conn.execute(
            "SELECT metric_value "
            "FROM models "
            "WHERE split='val' AND status='ok' AND metric_value IS NOT NULL "
            "ORDER BY metric_value ASC "
            "LIMIT 1"
        ).fetchone()
        if val_row:
            result["best_val_metric"] = float(val_row[0])

        # Best test model (lowest metric_value)
        test_row = conn.execute(
            "SELECT name, metric_value, mean_nll "
            "FROM models "
            "WHERE split='test' AND status='ok' AND metric_value IS NOT NULL "
            "ORDER BY metric_value ASC "
            "LIMIT 1"
        ).fetchone()
        if test_row:
            result["best_test_metric"] = float(test_row[1])
            result["best_test_nll"] = (
                float(test_row[2]) if test_row[2] is not None else None
            )
            result["has_test_eval"] = True

            # Look up individual differences for the best test model
            if "individual_differences" in tables:
                test_model_name = test_row[0]
                id_row = conn.execute(
                    "SELECT id.mean_r2, id.max_r2, id.best_param "
                    "FROM individual_differences id "
                    "JOIN models m ON m.model_id = id.model_id "
                    "WHERE m.name=? AND m.split='test' AND id.split='test'",
                    [test_model_name],
                ).fetchone()
                if id_row:
                    result["best_test_mean_r2"] = (
                        float(id_row[0]) if id_row[0] is not None else None
                    )
                    result["best_test_max_r2"] = (
                        float(id_row[1]) if id_row[1] is not None else None
                    )
                    result["best_test_param"] = id_row[2]
                    result["has_individual_differences"] = True
    finally:
        conn.close()

    return result


# --------------------------------------------------------------------------- #
# CSV export
# --------------------------------------------------------------------------- #


def write_results_csv(
    summaries: list[dict[str, Any]], output_dir: str | Path
) -> Path:
    """Write a ``results.csv`` with one row per run summary.

    Parameters
    ----------
    summaries:
        List of run summary dicts (as returned by :func:`summarise_run`).
    output_dir:
        Directory to write ``results.csv`` into.

    Returns
    -------
    Path
        Absolute path to the written CSV file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMN_ORDER, extrasaction="ignore")
        writer.writeheader()
        for summary in summaries:
            row = {k: summary.get(k) for k in _COLUMN_ORDER}
            writer.writerow(row)

    return csv_path.resolve()


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GeCCo Results Comparison</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 1em; color: #333; }}
  h1, h2, h3 {{ color: #1a1a2e; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.9em; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background-color: #f5f5f5; font-weight: 600; }}
  tr:nth-child(even) {{ background-color: #fafafa; }}
  .missing {{ color: #999; font-style: italic; }}
  .stat {{ font-weight: 600; }}
  .fig-container {{ display: flex; flex-wrap: wrap; gap: 1em; margin: 1em 0; }}
  .fig-container img {{ max-width: 100%; border: 1px solid #eee; border-radius: 4px; }}
  .note {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 0.75em 1em; margin: 1em 0; }}
</style>
</head>
<body>
<h1>GeCCo Results Comparison</h1>
<p>Generated from <strong>{n_runs}</strong> run director{plural}.</p>

{missing_data_notes}

<h2>Run Overview</h2>
<table>
<thead>
<tr>
  <th>Config</th>
  <th>Run ID</th>
  <th>Best Model</th>
  <th>Train Metric</th>
  <th>Val Metric</th>
  <th>Test Metric</th>
  <th>Test NLL</th>
  <th>ID Mean R²</th>
  <th>ID Max R²</th>
  <th>ID Best Param</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>

<h2>Figures</h2>
<div class="fig-container">
  <div><h3>Model Fit by Config</h3><img src="figures/model_fit_by_config.png" alt="Model fit by config"></div>
  <div><h3>Individual Differences by Config</h3><img src="figures/individual_differences_by_config.png" alt="Individual differences by config"></div>
  <div><h3>Fit vs Prediction</h3><img src="figures/fit_vs_prediction.png" alt="Fit vs prediction"></div>
</div>

<p><em>Report generated by GeCCo results-comparison tool.</em></p>
</body>
</html>
"""


def _format_val(value: Any) -> str:
    """Format a value for HTML display, showing '—' for None/empty."""
    if value is None or value == "" or (isinstance(value, float) and value != value):
        return '<span class="missing">—</span>'
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_report_html(
    summaries: list[dict[str, Any]], output_dir: str | Path
) -> Path:
    """Render a self-contained static HTML report from the run summaries.

    Parameters
    ----------
    summaries:
        List of run summary dicts.
    output_dir:
        Directory to write ``report.html`` into.

    Returns
    -------
    Path
        Absolute path to the written HTML file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build table rows
    rows_html = ""
    missing_notes = []

    n_no_test = sum(1 for s in summaries if not s.get("has_test_eval"))
    n_no_id = sum(1 for s in summaries if not s.get("has_individual_differences"))

    if n_no_test:
        missing_notes.append(
            f"<div class=\"note\"><strong>Note:</strong> {n_no_test} run(s) "
            "have no test evaluation data.</div>"
        )
    if n_no_id:
        missing_notes.append(
            f"<div class=\"note\"><strong>Note:</strong> {n_no_id} run(s) "
            "have no individual differences data.</div>"
        )

    for s in summaries:
        rows_html += (
            "<tr>"
            f"<td>{_format_val(s.get('config_label'))}</td>"
            f"<td>{_format_val(s.get('run_id'))}</td>"
            f"<td>{_format_val(s.get('best_model_name'))}</td>"
            f"<td>{_format_val(s.get('best_train_metric'))}</td>"
            f"<td>{_format_val(s.get('best_val_metric'))}</td>"
            f"<td>{_format_val(s.get('best_test_metric'))}</td>"
            f"<td>{_format_val(s.get('best_test_nll'))}</td>"
            f"<td>{_format_val(s.get('best_test_mean_r2'))}</td>"
            f"<td>{_format_val(s.get('best_test_max_r2'))}</td>"
            f"<td>{_format_val(s.get('best_test_param'))}</td>"
            "</tr>\n"
        )

    n_runs = len(summaries)
    plural = "y" if n_runs == 1 else "ies"

    html = _HTML_TEMPLATE.format(
        n_runs=n_runs,
        plural=plural,
        missing_data_notes="\n".join(missing_notes),
        rows_html=rows_html,
    )

    html_path = out / "report.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path.resolve()


# --------------------------------------------------------------------------- #
# Figure export
# --------------------------------------------------------------------------- #


def export_figures(
    summaries: list[dict[str, Any]], output_dir: str | Path
) -> Path:
    """Export comparison figures as PNG and PDF under ``output_dir/figures/``.

    Three figures are generated:

    1. ``model_fit_by_config`` — bar chart of best train/val/test metrics per run.
    2. ``individual_differences_by_config`` — bar chart of ID R² values per run.
    3. ``fit_vs_prediction`` — scatter of train vs test metrics when both exist.

    Parameters
    ----------
    summaries:
        List of run summary dicts.
    output_dir:
        Directory under which a ``figures/`` sub-directory is created.

    Returns
    -------
    Path
        Absolute path to the ``figures/`` directory.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    labels = [s.get("config_label", s.get("run_id", f"run_{i}")) for i, s in enumerate(summaries)]

    # -- 1. Model fit by config --
    train_vals = [s.get("best_train_metric") for s in summaries]
    val_vals = [s.get("best_val_metric") for s in summaries]
    test_vals = [s.get("best_test_metric") for s in summaries]

    _bar_chart(
        fig_dir / "model_fit_by_config",
        labels,
        [
            ("Train", train_vals),
            ("Val", val_vals),
            ("Test", test_vals),
        ],
        "Model Fit by Config",
        "Metric Value",
        "Best Fit Metric (lower is better)",
    )

    # -- 2. Individual differences by config --
    id_mean = [s.get("best_test_mean_r2") for s in summaries]
    id_max = [s.get("best_test_max_r2") for s in summaries]

    _bar_chart(
        fig_dir / "individual_differences_by_config",
        labels,
        [
            ("Mean R²", id_mean),
            ("Max R²", id_max),
        ],
        "Individual Differences by Config",
        "R²",
        "Test Individual Differences (higher is better)",
    )

    # -- 3. Fit vs Prediction (train vs test scatter) --
    _scatter_plot(
        fig_dir / "fit_vs_prediction",
        labels,
        train_vals,
        test_vals,
        "Train Metric vs Test Metric",
        "Best Train Metric",
        "Best Test Metric",
    )

    plt.close("all")
    return fig_dir.resolve()


def _bar_chart(
    base_path: Path,
    labels: list[str],
    series: list[tuple[str, list[float | None]]],
    title: str,
    ylabel: str,
    caption: str,
) -> None:
    """Draw a grouped bar chart and save as PNG + PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_groups = len(labels)
    n_series = len(series)

    if n_groups == 0:
        return

    fig, ax = plt.subplots(figsize=(max(6, n_groups * 0.8), 4))
    index = np.arange(n_groups)
    bar_width = 0.8 / n_series

    for i, (name, values) in enumerate(series):
        offset = (i - (n_series - 1) / 2) * bar_width
        clean = [v if v is not None else 0 for v in values]
        bars = ax.bar(index + offset, clean, bar_width, label=name, alpha=0.8)
        _label_bars(bars, values)

    ax.set_xlabel("Config")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.text(
        0.5, -0.25, caption, transform=ax.transAxes,
        ha="center", fontsize=8, color="gray", style="italic",
    )
    fig.tight_layout()
    fig.savefig(str(base_path) + ".png", dpi=150)
    fig.savefig(str(base_path) + ".pdf")
    plt.close(fig)


def _scatter_plot(
    base_path: Path,
    labels: list[str],
    x_vals: list[float | None],
    y_vals: list[float | None],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Draw a scatter plot and save as PNG + PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    valid = [(l, x, y) for l, x, y in zip(labels, x_vals, y_vals) if x is not None and y is not None]
    if len(valid) < 2:
        # Not enough points; draw a placeholder
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "Insufficient data for scatter plot",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(str(base_path) + ".png", dpi=150)
        fig.savefig(str(base_path) + ".pdf")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    l, x, y = zip(*valid)
    ax.scatter(x, y, alpha=0.7)

    for li, xi, yi in zip(l, x, y):
        ax.annotate(li, (xi, yi), fontsize=7, alpha=0.8)

    # Diagonal reference line
    all_vals = [v for v in x + y if v is not None]
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        margin = (hi - lo) * 0.1 if hi > lo else 1.0
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "--", color="gray", alpha=0.5, linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(base_path) + ".png", dpi=150)
    fig.savefig(str(base_path) + ".pdf")
    plt.close(fig)


def _label_bars(bars, values: list[float | None]) -> None:
    """Add text labels above bars when value is not None."""
    for bar, val in zip(bars, values):
        if val is not None:
            height = bar.get_height()
            ax = bar.axes
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.2f}" if abs(val) < 1000 else f"{val:.1e}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=6,
            )
