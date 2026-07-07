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


def _count_excluded_test_rows(conn) -> tuple[int, list[str]]:
    """Count test-split rows excluded from best-metric selection.

    Only rows with ``status != 'ok'`` are excluded; successful test rows
    with ``code IS NULL`` are no longer treated as legacy summary-only rows
    and are counted as valid for test-evaluation statistics.
    """
    reasons = []
    # Rows with non-ok status on test split
    invalid_count = conn.execute(
        "SELECT COUNT(*) FROM models "
        "WHERE split='test' AND status != 'ok'"
    ).fetchone()[0] or 0
    if invalid_count:
        reasons.append(f"{invalid_count} invalid test row(s) excluded (status != 'ok')")
    return invalid_count, reasons


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
        "n_excluded_rows": 0,
        "exclusion_warnings": [],
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

        # Count excluded rows (legacy summary-only + invalid)
        n_excluded, exclusion_reasons = _count_excluded_test_rows(conn)
        result["n_excluded_rows"] = n_excluded
        result["exclusion_warnings"] = exclusion_reasons

        # Best train model (lowest metric_value) — valid-only
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

        # Best val model (lowest metric_value) — valid-only
        val_row = conn.execute(
            "SELECT metric_value "
            "FROM models "
            "WHERE split='val' AND status='ok' AND metric_value IS NOT NULL "
            "ORDER BY metric_value ASC "
            "LIMIT 1"
        ).fetchone()
        if val_row:
            result["best_val_metric"] = float(val_row[0])

        # Best test model (lowest metric_value) — valid-only, including code-null rows
        test_row = conn.execute(
            "SELECT name, metric_value, mean_nll, model_id "
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
                test_model_id = test_row[3]
                id_row = conn.execute(
                    "SELECT id.mean_r2, id.max_r2, id.best_param "
                    "FROM individual_differences id "
                    "WHERE id.model_id=? AND id.split='test'",
                    [test_model_id],
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

    # -- Sibling test-only fallback --
    # If the primary DB is diagnostics_unified.duckdb and has no valid test
    # rows, try a sibling diagnostics.duckdb in the same directory for
    # test/individual-differences data only.
    if (
        not result["has_test_eval"]
        and Path(db_path).name == "diagnostics_unified.duckdb"
    ):
        sibling = Path(db_path).parent / "diagnostics.duckdb"
        if sibling.exists():
            try:
                _merge_sibling_test_rows(result, str(sibling))
            except Exception as exc:
                result["exclusion_warnings"].append(
                    f"Sibling fallback to {sibling.name} failed: {exc}"
                )

    return result


def _merge_sibling_test_rows(
    result: dict[str, Any], sibling_db_path: str
) -> None:
    """Query a sibling ``diagnostics.duckdb`` for test/ID rows and merge
    into *result* when the primary DB had none.

    Only the best-test-metric, test-NLL, individual-differences, and
    test-evaluation booleans are overwritten; train/val/metrics and model
    counts remain as set by the primary DB.
    """
    conn = duckdb.connect(sibling_db_path, read_only=True)
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        ]
        if "models" not in tables:
            return

        # Best test model (lowest metric_value) — valid-only
        test_row = conn.execute(
            "SELECT name, metric_value, mean_nll, model_id "
            "FROM models "
            "WHERE split='test' AND status='ok' AND metric_value IS NOT NULL "
            "ORDER BY metric_value ASC "
            "LIMIT 1"
        ).fetchone()
        if not test_row:
            return

        result["best_test_metric"] = float(test_row[1])
        result["best_test_nll"] = (
            float(test_row[2]) if test_row[2] is not None else None
        )
        result["has_test_eval"] = True

        if "individual_differences" in tables:
            test_model_id = test_row[3]
            id_row = conn.execute(
                "SELECT id.mean_r2, id.max_r2, id.best_param "
                "FROM individual_differences id "
                "WHERE id.model_id=? AND id.split='test'",
                [test_model_id],
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


# --------------------------------------------------------------------------- #
# Config-level aggregation
# --------------------------------------------------------------------------- #

_NUMERIC_METRICS = [
    "best_train_metric",
    "best_val_metric",
    "best_test_metric",
    "best_test_nll",
    "best_test_mean_r2",
    "best_test_max_r2",
]

_CONFIG_COLUMN_ORDER = [
    "config_label",
    "n_runs",
    "n_with_test_eval",
    "n_with_individual_differences",
    "n_models_total",
    "n_failed_models_total",
    "best_train_metric_mean",
    "best_train_metric_std",
    "best_train_metric_n",
    "best_val_metric_mean",
    "best_val_metric_std",
    "best_val_metric_n",
    "best_test_metric_mean",
    "best_test_metric_std",
    "best_test_metric_n",
    "best_test_nll_mean",
    "best_test_nll_std",
    "best_test_nll_n",
    "best_test_mean_r2_mean",
    "best_test_mean_r2_std",
    "best_test_mean_r2_n",
    "best_test_max_r2_mean",
    "best_test_max_r2_std",
    "best_test_max_r2_n",
]


def aggregate_configs(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group run-level summaries by ``config_label`` and compute config-level aggregates.

    For each config group, numeric metrics are summarised with mean, std
    (sample standard deviation, ddof=1), and non-missing count.  Count
    columns and boolean flags are summed or counted across runs.

    Parameters
    ----------
    summaries:
        List of run summary dicts (as returned by :func:`summarise_run`).

    Returns
    -------
    list[dict]
        One row per config label, sorted alphabetically by label.  Each row
        contains all keys in ``_CONFIG_COLUMN_ORDER``.
    """
    import numpy as np

    groups: dict[str, list[dict[str, Any]]] = {}
    for s in summaries:
        label = s.get("config_label", "")
        groups.setdefault(label, []).append(s)

    config_rows: list[dict[str, Any]] = []
    for label in sorted(groups):
        runs = groups[label]
        row: dict[str, Any] = {"config_label": label}

        # Count columns
        row["n_runs"] = len(runs)
        row["n_with_test_eval"] = sum(
            1 for r in runs if r.get("has_test_eval")
        )
        row["n_with_individual_differences"] = sum(
            1 for r in runs if r.get("has_individual_differences")
        )
        row["n_models_total"] = sum(r.get("n_models", 0) for r in runs)
        row["n_failed_models_total"] = sum(
            r.get("n_failed_models", 0) for r in runs
        )

        # Numeric metric aggregates
        for metric in _NUMERIC_METRICS:
            vals = [
                r.get(metric) for r in runs if r.get(metric) is not None
            ]
            if vals:
                arr = np.array(vals, dtype=float)
                row[f"{metric}_mean"] = float(np.mean(arr))
                row[f"{metric}_std"] = (
                    float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                )
                row[f"{metric}_n"] = len(arr)
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
                row[f"{metric}_n"] = 0

        config_rows.append(row)

    return config_rows


def write_config_summary_csv(
    config_rows: list[dict[str, Any]], output_dir: str | Path
) -> Path:
    """Write a ``config_summary.csv`` with one row per config label.

    Parameters
    ----------
    config_rows:
        List of config summary dicts (as returned by :func:`aggregate_configs`).
    output_dir:
        Directory to write ``config_summary.csv`` into.

    Returns
    -------
    Path
        Absolute path to the written CSV file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "config_summary.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=_CONFIG_COLUMN_ORDER, extrasaction="ignore"
        )
        writer.writeheader()
        for row in config_rows:
            out_row = {k: row.get(k) for k in _CONFIG_COLUMN_ORDER}
            writer.writerow(out_row)

    return csv_path.resolve()


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
  .summary-cell {{ font-size: 0.85em; color: #555; }}
</style>
</head>
<body>
<h1>GeCCo Results Comparison</h1>
<p>Generated from <strong>{n_runs}</strong> run director{plural} across <strong>{n_configs}</strong> configuration{config_plural}.</p>

{missing_data_notes}

{config_table_html}

<h2>Run Details</h2>
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
  <div><h3>Test Evaluation by Config</h3><img src="figures/model_fit_by_config.png" alt="Test evaluation by config"></div>
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
    summaries: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    config_rows: list[dict[str, Any]] | None = None,
) -> Path:
    """Render a self-contained static HTML report from the run summaries.

    Parameters
    ----------
    summaries:
        List of run summary dicts.
    output_dir:
        Directory to write ``report.html`` into.
    config_rows:
        Optional list of config-level summary dicts (as returned by
        :func:`aggregate_configs`).  When provided a "Config Summary"
        table is rendered above the run-level table.

    Returns
    -------
    Path
        Absolute path to the written HTML file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build missing data notes
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

    # Build config-level summary table
    config_table_html = ""
    if config_rows:
        config_table_html = (
            "<h2>Config Summary</h2>\n"
            "<table>\n<thead>\n<tr>\n"
            "  <th>Config</th>\n"
            "  <th>Runs</th>\n"
            "  <th>Train Metric</th>\n"
            "  <th>Val Metric</th>\n"
            "  <th>Test Metric</th>\n"
            "  <th>Test NLL</th>\n"
            "  <th>ID Mean R²</th>\n"
            "  <th>ID Max R²</th>\n"
            "  <th>Test Eval</th>\n"
            "  <th>ID Data</th>\n"
            "</tr>\n</thead>\n<tbody>\n"
        )
        for cr in config_rows:
            def _mean_std_cell(prefix: str) -> str:
                mean = cr.get(f"{prefix}_mean")
                std = cr.get(f"{prefix}_std")
                n = cr.get(f"{prefix}_n", 0)
                if mean is None:
                    return '<span class="missing">—</span>'
                if std is not None and n is not None and n > 1:
                    return f"{_format_val(mean)} ± {_format_val(std)} <span class=\"summary-cell\">(n={n})</span>"
                return f"{_format_val(mean)} <span class=\"summary-cell\">(n={n})</span>"

            config_table_html += (
                "<tr>"
                f"<td>{_format_val(cr.get('config_label'))}</td>"
                f"<td>{cr.get('n_runs', 0)}</td>"
                f"<td>{_mean_std_cell('best_train_metric')}</td>"
                f"<td>{_mean_std_cell('best_val_metric')}</td>"
                f"<td>{_mean_std_cell('best_test_metric')}</td>"
                f"<td>{_mean_std_cell('best_test_nll')}</td>"
                f"<td>{_mean_std_cell('best_test_mean_r2')}</td>"
                f"<td>{_mean_std_cell('best_test_max_r2')}</td>"
                f"<td>{cr.get('n_with_test_eval', 0)}/{cr.get('n_runs', 0)}</td>"
                f"<td>{cr.get('n_with_individual_differences', 0)}/{cr.get('n_runs', 0)}</td>"
                "</tr>\n"
            )
        config_table_html += "</tbody>\n</table>\n"

    # Build run-level table rows
    rows_html = ""
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
    n_configs = len(config_rows) if config_rows else 0
    plural = "y" if n_runs == 1 else "ies"
    config_plural = "s" if n_configs != 1 else ""

    html = _HTML_TEMPLATE.format(
        n_runs=n_runs,
        plural=plural,
        n_configs=n_configs,
        config_plural=config_plural,
        missing_data_notes="\n".join(missing_notes),
        config_table_html=config_table_html,
        rows_html=rows_html,
    )

    html_path = out / "report.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path.resolve()


# --------------------------------------------------------------------------- #
# Figure export
# --------------------------------------------------------------------------- #


def _prepare_fit_vs_prediction_data(
    summaries: list[dict[str, Any]],
    config_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[float | None], list[float | None]]:
    """Prepare data for the fit-vs-prediction scatter plot.

    When *config_rows* is provided the scatter uses config-level means
    (one point per config).  Otherwise run-level values are used.
    """
    if config_rows:
        labels = [
            cr.get("config_label", f"cfg_{i}")
            for i, cr in enumerate(config_rows)
        ]
        x_vals = [cr.get("best_train_metric_mean") for cr in config_rows]
        y_vals = [cr.get("best_test_metric_mean") for cr in config_rows]
    else:
        labels = [
            s.get("config_label", s.get("run_id", f"run_{i}"))
            for i, s in enumerate(summaries)
        ]
        x_vals = [s.get("best_train_metric") for s in summaries]
        y_vals = [s.get("best_test_metric") for s in summaries]
    return labels, x_vals, y_vals


def _config_level_series(
    config_rows: list[dict[str, Any]], prefix: str
) -> tuple[list[float], list[float]]:
    """Extract config-level means and stds for a metric *prefix*.

    Missing (``None``) values are returned as ``NaN`` so that they are
    not rendered as zero-height bars in bar charts.
    """
    import numpy as np

    means: list[float] = []
    errors: list[float] = []
    for cr in config_rows:
        m = cr.get(f"{prefix}_mean")
        s = cr.get(f"{prefix}_std")
        means.append(float(m) if m is not None else np.nan)
        errors.append(float(s) if s is not None else np.nan)
    return means, errors


def export_figures(
    summaries: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    config_rows: list[dict[str, Any]] | None = None,
) -> Path:
    """Export comparison figures as PNG and PDF under ``output_dir/figures/``.

    Three figures are generated:

    1. ``model_fit_by_config`` — bar chart of model-comparison metrics per
       config.  At config level this uses only test-evaluation metrics; at
       run level it continues to show best train/val/test metrics.
    2. ``individual_differences_by_config`` — bar chart of ID R² values per
       config.  When *config_rows* is provided, bars show config-level means
       with error bars.
    3. ``fit_vs_prediction`` — scatter of train vs test metrics.  When
       *config_rows* is provided the scatter uses config-level means (one
       point per config); otherwise run-level values are used.

    Parameters
    ----------
    summaries:
        List of run summary dicts.
    output_dir:
        Directory under which a ``figures/`` sub-directory is created.
    config_rows:
        Optional list of config-level summary dicts.  When provided the
        config comparison bar charts and the fit-vs-prediction scatter use
        config-level aggregates; the main comparison chart is driven only by
        test-evaluation metrics.

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

    if config_rows:
        _export_config_level_figures(fig_dir, config_rows)
    else:
        # Fallback: use run-level summaries as before
        labels_run = [
            s.get("config_label", s.get("run_id", f"run_{i}"))
            for i, s in enumerate(summaries)
        ]
        train_vals = [s.get("best_train_metric") for s in summaries]
        val_vals = [s.get("best_val_metric") for s in summaries]
        test_vals = [s.get("best_test_metric") for s in summaries]

        _bar_chart(
            fig_dir / "model_fit_by_config",
            labels_run,
            [
                ("Train", train_vals),
                ("Val", val_vals),
                ("Test", test_vals),
            ],
            "Model Fit by Config",
            "Metric Value",
            "Best Fit Metric (lower is better)",
        )

        id_mean = [s.get("best_test_mean_r2") for s in summaries]
        id_max = [s.get("best_test_max_r2") for s in summaries]

        _bar_chart(
            fig_dir / "individual_differences_by_config",
            labels_run,
            [
                ("Mean R²", id_mean),
                ("Max R²", id_max),
            ],
            "Individual Differences by Config",
            "R²",
            "Test Individual Differences (higher is better)",
        )

    # -- 3. Fit vs Prediction --
    labels_scatter, train_vals_scatter, test_vals_scatter = (
        _prepare_fit_vs_prediction_data(summaries, config_rows=config_rows)
    )

    title = (
        "Train Metric vs Test Metric (config means)"
        if config_rows
        else "Train Metric vs Test Metric"
    )
    _scatter_plot(
        fig_dir / "fit_vs_prediction",
        labels_scatter,
        train_vals_scatter,
        test_vals_scatter,
        title,
        "Best Train Metric",
        "Best Test Metric",
    )

    plt.close("all")
    return fig_dir.resolve()


def _export_config_level_figures(
    fig_dir: Path, config_rows: list[dict[str, Any]]
) -> None:
    """Draw config-level bar charts using config-row means with error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [cr.get("config_label", f"cfg_{i}") for i, cr in enumerate(config_rows)]

    # -- 1. Main config comparison figure (test-evaluation only) --
    test_means, test_errs = _config_level_series(config_rows, "best_test_metric")

    _bar_chart_with_errors(
        fig_dir / "model_fit_by_config",
        labels,
        [
            ("Test", test_means, test_errs),
        ],
        "Test Evaluation by Config (mean ± SD)",
        "Metric Value",
        "Best Test Metric (lower is better)",
    )

    # -- 2. Individual differences by config (config-level means) --
    id_mean_vals, id_mean_errs = _config_level_series(config_rows, "best_test_mean_r2")
    id_max_vals, id_max_errs = _config_level_series(config_rows, "best_test_max_r2")

    _bar_chart_with_errors(
        fig_dir / "individual_differences_by_config",
        labels,
        [
            ("Mean R²", id_mean_vals, id_mean_errs),
            ("Max R²", id_max_vals, id_max_errs),
        ],
        "Individual Differences by Config (mean ± SD)",
        "R²",
        "Test Individual Differences (higher is better)",
    )


def _bar_chart_with_errors(
    base_path: Path,
    labels: list[str],
    series: list[tuple[str, list[float], list[float]]],
    title: str,
    ylabel: str,
    caption: str,
) -> None:
    """Draw a grouped bar chart with error bars and save as PNG + PDF."""
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

    for i, (name, values, errors) in enumerate(series):
        offset = (i - (n_series - 1) / 2) * bar_width
        # Pass yerr when at least one error is valid; matplotlib skips
        # NaN entries in error bars.
        has_valid_err = any(
            e is not None and not np.isnan(e) and e > 0 for e in errors
        )
        bars = ax.bar(
            index + offset, values, bar_width, label=name, alpha=0.8,
            yerr=errors if has_valid_err else None,
            capsize=3,
        )
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
    """Add text labels above bars when value is not None and not NaN."""
    import numpy as np

    for bar, val in zip(bars, values):
        if val is not None and not np.isnan(val):
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
