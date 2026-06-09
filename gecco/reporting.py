"""DuckDB-backed report summary and render helpers."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import duckdb


def _resolve_db_path(results_dir: Path | str | None, db_path: Path | str | None) -> Path:
    """Resolve the explicit DuckDB path for report loading."""
    if db_path is not None:
        return Path(db_path)
    if results_dir is not None:
        return Path(results_dir) / "diagnostics.duckdb"
    raise ValueError("Either results_dir or db_path must be provided")


def _fetch_metric_trajectory(connection: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Return the best metric per iteration using DuckDB aggregation."""
    rows = connection.execute(
        """
        SELECT iteration, MIN(metric_value) AS best_metric
        FROM models
        WHERE split = 'train' AND metric_value IS NOT NULL
        GROUP BY iteration
        ORDER BY iteration
        """
    ).fetchall()
    return [
        {"iteration": int(iteration), "best_metric": best_metric}
        for iteration, best_metric in rows
    ]


def _fetch_best_models(connection: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Return the best train-split model for each iteration."""
    rows = connection.execute(
        """
        SELECT iteration, function_name, metric_value, status
        FROM (
            SELECT
                iteration,
                name AS function_name,
                metric_value,
                status,
                ROW_NUMBER() OVER (
                    PARTITION BY iteration
                    ORDER BY metric_value ASC NULLS LAST, model_id ASC
                ) AS row_num
            FROM models
            WHERE split = 'train' AND metric_value IS NOT NULL
        ) ranked
        WHERE row_num = 1
        ORDER BY iteration
        """
    ).fetchall()
    return [
        {
            "iteration": int(iteration),
            "function_name": function_name,
            "metric_value": metric_value,
            "status": status,
        }
        for iteration, function_name, metric_value, status in rows
    ]


def _fetch_best_overall(connection: duckdb.DuckDBPyConnection) -> dict[str, Any] | None:
    """Return the best overall train-split model."""
    row = connection.execute(
        """
        SELECT iteration, name AS function_name, metric_value, status
        FROM models
        WHERE split = 'train' AND metric_value IS NOT NULL
        ORDER BY metric_value ASC NULLS LAST, model_id ASC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    iteration, function_name, metric_value, status = row
    return {
        "iteration": int(iteration),
        "function_name": function_name,
        "metric_value": metric_value,
        "status": status,
    }


def _load_summary_from_duckdb(db_path: Path) -> dict[str, Any]:
    """Build a compact serialisable summary from focused DuckDB queries."""
    with duckdb.connect(str(db_path), read_only=True) as connection:
        iteration_count_row = connection.execute("SELECT COUNT(*) FROM iterations").fetchone()
        iteration_count = 0 if iteration_count_row is None else int(iteration_count_row[0])
        metric_trajectory = _fetch_metric_trajectory(connection)
        best_models = _fetch_best_models(connection)
        best_overall = _fetch_best_overall(connection)

    results_root = db_path.parent
    return {
        "result_name": results_root.name,
        "results_dir": str(results_root),
        "db_path": str(db_path),
        "iteration_count": iteration_count,
        "metric_trajectory": metric_trajectory,
        "best_models": best_models,
        "best_overall": best_overall,
    }


def load_report_summary(
    *,
    results_dir: Path | str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load a compact report summary from an explicit DuckDB source."""
    resolved_db_path = _resolve_db_path(results_dir, db_path)
    if not resolved_db_path.exists():
        raise FileNotFoundError(f"No DuckDB report source found at {resolved_db_path}")
    return _load_summary_from_duckdb(resolved_db_path)


def render_report_text(summary: dict[str, Any]) -> str:
    """Render a plain-text report from an in-memory summary."""
    lines = [
        f"Report: {summary.get('result_name', 'unknown')}",
        f"Iteration count: {summary.get('iteration_count', 0)}",
    ]

    best_overall = summary.get("best_overall")
    if best_overall:
        lines.append(
            f"Best overall: {best_overall.get('function_name', 'unknown')} "
            f"({best_overall.get('metric_value', 'n/a')})"
        )

    trajectory = summary.get("metric_trajectory", []) or []
    if trajectory:
        lines.append("Metric trajectory:")
        for item in trajectory:
            lines.append(
                f"- iter {item.get('iteration')}: {item.get('best_metric', 'n/a')}"
            )

    best_models = summary.get("best_models", []) or []
    if best_models:
        lines.append("Best models:")
        for item in best_models:
            lines.append(
                f"- iter {item.get('iteration')}: {item.get('function_name', 'unknown')} "
                f"({item.get('metric_value', 'n/a')})"
            )

    return "\n".join(lines)


def render_report_html(summary: dict[str, Any]) -> str:
    """Render a minimal HTML report from an in-memory summary."""
    title = escape(str(summary.get("result_name", "unknown")))
    iteration_count = escape(str(summary.get("iteration_count", 0)))
    best_overall = summary.get("best_overall") or {}
    metric_trajectory = summary.get("metric_trajectory", []) or []
    best_models = summary.get("best_models", []) or []

    trajectory_items = "".join(
        f"<li>iter {escape(str(item.get('iteration')))}: "
        f"{escape(str(item.get('best_metric', 'n/a')))}</li>"
        for item in metric_trajectory
    )
    best_model_items = "".join(
        f"<li>iter {escape(str(item.get('iteration')))}: "
        f"{escape(str(item.get('function_name', 'unknown')))} "
        f"({escape(str(item.get('metric_value', 'n/a')))})</li>"
        for item in best_models
    )
    best_overall_text = ""
    if best_overall:
        best_overall_text = (
            f"<p><strong>Best overall:</strong> "
            f"{escape(str(best_overall.get('function_name', 'unknown')))} "
            f"({escape(str(best_overall.get('metric_value', 'n/a')))})</p>"
        )

    return (
        "<html><body>"
        f"<h1>Report: {title}</h1>"
        f"<p><strong>Iteration count:</strong> {iteration_count}</p>"
        f"{best_overall_text}"
        f"<h2>Metric trajectory</h2><ul>{trajectory_items}</ul>"
        f"<h2>Best models</h2><ul>{best_model_items}</ul>"
        "</body></html>"
    )
