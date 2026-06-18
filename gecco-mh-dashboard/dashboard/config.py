"""Dashboard configuration and result-discovery helpers."""

from __future__ import annotations

from pathlib import Path


class DashboardConfig:
    """Small configuration bundle for the Streamlit dashboard."""

    default_task = "test-evaluation"
    default_refresh_seconds = 5.0
    default_history_points = 25
    default_top_n = 10
    tab_names = ("Overview", "Models", "Clients", "Artifacts", "Judge")


def project_root() -> Path:
    """Return the repository root."""

    return Path(__file__).resolve().parents[2]


def default_results_dir(task_name: str | None = None) -> Path:
    """Return the canonical results directory for *task_name*."""

    task = task_name or DashboardConfig.default_task
    return project_root() / "results" / task


def _is_dashboard_result_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "shared_registry.duckdb").exists():
        return True
    if (path / "diagnostics.duckdb").exists():
        return True
    return any(path.glob("diagnostics_*.duckdb"))


def available_tasks() -> list[str]:
    """List result directories that contain current DuckDB dashboard state."""

    results_root = project_root() / "results"
    if not results_root.exists():
        return []

    tasks = [path.name for path in results_root.iterdir() if _is_dashboard_result_dir(path)]
    return sorted(tasks)
