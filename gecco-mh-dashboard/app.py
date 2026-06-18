"""Streamlit entrypoint for the GeCCo-MH dashboard."""

from __future__ import annotations

import time
from pathlib import Path

try:  # pragma: no cover - streamlit is optional in the test environment
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None

from dashboard.config import DashboardConfig, available_tasks, default_results_dir, project_root
from dashboard.components import sidebar_section
from dashboard.data_adapter import (
    load_feedback_artifacts,
    load_diagnostics_model_rows,
    load_diagnostics_summary,
    load_judge_trace_artifacts,
    load_registry_snapshot,
    normalize_judge_iterations,
    summary_stats,
)
from dashboard.history_store import append_snapshot, get_history
from dashboard.theme import apply_dashboard_theme
from dashboard.views import (
    render_clients_tab,
    render_judge_tab,
    render_models_tab,
    render_overview_tab,
    render_results_tab,
    render_waiting_state,
)


class SidebarSelection:
    def __init__(
        self,
        *,
        task_name: str,
        results_dir: Path,
        auto_refresh: bool,
        refresh_seconds: float,
        max_history_points: int,
        top_n: int,
        manual_refresh: bool,
    ) -> None:
        self.task_name = task_name
        self.results_dir = results_dir
        self.auto_refresh = auto_refresh
        self.refresh_seconds = refresh_seconds
        self.max_history_points = max_history_points
        self.top_n = top_n
        self.manual_refresh = manual_refresh


def _ensure_streamlit() -> None:
    if st is None:
        raise ImportError("Streamlit is not installed. Install the dashboard extra to run the app.")


def _read_sidebar() -> SidebarSelection:
    cfg = DashboardConfig()
    tasks = available_tasks() or [cfg.default_task]
    with sidebar_section("Data source"):
        task_name = st.sidebar.selectbox("Result task", tasks, index=0)
        results_dir_text = st.sidebar.text_input("Results directory", value=str(default_results_dir(task_name)))

    with sidebar_section("Refresh"):
        auto_refresh = st.sidebar.checkbox("Auto refresh", value=True)
        refresh_seconds = st.sidebar.number_input(
            "Refresh interval (seconds)",
            min_value=0.5,
            max_value=120.0,
            value=float(cfg.default_refresh_seconds),
            step=0.5,
        )
        manual_refresh = st.sidebar.button("Refresh now")

    with sidebar_section("Display"):
        max_history_points = int(
            st.sidebar.number_input(
                "Max history points",
                min_value=1,
                max_value=500,
                value=cfg.default_history_points,
                step=1,
            )
        )
        top_n = int(
            st.sidebar.number_input(
                "Top N models",
                min_value=1,
                max_value=100,
                value=cfg.default_top_n,
                step=1,
            )
        )
    return SidebarSelection(
        task_name=task_name,
        results_dir=Path(results_dir_text),
        auto_refresh=auto_refresh,
        refresh_seconds=float(refresh_seconds),
        max_history_points=max_history_points,
        top_n=top_n,
        manual_refresh=manual_refresh,
    )


def _load_state(selection: SidebarSelection) -> dict[str, object]:
    registry_path = selection.results_dir / "shared_registry.duckdb"
    registry_available = registry_path.exists()

    snapshot = None
    summary = None
    judge_rows: list[dict[str, object]] = []
    feedback_artifacts: list[dict[str, object]] = []
    judge_trace_artifacts: list[dict[str, object]] = []
    stats: dict[str, int] = {"complete": 0, "running": 0, "errors": 0, "recovery_failed": 0}
    diagnostics_rows = []

    if registry_available:
        snapshot = load_registry_snapshot(selection.results_dir)
        summary = load_diagnostics_summary(selection.results_dir)
        feedback_artifacts = load_feedback_artifacts(selection.results_dir)
        judge_trace_artifacts = load_judge_trace_artifacts(selection.results_dir)
        if snapshot is not None:
            stats = summary_stats(snapshot)
            judge_rows = normalize_judge_iterations(snapshot)
        diagnostics_rows = load_diagnostics_model_rows(selection.results_dir, iteration=0)

    return {
        "registry_available": registry_available,
        "registry_path": registry_path,
        "snapshot": snapshot,
        "summary": summary,
        "feedback_artifacts": feedback_artifacts,
        "judge_trace_artifacts": judge_trace_artifacts,
        "stats": stats,
        "judge_rows": judge_rows,
        "diagnostics_rows": diagnostics_rows,
    }


def main() -> None:
    """Render the dashboard shell."""

    _ensure_streamlit()

    st.set_page_config(page_title="GeCCo-MH Dashboard", layout="wide")
    apply_dashboard_theme()
    st.title("GeCCo-MH Dashboard")
    st.caption(f"Project root: {project_root()}")

    selection = _read_sidebar()
    state = _load_state(selection)

    append_snapshot(
        {
            "task_name": selection.task_name,
            "results_dir": str(selection.results_dir),
            "registry_available": state["registry_available"],
            "stats": state["stats"],
        },
        max_points=selection.max_history_points,
    )

    history = get_history()
    if not state["registry_available"]:
        render_waiting_state(str(selection.results_dir), str(state["registry_path"]))

    tabs = st.tabs(list(DashboardConfig.tab_names))
    tab_renderers = (
        lambda: render_overview_tab(state["snapshot"], state["summary"], history),
        lambda: render_models_tab(state["summary"], snapshot=state["snapshot"], top_n=selection.top_n),
        lambda: render_clients_tab(state["snapshot"]),
        lambda: render_results_tab(
            state["summary"],
            results_dir=selection.results_dir,
            feedback_artifacts=state["feedback_artifacts"],
            snapshot=state["snapshot"],
            trace_artifacts=state["judge_trace_artifacts"],
        ),
        lambda: render_judge_tab(
            state["judge_rows"],
            results_dir=selection.results_dir,
            trace_artifacts=state["judge_trace_artifacts"],
        ),
    )
    for tab, render in zip(tabs, tab_renderers, strict=True):
        with tab:
            render()

    if selection.manual_refresh:
        st.rerun()
        return

    if selection.auto_refresh and state["registry_available"]:
        time.sleep(selection.refresh_seconds)
        st.rerun()


if __name__ == "__main__":  # pragma: no cover
    main()
