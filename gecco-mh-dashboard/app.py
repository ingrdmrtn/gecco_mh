#!/usr/bin/env python
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.config import DashboardConfig, available_tasks, default_results_dir
from dashboard.data_adapter import load_registry_snapshot
from dashboard.history_store import append_snapshot, init_history_state
from dashboard.views import (
    render_clients,
    render_header,
    render_history,
    render_models,
    render_overview,
    render_r2,
    render_results_browser,
    render_trajectory,
)


def main() -> None:
    st.set_page_config(
        page_title="GeCCo Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    cfg = DashboardConfig()
    init_history_state()

    with st.sidebar:
        st.header("Controls")
        tasks = available_tasks()
        if tasks:
            default_idx = tasks.index(cfg.default_task) if cfg.default_task in tasks else 0
            task = st.selectbox("Task", options=tasks, index=default_idx)
        else:
            task = st.text_input("Task name", value=cfg.default_task)
        results_override = st.text_input("Results directory (optional)", value="")
        refresh_seconds = st.slider("Refresh interval (seconds)", min_value=2, max_value=120, value=cfg.default_refresh_seconds)
        max_history_points = st.slider("Max session history points", min_value=50, max_value=5000, value=cfg.default_max_history_points, step=50)
        top_n = st.slider("Top models to show", min_value=5, max_value=100, value=20)
        auto_refresh = st.toggle("Auto refresh", value=True)
        refresh_now = st.button("Refresh now", type="primary")

    results_dir = Path(results_override).expanduser() if results_override else default_results_dir(task)

    render_header(str(results_dir))

    data = load_registry_snapshot(results_dir)
    if data is None:
        st.warning(f"Waiting for registry: {results_dir / 'shared_registry.json'}")
        st.stop()

    if refresh_now or auto_refresh:
        append_snapshot(data, max_points=max_history_points)

    tab_overview, tab_clients, tab_models, tab_results = st.tabs(
        ["Overview", "Clients", "Models", "Results"]
    )

    with tab_overview:
        render_overview(data)
        render_trajectory(data)
        render_history(st.session_state.dashboard_history)

    with tab_clients:
        render_clients(data)

    with tab_models:
        render_models(data, top_n=top_n)
        render_r2(data)

    with tab_results:
        render_results_browser(data, results_dir)

    if auto_refresh:
        time.sleep(refresh_seconds)
        st.rerun()


if __name__ == "__main__":
    main()
