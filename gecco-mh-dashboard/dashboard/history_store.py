from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st


def init_history_state() -> None:
    if "dashboard_history" not in st.session_state:
        st.session_state.dashboard_history = []


def append_snapshot(snapshot: dict[str, Any], max_points: int) -> None:
    init_history_state()
    best_bic = None
    best = snapshot.get("global_best") if snapshot else None
    if isinstance(best, dict):
        best_bic = best.get("metric_value")

    st.session_state.dashboard_history.append(
        {
            "ts": datetime.now(),
            "best_bic": best_bic,
            "n_iterations": len(snapshot.get("iteration_history", [])) if snapshot else 0,
        }
    )

    prune_history(max_points)


def prune_history(max_points: int) -> None:
    init_history_state()
    if max_points <= 0:
        st.session_state.dashboard_history = []
        return

    history = st.session_state.dashboard_history
    if len(history) > max_points:
        st.session_state.dashboard_history = history[-max_points:]
