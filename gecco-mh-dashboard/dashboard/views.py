from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.data_adapter import (
    build_client_df,
    build_iteration_df,
    build_landscape_df,
    build_r2_df,
    summary_stats,
)


def _fmt_float(x: Any) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return "-"


def render_header(results_dir: str) -> None:
    st.title("🧠 GeCCo Distributed Dashboard")
    st.caption(f"Run: {results_dir} · Updated: {datetime.now().strftime('%H:%M:%S')}")


def render_overview(data: dict[str, Any]) -> None:
    stats = summary_stats(data)
    baseline = data.get("baseline") or {}
    best = data.get("global_best") or {}

    cols = st.columns(6)
    cols[0].metric("Clients", f"{stats['n_clients']}", f"{stats['running']} running")
    cols[1].metric("Completed", f"{stats['complete']}")
    cols[2].metric("Iterations", f"{stats['iterations']}")
    cols[3].metric("Models", f"{stats['models']}")
    cols[4].metric("Param sets", f"{stats['param_combos']}")
    cols[5].metric("Best BIC", _fmt_float(best.get("metric_value")))

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Global best")
            if best:
                st.write(f"- Client: **{best.get('client_id', '-')}**")
                st.write(f"- Iteration: **{best.get('iteration', '-')}**")
                st.write(f"- Parameters: {', '.join(best.get('param_names', [])) or '-'}")
            else:
                st.info("No fitted model yet.")
        with c2:
            st.subheader("Baseline")
            if baseline and baseline.get("metric_value") is not None:
                st.write(f"- Baseline BIC: **{_fmt_float(baseline.get('metric_value'))}**")
                if best and best.get("metric_value") is not None:
                    delta = baseline["metric_value"] - best["metric_value"]
                    st.write(f"- Improvement vs baseline: **{delta:+.2f}**")
                st.write(f"- Parameters: {', '.join(baseline.get('param_names', [])) or '-'}")
            else:
                st.info("Baseline not available yet.")


def render_clients(data: dict[str, Any]) -> None:
    st.subheader("Client status")
    df = build_client_df(data)
    if df.empty:
        st.info("No client updates yet.")
        return

    st.dataframe(df, use_container_width=True, hide_index=True)


def render_trajectory(data: dict[str, Any]) -> None:
    st.subheader("BIC trajectory")
    tdf = build_iteration_df(data)
    if tdf.empty:
        st.info("No iteration data yet.")
        return

    pivot = tdf.pivot_table(index="Iteration", columns="Client", values="Best BIC", aggfunc="min")
    st.line_chart(pivot, use_container_width=True)
    with st.expander("Trajectory table"):
        st.dataframe(tdf, use_container_width=True, hide_index=True)


def render_models(data: dict[str, Any], top_n: int) -> None:
    st.subheader("Model landscape")
    ldf = build_landscape_df(data)
    if ldf.empty:
        st.info("No models evaluated yet.")
        return

    show = ldf.head(top_n).copy()
    st.dataframe(show, use_container_width=True, hide_index=True)


def render_r2(data: dict[str, Any]) -> None:
    st.subheader("Individual differences (R²)")
    rdf = build_r2_df(data)
    if rdf.empty:
        st.caption("No R² metadata available yet.")
        return

    st.dataframe(rdf, use_container_width=True, hide_index=True)


def render_history(history: list[dict[str, Any]]) -> None:
    st.subheader("Session history")
    if not history:
        st.caption("No snapshots in this browser session.")
        return

    hdf = pd.DataFrame(history)
    hdf["ts"] = pd.to_datetime(hdf["ts"])
    hdf = hdf.sort_values("ts")
    st.line_chart(hdf.set_index("ts")[["best_bic"]], use_container_width=True)
