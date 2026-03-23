from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from pathlib import Path

from dashboard.data_adapter import (
    build_baseline_row,
    build_client_df,
    build_iteration_df,
    build_landscape_df,
    build_r2_df,
    get_iteration_results,
    list_iterations,
    load_json_file,
    load_text_file,
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

    failed = stats.get("failed", 0)
    succeeded = stats["models"] - failed
    cols = st.columns(7)
    cols[0].metric("Clients", f"{stats['n_clients']}", f"{stats['running']} running")
    cols[1].metric("Completed", f"{stats['complete']}")
    cols[2].metric("Iterations", f"{stats['iterations']}")
    cols[3].metric("Models", f"{succeeded}")
    cols[4].metric("Failed", f"{failed}")
    cols[5].metric("Param sets", f"{stats['param_combos']}")
    cols[6].metric("Best BIC", _fmt_float(best.get("metric_value")))

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

    # Show baseline as a fixed reference row
    baseline_df = build_baseline_row(data)
    if baseline_df is not None:
        st.caption("Baseline")
        st.dataframe(baseline_df, use_container_width=True, hide_index=True)

    ldf = build_landscape_df(data)
    if ldf.empty:
        st.info("No models evaluated yet.")
        return

    st.caption(f"Top {top_n} models (by BIC)")
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


# ============================================================
# Results browser
# ============================================================

def _file_pattern(iteration: int, client_id: Any, suffix: str) -> str:
    """Build the filename pattern for a given iteration/client."""
    tag = f"_client{client_id}" if client_id is not None else ""
    return f"iter{iteration}{tag}_run0{suffix}"


def render_results_browser(data: dict[str, Any], results_dir: Path) -> None:
    """Rich results browser for inspecting iteration artifacts."""
    iterations = list_iterations(data)
    if not iterations:
        st.info("No iteration data available yet.")
        return

    # --- Filters ---
    clients = sorted({str(it["client_id"]) for it in iterations})
    col1, col2 = st.columns(2)
    with col1:
        selected_client = st.selectbox(
            "Client", clients, index=0, key="results_client"
        )
    # Filter iterations for selected client
    client_iters = [it for it in iterations if str(it["client_id"]) == selected_client]
    iter_options = [it["iteration"] for it in client_iters]
    with col2:
        selected_iter = st.selectbox(
            "Iteration", iter_options,
            index=len(iter_options) - 1 if iter_options else 0,
            key="results_iter",
        )

    if selected_iter is None:
        return

    # Resolve client_id back to its original type
    client_id_typed: Any = selected_client
    for it in iterations:
        if str(it["client_id"]) == selected_client:
            client_id_typed = it["client_id"]
            break

    results = get_iteration_results(data, client_id_typed, selected_iter)

    # --- Iteration summary ---
    iter_info = next(
        (it for it in client_iters if it["iteration"] == selected_iter), None
    )
    with st.container(border=True):
        st.markdown(
            f"**Iteration {selected_iter}** · Client {selected_client} · "
            f"**{iter_info['n_models'] if iter_info else 0}** models evaluated"
        )
        if iter_info and iter_info.get("best_bic") is not None:
            best_result = min(
                (r for r in results if r.get("metric_value") is not None),
                key=lambda r: r["metric_value"],
                default=None,
            )
            if best_result:
                st.markdown(
                    f"Best this iteration: **{best_result.get('function_name', '?')}** "
                    f"(BIC: **{best_result['metric_value']:.2f}**)"
                )

    # --- Models ---
    st.subheader("Models")

    # Load structured metadata if available
    structured_meta = load_json_file(
        results_dir, "models", _file_pattern(selected_iter, client_id_typed, ".json")
    )
    meta_by_idx: dict[int, dict] = {}
    if isinstance(structured_meta, list):
        for idx, m in enumerate(structured_meta):
            if isinstance(m, dict):
                meta_by_idx[idx] = m

    if not results:
        st.caption("No model results for this iteration.")
    else:
        for i, r in enumerate(results):
            name = r.get("function_name", f"model_{i+1}")
            bic = r.get("metric_value")
            bic_str = f"{bic:.2f}" if bic is not None else "N/A"
            metric_name = r.get("metric_name", "BIC")

            # Status indicator
            if metric_name == "RECOVERY_FAILED":
                icon = "🔴"
                recovery_r = r.get("recovery_r")
                r_note = f" (r={recovery_r:.2f})" if recovery_r is not None else ""
                status_note = f" — recovery failed{r_note}"
            elif metric_name == "FIT_ERROR":
                icon = "🔴"
                status_note = " — fitting error"
            elif bic is not None and bic < float("inf"):
                icon = "🟢"
                status_note = ""
            else:
                icon = "🟡"
                status_note = " — failed to fit"

            with st.expander(f"{icon} **{name}** — {metric_name}: {bic_str}{status_note}"):
                meta = meta_by_idx.get(i, {})

                # Error details for failed models
                error_msg = r.get("error")
                if error_msg:
                    st.error(f"**Error:** {error_msg}")
                recovery_per_param = r.get("recovery_per_param")
                if metric_name == "RECOVERY_FAILED" and recovery_per_param:
                    parts = [f"{k}: r={v:.2f}" for k, v in recovery_per_param.items()]
                    st.warning(f"**Recovery per param:** {', '.join(parts)}")

                # Rationale
                rationale = meta.get("rationale") or ""
                if rationale:
                    st.markdown(f"*{rationale}*")

                # Parameters
                params = r.get("param_names", [])
                if params:
                    st.markdown(f"**Parameters:** `{', '.join(params)}`")

                # R² info
                max_r2 = r.get("max_r2")
                best_param_r2 = r.get("best_param")
                mean_r2 = r.get("mean_r2")
                per_param_r2 = r.get("per_param_r2")
                if max_r2 is not None:
                    bp_note = f" ({best_param_r2})" if best_param_r2 else ""
                    st.markdown(f"**Best param R²:** {max_r2:.3f}{bp_note} · **Mean R²:** {mean_r2:.3f}" if mean_r2 is not None else f"**Best param R²:** {max_r2:.3f}{bp_note}")
                elif mean_r2 is not None:
                    st.markdown(f"**Mean R²:** {mean_r2:.3f}")
                if per_param_r2:
                    r2_parts = [f"{k}: {v:.3f}" for k, v in per_param_r2.items()]
                    st.markdown(f"**Per-param R²:** {', '.join(r2_parts)}")

                # Analysis / thinking
                analysis = meta.get("analysis") or ""
                if analysis:
                    with st.expander("LLM analysis / thinking", expanded=False):
                        st.markdown(analysis)

                # Code
                code = r.get("code") or ""
                if code:
                    st.code(code, language="python")

    # --- Feedback ---
    st.subheader("Feedback")
    feedback_text = load_text_file(
        results_dir, "feedback", _file_pattern(selected_iter, client_id_typed, ".txt")
    )
    if feedback_text:
        with st.expander("Feedback sent to LLM", expanded=False):
            st.text(feedback_text)
    else:
        st.caption("No feedback file found for this iteration.")

    # --- Raw LLM output ---
    raw_output = load_text_file(
        results_dir, "models", _file_pattern(selected_iter, client_id_typed, ".txt")
    )
    if raw_output:
        with st.expander("Raw LLM response", expanded=False):
            st.code(raw_output, language="text")

    # --- Raw registry JSON (moved from old Advanced tab) ---
    with st.expander("Raw shared registry JSON"):
        st.json(data)
