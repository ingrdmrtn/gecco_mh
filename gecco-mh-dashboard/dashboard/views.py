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
    get_iteration_results_by_idx,
    get_model_code,
    list_iterations,
    list_judge_traces,
    load_judge_trace,
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
    recovery_failed = stats.get("recovery_failed", 0)
    errors = stats.get("errors", 0)
    cols = st.columns(8)  # Changed from 7 to 8
    cols[0].metric("Clients", f"{stats['n_clients']}", f"{stats['running']} running")
    cols[1].metric("Completed", f"{stats['complete']}")
    cols[2].metric("Iterations", f"{stats['iterations']}")
    cols[3].metric("Models", f"{succeeded}")
    cols[4].metric("Recovery Failed", f"{recovery_failed}")
    cols[5].metric("Errors", f"{errors}")
    cols[6].metric("Param sets", f"{stats['param_combos']}")
    cols[7].metric("Best BIC", _fmt_float(best.get("metric_value")))

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Global best")
            if best:
                st.write(f"- Client: **{best.get('client_id', '-')}**")
                st.write(f"- Iteration: **{best.get('iteration', '-')}**")
                st.write(
                    f"- Parameters: {', '.join(best.get('param_names', [])) or '-'}"
                )
            else:
                st.info("No fitted model yet.")
        with c2:
            st.subheader("Baseline")
            if baseline and baseline.get("metric_value") is not None:
                st.write(
                    f"- Baseline BIC: **{_fmt_float(baseline.get('metric_value'))}**"
                )
                if best and best.get("metric_value") is not None:
                    delta = baseline["metric_value"] - best["metric_value"]
                    st.write(f"- Improvement vs baseline: **{delta:+.2f}**")
                st.write(
                    f"- Parameters: {', '.join(baseline.get('param_names', [])) or '-'}"
                )
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

    pivot = tdf.pivot_table(
        index="Iteration", columns="Client", values="Best BIC", aggfunc="min"
    )
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

    st.caption(f"Top {top_n} models (by BIC) — click a row to view code")

    # Show top_n successful models first, then up to 10 failed models
    success_df = ldf[ldf["Status"] == "success"].head(top_n)
    failed_df = ldf[ldf["Status"] != "success"].head(10)

    show = pd.concat([success_df, failed_df])

    # Configure column styling
    styled_df = show.copy()
    styled_df["Status"] = styled_df["Status"].map(
        {
            "success": "🟢 Success",
            "recovery_failed": "🟠 Recovery Failed",
            "error": "🔴 Error",
        }
    )

    # Select display columns
    display_cols = [
        "Model",
        "Status",
        "BIC",
        "Max R²",
        "Best Param",
        "Mean R²",
        "Params",
        "Client",
        "Iteration",
    ]
    display_df = styled_df[display_cols]

    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    if event.selection.rows:
        row = show.iloc[event.selection.rows[0]]
        code = get_model_code(data, row["Model"], row["Client"], row["Iteration"])
        if code:
            st.subheader(f"Code: {row['Model']}")
            st.code(code, language="python")
        else:
            st.caption("Code not available for this model.")

        # Show error details for failed models
        if row["Status"] != "success":
            if row["Status"] == "recovery_failed" and row.get("Recovery R"):
                st.warning(f"Parameter recovery failed (r={row['Recovery R']:.2f})")
            elif row["Status"] == "error" and row.get("Error"):
                st.error(f"Error: {row['Error']}")


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
        selected_client = st.selectbox("Client", clients, index=0, key="results_client")
    # Filter iterations for selected client
    client_iters = [it for it in iterations if str(it["client_id"]) == selected_client]

    # Build display labels that disambiguate re-runs of the same iteration
    iter_labels: list[str] = []
    for it in client_iters:
        label = str(it["iteration"])
        if it["run"] > 0:
            label += f" (run {it['run'] + 1})"
        iter_labels.append(label)

    with col2:
        selected_label = st.selectbox(
            "Iteration",
            iter_labels,
            index=len(iter_labels) - 1 if iter_labels else 0,
            key="results_iter",
        )

    if not selected_label:
        return

    # Map selected label back to the iteration info
    label_idx = (
        iter_labels.index(selected_label) if selected_label in iter_labels else 0
    )
    iter_info = client_iters[label_idx]
    history_idx = iter_info["history_idx"]
    selected_iter = iter_info["iteration"]

    # Resolve client_id back to its original type
    client_id_typed: Any = selected_client
    for it in iterations:
        if str(it["client_id"]) == selected_client:
            client_id_typed = it["client_id"]
            break

    results = get_iteration_results_by_idx(data, history_idx)
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
            name = r.get("function_name", f"model_{i + 1}")
            bic = r.get("metric_value")
            bic_str = f"{bic:.2f}" if bic is not None else "N/A"
            metric_name = r.get("metric_name", "BIC")

            # Status indicator
            if metric_name == "RECOVERY_FAILED":
                icon = "🔴"
                if (
                    r.get("simulation_error")
                    and r.get("recovery_n_successful", -1) == 0
                ):
                    status_note = " — simulation error"
                else:
                    recovery_r = r.get("recovery_r")
                    r_note = f" (r={recovery_r:.2f})" if recovery_r is not None else ""
                    status_note = f" — recovery failed{r_note}"
            elif metric_name == "VALIDATION_ERROR":
                icon = "🔴"
                status_note = f" — validation error: {r.get('error_type', 'unknown')}"
            elif metric_name == "FIT_ERROR":
                icon = "🔴"
                status_note = " — fitting error"
            elif bic is not None and bic < float("inf"):
                icon = "🟢"
                status_note = ""
            else:
                icon = "🟡"
                status_note = " — failed to fit"

            with st.expander(
                f"{icon} **{name}** — {metric_name}: {bic_str}{status_note}"
            ):
                meta = meta_by_idx.get(i, {})

                # Error details for failed models
                error_msg = r.get("error")
                if error_msg:
                    st.error(f"**Error:** {error_msg}")
                sim_error = r.get("simulation_error")
                if sim_error:
                    st.error(f"**Simulation error:** {sim_error}")
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
                    st.markdown(
                        f"**Best param R²:** {max_r2:.3f}{bp_note} · **Mean R²:** {mean_r2:.3f}"
                        if mean_r2 is not None
                        else f"**Best param R²:** {max_r2:.3f}{bp_note}"
                    )
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

_CONFIDENCE_ICONS = {"high": "🟢", "medium": "🟡", "low": "🔴"}


def render_judge_tab(data: dict[str, Any], results_dir: Path) -> None:
    """Render the Judge tab with trace browser."""
    traces = list_judge_traces(results_dir)
    if not traces:
        st.info(
            "No judge traces available. Judge traces appear when tool-using judge "
            "mode is enabled (`judge.mode: tool_using` in config)."
        )
        return

    iterations = list_iterations(data)
    if not iterations:
        st.info("No iteration data available.")
        return

    clients = sorted({str(it["client_id"]) for it in iterations})

    col1, col2 = st.columns(2)
    with col1:
        if len(clients) == 1:
            selected_client = clients[0]
            st.markdown(f"**Client:** {selected_client}")
        else:
            selected_client = st.selectbox(
                "Client", clients, index=0, key="judge_client"
            )

    client_iters = [it for it in iterations if str(it["client_id"]) == selected_client]

    available_trace_keys = {
        (t["iteration"], t["run_idx"], t["tag"]) for t in traces
    }

    iter_labels: list[str] = []
    iter_trace_map: list[tuple | None] = []
    for it in client_iters:
        label = str(it["iteration"])
        if it["run"] > 0:
            label += f" (run {it['run'] + 1})"
        key = (it["iteration"], it["run"], "")
        has_trace = key in available_trace_keys
        if not has_trace:
            tag_matches = [
                k for k in available_trace_keys
                if k[0] == it["iteration"] and k[1] == it["run"]
            ]
            if tag_matches:
                key = tag_matches[0]
                has_trace = True
        if has_trace:
            label += " ✅"
        else:
            label += " ⬜"
        iter_labels.append(label)
        iter_trace_map.append(key if has_trace else None)

    with col2:
        sel_idx = len(iter_labels) - 1 if iter_labels else 0
        selected_label = st.selectbox(
            "Iteration", iter_labels, index=sel_idx, key="judge_iter"
        )

    if not selected_label or not iter_labels:
        return

    label_idx = iter_labels.index(selected_label) if selected_label in iter_labels else 0
    trace_key = iter_trace_map[label_idx]

    if trace_key is None:
        st.info(
            "No judge trace for this iteration — the tool-using judge may be "
            "disabled or this is a legacy feedback run."
        )
        return

    iteration, run_idx, tag = trace_key
    trace = load_judge_trace(results_dir, iteration, run_idx, tag)
    if trace is None:
        st.warning("Trace file could not be loaded.")
        return

    render_judge_trace_viewer(trace)


def render_judge_timeline(full_trace: list[dict]) -> None:
    """Render the full investigation timeline as a vertical event stream."""
    for event in full_trace:
        event_type = event.get("type")
        
        if event_type == "planning":
            with st.container(border=True):
                st.markdown("**📋 Planning**")
                content = event.get("content", "")
                st.markdown(content)
        
        elif event_type == "tool_call":
            with st.container(border=True):
                tool_name = event.get("tool", "unknown")
                args = event.get("args", {})
                # Format args preview
                args_preview = ", ".join(f"{k}={v!r}" for k, v in args.items())
                if len(args_preview) > 60:
                    args_preview = args_preview[:57] + "..."
                st.markdown(f"**🔧 {tool_name}**({args_preview})")
                
                result_summary = event.get("result_summary", "")
                if result_summary:
                    st.caption(result_summary[:300])
                
                # Expander for full args and result
                with st.expander("Full details"):
                    st.subheader("Arguments")
                    st.json(args)
                    st.subheader("Result")
                    st.text(result_summary)
        
        elif event_type == "reflection":
            with st.container(border=True):
                st.markdown("*💭 **Reflection***")
                content = event.get("content", "")
                st.markdown(content)


def render_judge_trace_viewer(trace: dict[str, Any]) -> None:
    """Render a single judge trace with all sections."""
    is_short_circuit = trace.get("short_circuit", False)

    if is_short_circuit:
        source_iter = trace.get("source_iter", "?")
        st.warning(
            f"⚡ **Short-circuit verdict** — All candidate models from iteration "
            f"{source_iter} failed parameter recovery. The verdict was reused "
            f"from a prior iteration with failure notes appended."
        )
        recovery_failures = trace.get("recovery_failures", [])
        if recovery_failures:
            rows = []
            for rf in recovery_failures:
                name = rf.get("model", rf.get("function_name", "?"))
                mean_r = rf.get("mean_r", "N/A")
                mean_r_str = f"{mean_r:.2f}" if isinstance(mean_r, (int, float)) else str(mean_r)
                per_param = rf.get("recovery_per_param", {})
                if per_param:
                    worst = min(per_param, key=lambda k: per_param[k] if isinstance(per_param[k], (int, float)) else 0)
                    worst_str = f"{worst} r={per_param[worst]:.2f}" if isinstance(per_param[worst], (int, float)) else ""
                    worst_params = [f"{k} r={v:.2f}" for k, v in sorted(per_param.items(), key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else 0)][:3]
                    worst_str = ", ".join(worst_params)
                else:
                    worst_str = "-"
                rows.append({"Model": name, "Mean r": mean_r_str, "Worst parameters": worst_str})
            st.dataframe(rows, use_container_width=True, hide_index=True)

    tool_call_count = trace.get("tool_call_count", 0)
    wall_time = trace.get("wall_time_seconds", 0.0)
    timestamp = trace.get("timestamp", "")
    if timestamp:
        try:
            from datetime import datetime as _dt
            ts_display = _dt.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            ts_display = timestamp[:19] if len(timestamp) >= 19 else timestamp
    else:
        ts_display = "-"

    c1, c2, c3 = st.columns(3)
    c1.metric("Tool calls", tool_call_count)
    c2.metric("Wall time", f"{wall_time:.1f}s")
    c3.metric("Timestamp", ts_display)

    full_trace = trace.get("full_trace", [])
    tool_calls = trace.get("tool_call_trace", [])

    if full_trace:
        with st.expander("**Investigation Timeline**", expanded=True):
            render_judge_timeline(full_trace)
    elif tool_calls:
        with st.expander(f"**Tool Calls** ({len(tool_calls)} calls)", expanded=False):
            rows = []
            for i, tc in enumerate(tool_calls):
                args = tc.get("args", {})
                args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                if len(args_str) > 80:
                    args_str = args_str[:77] + "..."
                result_preview = tc.get("result_summary", "")[:200]
                rows.append({
                    "#": i + 1,
                    "Tool": tc.get("tool", ""),
                    "Args": args_str,
                    "Result Preview": result_preview,
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

            for i, tc in enumerate(tool_calls):
                with st.expander(
                    f"Call {i + 1}: {tc.get('tool', '?')}", expanded=False
                ):
                    st.json(tc.get("args", {}))
                    st.text(tc.get("result_summary", ""))
    elif is_short_circuit:
        st.caption("No tool calls (short-circuit verdict).")
    else:
        st.caption("No tool calls recorded.")

    per_angle = trace.get("per_angle", [])
    if per_angle:
        st.subheader("Per-Angle Analysis")
        for row_idx in range(0, len(per_angle), 2):
            cols = st.columns(2)
            for col_idx in range(2):
                angle_idx = row_idx + col_idx
                if angle_idx >= len(per_angle):
                    break
                angle_data = per_angle[angle_idx]
                with cols[col_idx]:
                    with st.container(border=True):
                        angle_name = angle_data.get("angle", "Unknown")
                        confidence = angle_data.get("confidence", "")
                        icon = _CONFIDENCE_ICONS.get(confidence, "⚪")
                        st.markdown(f"**{angle_name}**")
                        st.markdown(f"{icon} **{confidence.upper()}**" if confidence else "⚪ Analysis pending")
                        supporting = angle_data.get("supporting_tool_calls", [])
                        if supporting:
                            pills = " ".join(f"`{t}`" for t in supporting)
                            st.markdown(f"<small>📎 {pills}</small>", unsafe_allow_html=True)
                        findings = angle_data.get("findings", "")
                        if findings:
                            lines = findings.split("\n")
                            preview = "\n".join(lines[:3])
                            if len(lines) > 3:
                                with st.expander("Full findings"):
                                    st.markdown(findings)
                            else:
                                st.markdown(preview)
    elif is_short_circuit:
        pass
    else:
        st.info("No per-angle analysis available.")

    recommendations = trace.get("key_recommendations", [])
    if recommendations:
        st.subheader("Key Recommendations")
        for i, rec in enumerate(recommendations[:5], 1):
            st.markdown(f"{i}. {rec}")

    feedback = trace.get("synthesized_feedback", "")
    if feedback:
        with st.expander("**Synthesized Feedback**"):
            st.caption(
                "Feedback is written for the model code generator — not for human interpretation."
            )
            st.markdown(feedback)
