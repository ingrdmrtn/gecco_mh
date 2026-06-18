"""Minimal Streamlit views for the dashboard bootstrap."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from dashboard import components
from dashboard.data_adapter import build_client_df, build_model_comparison_frame, build_overview_summary, model_dashboard_key

try:  # pragma: no cover - streamlit is optional in the test environment
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None


def _overview_caption(summary: dict[str, Any], history: list[dict[str, Any]]) -> str:
    caption_bits = ["Command center for the current DuckDB registry snapshot."]
    if history:
        latest = history[-1]
        captured_at = latest.get("captured_at")
        if captured_at:
            caption_bits.append(f"Last captured {captured_at}.")
    if summary.get("run_state_label"):
        caption_bits.append(f"Run state: {summary['run_state_label']}.")
    return " ".join(caption_bits)


def _overview_metric_rows(summary: dict[str, Any]) -> None:
    components.metric_grid(
        [
            {
                "label": "Run state",
                "value": summary.get("run_state_label") or "Idle",
                "help": "Current run status derived from registry client state.",
            },
            {
                "label": "Best model",
                "value": summary.get("best_model_name") or "Not available",
                "delta": (
                    f"BIC {summary['best_bic']:.2f}" if summary.get("best_bic") is not None else None
                ),
                "help": "Best model found so far from the registry trajectory.",
            },
            {
                "label": "Baseline BIC",
                "value": components.format_value(summary.get("baseline_bic")),
                "delta": (
                    f"Δ {summary['bic_delta']:+.2f}"
                    if summary.get("bic_delta") is not None
                    else None
                ),
                "help": "Baseline comparison from the shared registry.",
            },
            {
                "label": "Health",
                "value": summary.get("health_label") or "Waiting",
                "help": "High-level run health derived from client status mix.",
            },
        ]
    )

    components.metric_grid(
        [
            {"label": "Clients", "value": components.format_value(summary.get("client_count"))},
            {"label": "Iterations", "value": components.format_value(summary.get("iteration_count"))},
            {"label": "Successful models", "value": components.format_value(summary.get("trajectory_points"))},
            {"label": "Failures", "value": components.format_value(summary.get("error_clients"))},
            {"label": "Param sets", "value": components.format_value(summary.get("param_set_count"))},
            {"label": "Best BIC", "value": components.format_value(summary.get("best_bic"))},
        ]
    )


def _render_overview_best_vs_baseline(summary: dict[str, Any]) -> None:
    if st is None:
        return

    components.section_header("Baseline comparison")
    left, right = st.columns(2) if hasattr(st, "columns") else (None, None)

    if left is not None:
        with left:
            if summary.get("best_bic") is None:
                components.empty_state("No global best model yet.")
            else:
                st.caption("Best model")
                st.write(f"Best model: {summary.get('best_model_name') or 'Not available'}")
                st.metric("Best BIC", components.format_value(summary.get("best_bic")))
                st.caption(
                    f"Client {components.format_value(summary.get('best_client_id'))} • Iteration {components.format_value(summary.get('best_iteration'))}"
                )
                if summary.get("best_param_names"):
                    st.caption(f"Parameters: {', '.join(str(value) for value in summary['best_param_names'])}")

    if right is not None:
        with right:
            if summary.get("baseline_bic") is None:
                components.empty_state("Baseline comparison not available.")
            else:
                st.metric(
                    "Baseline BIC",
                    components.format_value(summary.get("baseline_bic")),
                    delta=(
                        f"Δ {summary['bic_delta']:+.2f} ({summary['bic_delta_pct']:+.1f}%)"
                        if summary.get("bic_delta") is not None and summary.get("bic_delta_pct") is not None
                        else None
                    ),
                )
                st.caption("Lower BIC is better.")


def _render_overview_health(summary: dict[str, Any], history: list[dict[str, Any]]) -> None:
    if st is None:
        return

    components.section_header("Run health")
    left, right = st.columns(2) if hasattr(st, "columns") else (None, None)

    if left is not None:
        with left:
            st.metric(
                "Client mix",
                f"{components.format_value(summary.get('client_count'))} total",
                delta=(
                    f"{components.format_value(summary.get('running_clients'))} running / {components.format_value(summary.get('complete_clients'))} complete"
                    if summary.get("client_count") is not None
                    else None
                ),
            )
            st.caption(
                f"Failures: {components.format_value(summary.get('error_clients'))} • Recovery failed: {components.format_value(summary.get('recovery_failed_clients'))}"
            )

    if right is not None:
        with right:
            latest_capture = history[-1].get("captured_at") if history else None
            if latest_capture:
                st.metric("Latest capture", str(latest_capture))
            else:
                components.empty_state("No captured history yet.")
            if summary.get("latest_client_activity"):
                st.caption(f"Latest client activity: {summary['latest_client_activity']}")


def _render_overview_trend(summary: dict[str, Any]) -> None:
    if st is None:
        return

    components.section_header("BIC trajectory")
    iteration_frame = summary.get("iteration_frame")
    if iteration_frame is None or iteration_frame.empty:
        components.empty_state("No BIC trajectory available yet.")
        return

    trend_frame = iteration_frame[[column for column in ["iteration", "best_bic", "best_model_name", "client_id", "timestamp"] if column in iteration_frame.columns]].copy()
    trend_frame = trend_frame[trend_frame["best_bic"].notna()].copy() if "best_bic" in trend_frame.columns else trend_frame
    if trend_frame.empty:
        components.empty_state("No BIC trajectory available yet.")
        return

    trend_frame = trend_frame.sort_values(by="iteration", kind="stable")
    if hasattr(st, "line_chart"):
        chart_frame = trend_frame.set_index("iteration")[["best_bic"]]
        st.line_chart(chart_frame)
    components.write_dataframe(trend_frame)


def _render_overview_history(history: list[dict[str, Any]]) -> None:
    if st is None:
        return

    components.section_header("Session history")
    if not history:
        components.empty_state("No session history captured yet.")
        return

    history_frame = pd.DataFrame(history)
    columns = [column for column in ["captured_at", "task_name", "registry_available", "stats", "results_dir"] if column in history_frame.columns]
    components.write_dataframe(history_frame[columns] if columns else history_frame)


def _render_overview_debug(summary: dict[str, Any], *, show_debug: bool = False) -> None:
    if st is None:
        return

    with st.expander("Overview debug details", expanded=show_debug):
        st.caption("Derived tables only; raw registry JSON stays hidden by default.")
        client_frame = summary.get("client_frame")
        iteration_frame = summary.get("iteration_frame")
        if client_frame is not None:
            st.subheader("Client summary")
            components.write_dataframe(client_frame)
        if iteration_frame is not None:
            st.subheader("Iteration summary")
            components.write_dataframe(iteration_frame)


def render_overview(snapshot: dict[str, Any] | None, *, history: list[dict[str, Any]] | None = None, show_debug: bool = False) -> None:
    if st is None:
        return

    history = history or []
    summary = build_overview_summary(snapshot or {}) if snapshot is not None else build_overview_summary({})

    components.section_header("GeCCo run overview", caption=_overview_caption(summary, history))
    if summary.get("client_count", 0) == 0 and summary.get("iteration_count", 0) == 0 and not summary.get("has_global_best") and not summary.get("has_baseline"):
        components.empty_state("No running registry snapshot loaded yet.")
    _overview_metric_rows(summary)
    _render_overview_best_vs_baseline(summary)
    _render_overview_health(summary, history)
    components.section_header("Trajectory / session history")
    _render_overview_trend(summary)
    _render_overview_history(history)
    _render_overview_debug(summary, show_debug=show_debug)


def render_overview_tab(
    snapshot: dict[str, Any] | None,
    summary: pd.DataFrame | None,
    history: list[dict[str, Any]],
    *,
    show_debug: bool = False,
) -> None:
    del summary
    render_overview(snapshot, history=history, show_debug=show_debug)
def _artifact_title(prefix: str, artifact: dict[str, Any]) -> str:
    path = artifact.get("path") or artifact.get("absolute_path") or "artifact"
    kind = str(artifact.get("kind") or "artifact").replace("_", " ").title()
    return f"{prefix}: {kind} • {path}"


def _render_artifact_handoff(title: str, message: str) -> None:
    if st is None:
        return
    with st.expander(title, expanded=False):
        st.caption(message)


def _render_detail_payload(title: str, detail: dict[str, Any] | None) -> None:
    if st is None or not detail:
        return
    with st.expander(title, expanded=False):
        if detail.get("code"):
            st.subheader("Model code")
            st.code(detail["code"], language="python")
        if detail.get("validation_errors"):
            st.subheader("Validation errors")
            components.write_dataframe(pd.DataFrame(detail["validation_errors"]))
        if detail.get("parameter_recovery"):
            st.subheader("Parameter recovery")
            components.write_dataframe(pd.DataFrame([detail["parameter_recovery"]]))
        if detail.get("individual_differences"):
            st.subheader("R² details")
            components.write_dataframe(pd.DataFrame([detail["individual_differences"]]))
        if detail.get("ppc"):
            st.subheader("Feedback / raw LLM output")
            components.write_dataframe(pd.DataFrame(detail["ppc"]))
        if detail.get("block_residuals"):
            st.subheader("Block residuals")
            components.write_dataframe(pd.DataFrame(detail["block_residuals"]))
        components.debug_details("Raw model detail payload", detail)


def _render_plain_artifact(title: str, artifact: dict[str, Any]) -> None:
    if st is None:
        return
    with st.expander(title, expanded=False):
        if artifact.get("path"):
            st.caption(str(artifact["path"]))
        if artifact.get("content") is not None:
            if artifact.get("kind") == "model_code":
                st.code(str(artifact["content"]), language="python")
            else:
                st.write(str(artifact["content"]))
        if artifact.get("payload") is not None:
            st.json(artifact["payload"])
        if artifact.get("raw_text") is not None and artifact.get("error"):
            st.subheader("Raw text")
            st.code(str(artifact["raw_text"]), language="text")
        if artifact.get("error"):
            st.warning(str(artifact["error"]))
        components.debug_details("Raw artifact payload", artifact)


def _render_judge_artifact(title: str, artifact: dict[str, Any]) -> None:
    if st is None:
        return
    with st.expander(title, expanded=False):
        if artifact.get("path"):
            st.caption(str(artifact["path"]))
        for label, key in (
            ("Synthesized feedback", "synthesized_feedback"),
            ("Verdict", "verdict"),
            ("Tool call trace", "trace"),
            ("Full trace", "full_trace"),
        ):
            value = artifact.get(key)
            if value is None:
                continue
            st.subheader(label)
            if isinstance(value, (dict, list, tuple)):
                st.json(value)
            else:
                st.code(str(value), language="text")
        if artifact.get("payload") is not None:
            st.subheader("Raw payload")
            st.json(artifact["payload"])
        if artifact.get("error"):
            st.warning(str(artifact["error"]))
        components.debug_details("Raw judge trace artifact", artifact)


def _render_row_drilldown(title: str, row: dict[str, Any]) -> None:
    if st is None:
        return
    with st.expander(title, expanded=False):
        for key in ("metric_name", "metric_value", "mean_r2", "max_r2", "status", "split"):
            if key in row:
                components.key_value(key.replace("_", " ").title(), row.get(key))
        detail = row.get("detail") if isinstance(row.get("detail"), dict) else None
        if detail:
            _render_detail_payload("Model code / error / raw details", detail)
        components.debug_details("Raw result row payload", row)
def _missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, float):
        return pd.isna(value)
    return False


def _rankable_selection_frame(frame: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    if frame.empty:
        return frame

    ranked = frame[frame["display_rank"].notna()].copy() if "display_rank" in frame.columns else frame.copy()
    if "display_rank" in ranked.columns:
        ranked = ranked[ranked["display_rank"] <= top_n]
    non_rankable = frame[frame["display_rank"].isna()].copy() if "display_rank" in frame.columns else frame.iloc[0:0].copy()
    if ranked.empty and non_rankable.empty:
        return frame.iloc[0:0].copy()
    return pd.concat([ranked, non_rankable], ignore_index=True)


def _format_model_option(row: dict[str, Any], *, rank: Any = None) -> str:
    name = row.get("name") or row.get("function_name") or "Model"
    metric_name = row.get("metric_name") or "metric"
    metric_value = components.format_value(row.get("metric_value"), default="—")
    status = components.status_metadata(row.get("status"))["label"]
    prefix = f"#{int(rank)} " if isinstance(rank, int) else ""
    return f"{prefix}{name} • {metric_name} {metric_value} • {status}"


def _render_model_detail_panel(row: dict[str, Any], *, snapshot: dict[str, Any] | None = None) -> None:
    if st is None:
        return

    detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
    baseline = (snapshot or {}).get("baseline") or {}
    baseline_bic = baseline.get("metric_value")
    metric_value = row.get("metric_value")
    baseline_delta = None
    if baseline_bic is not None and metric_value is not None:
        try:
            baseline_delta = float(baseline_bic) - float(metric_value)
        except (TypeError, ValueError):
            baseline_delta = None

    caption_bits = [f"Selected {row.get('dashboard_model_key') or model_dashboard_key(row)}."]
    if baseline:
        caption_bits.append(f"Baseline: {baseline.get('name') or 'baseline'} ({components.format_value(baseline_bic, default='—')}).")
    if baseline_delta is not None:
        caption_bits.append(f"Δ vs baseline: {components.format_value(baseline_delta, default='—', precision=3)}.")
    components.section_header("Selected model detail", caption=" ".join(caption_bits))

    components.key_value("Model", row.get("name") or row.get("function_name") or "—")
    components.key_value("Status", components.status_metadata(row.get("status"))["label"])
    components.key_value("Metric", f"{components.format_value(metric_value, default='—')} ({row.get('metric_name') or 'metric'})")
    if row.get("split") is not None:
        components.key_value("Split", row.get("split"))
    if row.get("display_rank") is not None:
        components.key_value("Rank", row.get("display_rank"))

    params = row.get("param_names") or detail.get("param_names") or []
    if params:
        components.key_value("Parameters", ", ".join(str(param) for param in params))
    else:
        components.key_value("Parameters", "—")

    recovery = detail.get("parameter_recovery") or row.get("parameter_recovery")
    error_value = (
        detail.get("error")
        or row.get("error")
        or row.get("last_error")
        or row.get("error_message")
    )

    with st.expander("Error / recovery status", expanded=False):
        if error_value is not None and not _missing_value(error_value):
            st.warning(f"{error_value}")
        else:
            st.info("No explicit error recorded.")

        if isinstance(recovery, dict):
            if recovery.get("passed") is False:
                st.warning("Parameter recovery failed.")
            if recovery.get("simulation_error"):
                st.warning(f"Recovery simulation error: {recovery.get('simulation_error')}")
        elif recovery:
            st.write(recovery)

    provenance_bits = [
        f"{row.get('source_db') or 'registry'}",
        f"model_id={components.format_value(row.get('model_id'))}" if row.get("model_id") is not None else None,
        f"client_id={components.format_value(row.get('client_id'))}" if row.get("client_id") is not None else None,
        f"iteration={components.format_value(row.get('iteration'))}" if row.get("iteration") is not None else None,
        f"result_index={components.format_value(row.get('result_index'))}" if row.get("result_index") is not None else None,
        f"split={row.get('split')}" if row.get("split") is not None else None,
    ]
    provenance = " • ".join(bit for bit in provenance_bits if bit)
    components.key_value("Provenance", provenance)

    _render_artifact_handoff(
        "Model artifacts",
        "Raw generated code, recovery payloads, validation errors, R² details, and other debug material are centralized in the Artifacts tab.",
    )

    _render_detail_payload("Selected model details", detail or row)

    components.debug_details("Model summary payload", {"dashboard_model_key": row.get("dashboard_model_key"), "source_db": row.get("source_db"), "model_id": row.get("model_id"), "split": row.get("split")})


def render_waiting_state(results_dir: str, registry_path: str) -> None:
    if st is None:
        return
    st.warning(f"Waiting for {registry_path} in {results_dir}.")


def render_clients_tab(snapshot: dict[str, Any] | None) -> None:
    if st is None:
        return
    components.section_header("Clients", caption="Live client status and freshness.")
    if snapshot is None:
        components.empty_state("No client state available.")
        return
    frame = build_client_df(snapshot)
    if frame.empty:
        components.empty_state("No client rows available.")
        return

    status_series = frame["status"].map(components.normalize_status) if "status" in frame.columns else pd.Series(dtype=str)
    running_count = int(status_series.isin({"running", "retrying"}).sum())
    complete_count = int(status_series.isin({"complete", "complete_no_success", "success"}).sum())
    issue_count = int(status_series.isin({"error", "failed", "validation_error", "fit_error", "recovery_failed"}).sum())

    components.metric_grid(
        [
            {"label": "Clients", "value": components.format_value(len(frame))},
            {"label": "Running", "value": components.format_value(running_count)},
            {"label": "Complete", "value": components.format_value(complete_count)},
            {"label": "Issues", "value": components.format_value(issue_count)},
        ]
    )

    display_frame = frame.copy()
    display_frame["status"] = display_frame["status label"]
    display_frame["activity"] = display_frame["activity"].map(lambda value: components.format_value(value))
    display_frame["last iteration"] = display_frame["last iteration"].map(lambda value: components.format_value(value))
    display_frame["best BIC"] = display_frame["best BIC"].map(lambda value: components.format_value(value, precision=2))
    display_frame["updated"] = display_frame["updated age"]

    display_columns = [column for column in ["client", "status", "activity", "last iteration", "best BIC", "updated"] if column in display_frame.columns]
    components.write_dataframe(display_frame[display_columns])


def render_models_tab(summary: pd.DataFrame | None, *, snapshot: dict[str, Any] | None = None, top_n: int = 10) -> None:
    if st is None:
        return
    frame = build_model_comparison_frame(summary, snapshot=snapshot)
    if frame is None or frame.empty:
        caption = None
        if snapshot and snapshot.get("baseline"):
            baseline = snapshot["baseline"]
            caption = f"Baseline: {baseline.get('name') or 'baseline'} • BIC {components.format_value(baseline.get('metric_value'))}"
        components.section_header("Models", caption=caption)
        components.empty_state("No diagnostics or registry model rows available.")
        return

    baseline = (snapshot or {}).get("baseline") or {}
    caption_bits = []
    if baseline:
        caption_bits.append(f"Baseline: {baseline.get('name') or 'baseline'} • BIC {components.format_value(baseline.get('metric_value'))}")
    if frame["display_rank"].notna().any():
        best_row = frame[frame["display_rank"].notna()].iloc[0]
        if baseline and baseline.get("metric_value") is not None and best_row.get("metric_value") is not None:
            try:
                delta = float(baseline.get("metric_value")) - float(best_row.get("metric_value"))
            except (TypeError, ValueError):
                delta = None
            else:
                caption_bits.append(f"Best Δ vs baseline {components.format_value(delta, default='—', precision=3)}")
    components.section_header("Models", caption=" • ".join(caption_bits) if caption_bits else None)

    display_frame = _rankable_selection_frame(frame, top_n=top_n).reset_index(drop=True)
    display_columns = [
        column
        for column in ["display_rank", "dashboard_model_key", "name", "split", "metric_name", "metric_value", "mean_r2", "max_r2", "status"]
        if column in display_frame.columns
    ]
    components.write_dataframe(display_frame[display_columns])

    top_frame = display_frame
    if top_frame.empty:
        components.empty_state("No model rows to inspect.")
        return

    key_to_row = {str(row.get("dashboard_model_key") or model_dashboard_key(row.to_dict())): row.to_dict() for _, row in top_frame.iterrows()}
    option_keys = list(key_to_row)
    selectbox = getattr(st, "selectbox", None)
    if callable(selectbox):
        selected_key = selectbox(
            "Selected model",
            option_keys,
            index=0,
            format_func=lambda key: _format_model_option(key_to_row[str(key)], rank=key_to_row[str(key)].get("display_rank")),
        )
    else:
        selected_key = option_keys[0]
    selected_row = key_to_row[str(selected_key)]
    _render_model_detail_panel(selected_row, snapshot=snapshot)


def render_results_tab(
    summary: pd.DataFrame | None,
    *,
    results_dir: Path | str | None = None,
    feedback_artifacts: list[dict[str, Any]] | None = None,
    snapshot: dict[str, Any] | None = None,
    trace_artifacts: list[dict[str, Any]] | None = None,
) -> None:
    if st is None:
        return
    components.section_header("Artifacts", caption="Centralized raw registry, feedback, trace, and debug artifacts.")

    if snapshot is not None:
        with st.expander("Registry JSON", expanded=False):
            st.caption("Raw registry snapshot payload.")
            st.json(snapshot)

    if summary is None:
        components.empty_state("No diagnostics rows available.")
    else:
        columns = [column for column in ["dashboard_model_key", "name", "split", "metric_value", "status"] if column in summary.columns]
        components.write_dataframe(summary[columns].head(25) if columns else summary.head(25))

        st.subheader("Artifacts browser")
        for _, row in summary.head(25).iterrows():
            _render_row_drilldown(str(row.get("dashboard_model_key") or row.get("name") or "Result"), row.to_dict())

    if feedback_artifacts:
        components.section_header("Raw LLM output / feedback")
        for artifact in feedback_artifacts:
            title = _artifact_title("Artifact", artifact)
            _render_plain_artifact(title, artifact)

    if trace_artifacts:
        components.section_header("Full judge traces")
        for artifact in trace_artifacts:
            title = _artifact_title("Judge trace", artifact)
            _render_judge_artifact(title, artifact)


def _judge_feedback_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if not value:
            return None
        if "default" in value and value["default"]:
            return str(value["default"])
        parts = [f"{key}: {components.format_value(item)}" for key, item in value.items() if not _missing_value(item)]
        return "; ".join(parts) if parts else None
    if isinstance(value, (list, tuple)):
        items = [components.format_value(item) for item in value if not _missing_value(item)]
        return "; ".join(items) if items else None
    text = components.format_value(value)
    return text if text != "—" else None


def _judge_verdict_payload(row: dict[str, Any]) -> dict[str, Any]:
    verdict = row.get("verdict")
    return verdict if isinstance(verdict, dict) else {}


def _judge_verdict_label(row: dict[str, Any]) -> str:
    verdict = _judge_verdict_payload(row)
    if verdict:
        if verdict.get("accepted") is True:
            return "Accepted"
        if verdict.get("accepted") is False:
            return "Rejected"
        label = verdict.get("label") or verdict.get("decision") or verdict.get("status")
        if label is not None:
            return components.format_value(label, default="Verdict available")
        return "Verdict available"
    if bool(row.get("failed")):
        return "Failed"
    return "Awaiting verdict"


def _judge_status_label(row: dict[str, Any]) -> str:
    if bool(row.get("failed")):
        return "Failed"
    if _judge_verdict_payload(row):
        return "Resolved"
    return "Awaiting verdict"


def _judge_recommendations(row: dict[str, Any]) -> list[str]:
    verdict = _judge_verdict_payload(row)
    recommendations: list[str] = []
    raw_recommendations = verdict.get("key_recommendations") if verdict else None
    if isinstance(raw_recommendations, (list, tuple)):
        for recommendation in raw_recommendations:
            if not _missing_value(recommendation):
                recommendations.append(components.format_value(recommendation))
    elif not _missing_value(raw_recommendations):
        recommendations.append(components.format_value(raw_recommendations))
    return recommendations


def _judge_confidence_summary(row: dict[str, Any]) -> str | None:
    verdict = _judge_verdict_payload(row)
    per_angle = verdict.get("per_angle") if verdict else None
    if not isinstance(per_angle, (list, tuple)):
        return None

    entries: list[str] = []
    for angle in per_angle:
        if not isinstance(angle, dict):
            continue
        name = angle.get("angle") or angle.get("name") or "angle"
        confidence = angle.get("confidence")
        findings = angle.get("findings") or angle.get("summary")
        bits = [str(name)]
        if not _missing_value(confidence):
            bits.append(f"confidence={components.format_value(confidence)}")
        if not _missing_value(findings):
            bits.append(components.format_value(findings))
        entries.append(" • ".join(bits))

    if not entries:
        return None
    return "; ".join(entries)


def _judge_trace_artifact_for_iteration(iteration: Any, trace_artifacts: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not trace_artifacts:
        return None
    iteration_text = str(iteration)
    for artifact in trace_artifacts:
        artifact_iteration = artifact.get("iteration")
        if artifact_iteration is not None and str(artifact_iteration) == iteration_text:
            return artifact
        path = str(artifact.get("path") or "")
        payload = artifact.get("payload") if isinstance(artifact.get("payload"), dict) else {}
        if str(payload.get("iteration")) == str(iteration):
            return artifact
        basename = Path(path).name
        if re.search(rf"^iter{re.escape(iteration_text)}(?=(_|\.|$))", basename):
            return artifact
    return None


def _judge_trace_state(row: dict[str, Any], trace_artifact: dict[str, Any] | None) -> str:
    if trace_artifact is None:
        return "Trace JSON missing"

    payload = trace_artifact.get("payload") if isinstance(trace_artifact.get("payload"), dict) else {}
    trace = trace_artifact.get("trace")
    full_trace = trace_artifact.get("full_trace")
    trace_count = len(trace) if isinstance(trace, list) else 0
    full_trace_count = len(full_trace) if isinstance(full_trace, list) else 0

    if payload.get("short_circuit"):
        return f"Short-circuit trace • {trace_count} tool calls"
    if bool(row.get("failed")) and not trace_count and not full_trace_count:
        return "Failed before trace capture"

    parts = [f"{trace_count} tool calls"]
    if full_trace_count:
        parts.append(f"{full_trace_count} timeline steps")
    return " • ".join(parts)


def _judge_summary_frame(judge_rows: list[dict[str, Any]], trace_artifacts: list[dict[str, Any]] | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in judge_rows:
        trace_artifact = _judge_trace_artifact_for_iteration(row.get("iteration"), trace_artifacts)
        feedback_text = _judge_feedback_text(row.get("synthesized_feedback") or row.get("feedback"))
        recommendations = _judge_recommendations(row)
        confidence_summary = _judge_confidence_summary(row)
        rows.append(
            {
                "iteration": row.get("iteration"),
                "status": _judge_status_label(row),
                "verdict": _judge_verdict_label(row),
                "recommendations": "; ".join(recommendations) if recommendations else "—",
                "confidence summary": confidence_summary or "—",
                "trace state": _judge_trace_state(row, trace_artifact),
                "feedback": feedback_text or "—",
                "error": row.get("error") or "—",
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "iteration",
            "status",
            "verdict",
            "recommendations",
            "confidence summary",
            "trace state",
            "feedback",
            "error",
        ],
    )


def render_judge_tab(
    judge_rows: list[dict[str, Any]],
    *,
    results_dir: Path | str | None = None,
    trace_artifacts: list[dict[str, Any]] | None = None,
) -> None:
    if st is None:
        return
    components.section_header(
        "Judge",
        caption="Registry judge state, verdicts, recommendations, and confidence summaries appear before trace details.",
    )
    if not judge_rows:
        components.empty_state("No judge rows available.")
    else:
        components.section_header("Registry verdict summary")
        components.write_dataframe(_judge_summary_frame(judge_rows, trace_artifacts))

        components.section_header("Judge iteration details")
        st.subheader("Verdict / feedback drilldown")
        for row in judge_rows:
            trace_artifact = _judge_trace_artifact_for_iteration(row.get("iteration"), trace_artifacts)
            title = f"Judge iteration {row.get('iteration', '?')} — {_judge_verdict_label(row)}"
            with st.expander(title, expanded=False):
                st.subheader("Registry state")
                st.write(f"Status: {_judge_status_label(row)}")
                st.write(f"Verdict: {_judge_verdict_label(row)}")
                if row.get("failed"):
                    st.warning("Judge marked this iteration as failed.")
                feedback = _judge_feedback_text(row.get("synthesized_feedback") or row.get("feedback"))
                if feedback:
                    st.subheader("Feedback")
                    st.write(feedback)
                verdict = _judge_verdict_payload(row)
                if verdict:
                    st.subheader("Verdict")
                    st.write(verdict)
                recommendations = _judge_recommendations(row)
                if recommendations:
                    st.subheader("Recommendations")
                    for recommendation in recommendations:
                        st.write(f"- {recommendation}")
                confidence_summary = _judge_confidence_summary(row)
                if confidence_summary:
                    st.subheader("Confidence summary")
                    st.write(confidence_summary)
                trace_state = _judge_trace_state(row, trace_artifact)
                st.subheader("Trace state")
                st.write(trace_state)
                if trace_artifact is None:
                    components.empty_state("Trace JSON is optional; the registry verdict remains visible without it.")
                else:
                    st.caption("Full timeline/tool-call/raw trace details are centralized in Artifacts.")
                if row.get("error"):
                    st.subheader("Error")
                    st.write(row["error"])

        _render_artifact_handoff(
            "Judge artifacts",
            "Full judge traces, raw payloads, and generated debug output are centralized in the Artifacts tab.",
        )
