"""Read-only dashboard data adapters for DuckDB-backed persisted outputs."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd

from dashboard import components
from gecco.coordination import SharedRegistry


_TXT_ARTIFACT_NAME_RE = re.compile(
    r"^iter(?P<iteration>-?\d+)(?P<tag>.*?)_run(?P<run_idx>-?\d+)(?P<participant_suffix>_participant(?P<participant>.+))?\.txt$"
)
_JSON_REVIEW_NAME_RE = re.compile(r"^iter(?P<iteration>-?\d+)(?P<tag>.*?)\.json$")
_JSON_TRACE_NAME_RE = re.compile(
    r"^iter(?P<iteration>-?\d+)(?P<tag>.*?)_run(?P<run_idx>-?\d+)(?P<participant_suffix>_participant(?P<participant>.+))?\.json$"
)


class DiagnosticsModelRowsState:
    """Availability wrapper for iteration-scoped model rows."""

    def __init__(self, available: bool, rows: list[dict[str, Any]], iteration: int, message: str | None = None):
        self.available = available
        self.rows = rows
        self.iteration = iteration
        self.message = message


def load_registry_snapshot(results_dir: str | Path) -> dict[str, Any]:
    """Load the canonical registry snapshot from DuckDB only."""

    registry_path = Path(results_dir) / "shared_registry.duckdb"
    return SharedRegistry.open_existing(registry_path).read()


def summary_stats(data: dict[str, Any]) -> dict[str, int]:
    """Summarise client runtime states for the dashboard shell."""

    client_entries = data.get("client_entries") or {}
    stats = {"complete": 0, "running": 0, "errors": 0, "recovery_failed": 0}

    for entry in client_entries.values():
        status = str(entry.get("status") or "").strip().lower()
        if status in {"complete", "complete_no_success"}:
            stats["complete"] += 1
        elif status in {"running", "retrying"}:
            stats["running"] += 1
        elif status == "recovery_failed":
            stats["recovery_failed"] += 1
            stats["errors"] += 1
        elif status in {"error", "failed", "validation_error", "fit_error"}:
            stats["errors"] += 1

    return stats


def _coerce_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _sort_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    if text.lstrip("-").isdigit():
        return (0, int(text))
    return (1, text)


def format_client_update_age(updated_at: Any, *, now: datetime | None = None) -> str:
    """Render an update timestamp as a compact relative age."""

    if updated_at is None:
        return "—"

    parsed: datetime | None = None
    if isinstance(updated_at, datetime):
        parsed = updated_at
    else:
        text = str(updated_at).strip()
        if not text:
            return "—"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return text

    if parsed is None:
        return "—"

    if parsed.tzinfo is not None:
        reference = now or datetime.now(parsed.tzinfo)
    else:
        reference = now or datetime.now()
    delta = reference - parsed
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 60:
        return "just now" if seconds == 0 else f"{seconds}s ago"

    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"

    hours = minutes // 60
    if hours < 24:
        remainder_minutes = minutes % 60
        return f"{hours}h ago" if remainder_minutes == 0 else f"{hours}h {remainder_minutes}m ago"

    days = hours // 24
    remainder_hours = hours % 24
    return f"{days}d ago" if remainder_hours == 0 else f"{days}d {remainder_hours}h ago"


def build_client_df(data: dict[str, Any], *, now: datetime | None = None) -> pd.DataFrame:
    """Build a read-only client summary frame from the registry snapshot."""

    client_entries = data.get("client_entries") or {}
    rows: list[dict[str, Any]] = []
    for client_id, entry in sorted(client_entries.items(), key=lambda item: _sort_key(item[0])):
        entry = entry or {}
        status = str(entry.get("status") or "unknown").strip().lower() or "unknown"
        status_meta = components.status_metadata(status)
        updated = entry.get("updated_at")
        if updated is None:
            updated = entry.get("timestamp")
        activity = entry.get("activity")
        if activity is None:
            activity = entry.get("message")
        last_iteration = entry.get("last_iteration")
        if last_iteration is None:
            last_iteration = entry.get("iteration")
        rows.append(
            {
                "client": client_id,
                "status": status_meta["status"],
                "status label": status_meta["label"],
                "status tone": status_meta["tone"],
                "terminal": bool(status_meta["terminal"]),
                "activity": activity,
                "last iteration": last_iteration,
                "best BIC": _coerce_float(entry.get("best_metric")),
                "updated": updated,
                "updated age": format_client_update_age(updated, now=now),
            }
        )

    columns = [
        "client",
        "status",
        "status label",
        "status tone",
        "terminal",
        "activity",
        "last iteration",
        "best BIC",
        "updated",
        "updated age",
    ]
    return pd.DataFrame(rows, columns=columns, dtype=object)


def build_iteration_df(data: dict[str, Any]) -> pd.DataFrame:
    """Build a compact BIC trajectory frame from registry iteration history."""

    history = data.get("iteration_history") or []
    rows: list[dict[str, Any]] = []

    for entry in history:
        entry = entry or {}
        results = entry.get("results") or []
        valid_rows: list[tuple[float, dict[str, Any]]] = []
        for result in results:
            result = result or {}
            metric_value = _coerce_float(result.get("metric_value"))
            if metric_value is None:
                continue
            valid_rows.append((metric_value, result))

        best_metric: float | None = None
        best_result: dict[str, Any] | None = None
        if valid_rows:
            best_metric, best_result = min(valid_rows, key=lambda item: item[0])

        rows.append(
            {
                "client_id": entry.get("client_id"),
                "iteration": entry.get("iteration"),
                "timestamp": entry.get("timestamp"),
                "n_models": len(valid_rows),
                "best_bic": best_metric,
                "best_model_name": (best_result or {}).get("function_name") or (best_result or {}).get("name"),
                "best_param_names": (best_result or {}).get("param_names") or [],
                "best_result": best_result,
                "has_valid_models": bool(valid_rows),
            }
        )

    columns = [
        "client_id",
        "iteration",
        "timestamp",
        "n_models",
        "best_bic",
        "best_model_name",
        "best_param_names",
        "best_result",
        "has_valid_models",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.sort_values(by=["iteration", "client_id"], kind="stable", ignore_index=True)
    return frame


def _model_status_token(status: Any) -> str:
    return components.normalize_status(status)


_RANKABLE_STATUSES = {"complete", "success"}


def _is_rankable_model_row(status: Any, metric_value: Any) -> bool:
    token = _model_status_token(status)
    if token not in _RANKABLE_STATUSES:
        return False
    metric = _coerce_float(metric_value)
    return metric is not None and math.isfinite(metric)


def _model_dashboard_key(row: dict[str, Any]) -> str:
    if row.get("dashboard_model_key"):
        return str(row["dashboard_model_key"])
    source_db = row.get("source_db")
    model_id = row.get("model_id")
    split = row.get("split")
    if source_db is not None and model_id is not None and split is not None:
        return f"{source_db}::{model_id}::{split}"
    client_id = row.get("client_id")
    iteration = row.get("iteration")
    result_index = row.get("result_index")
    if client_id is not None and iteration is not None and result_index is not None:
        return f"registry::{client_id}::{iteration}::{result_index}"
    name = row.get("name") or row.get("function_name") or "model"
    return str(name)


def model_dashboard_key(row: dict[str, Any]) -> str:
    """Public model key helper shared with dashboard views."""

    return _model_dashboard_key(row)


def _registry_model_rows_from_snapshot(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    history = data.get("iteration_history") or []

    for entry in history:
        entry = entry or {}
        client_id = entry.get("client_id")
        iteration = entry.get("iteration")
        timestamp = entry.get("timestamp")
        results = entry.get("results") or []
        for result_index, result in enumerate(results):
            result = result or {}
            metric_value = _coerce_float(result.get("metric_value"))
            detail = {
                "code": result.get("code"),
                "validation_errors": result.get("validation_errors") or [],
                "parameter_recovery": result.get("parameter_recovery") or result.get("recovery_summary"),
                "individual_differences": result.get("individual_differences") or result.get("r2_details"),
                "ppc": result.get("ppc") or [],
                "block_residuals": result.get("block_residuals") or [],
                "error": result.get("error") or result.get("error_message"),
                "name": result.get("function_name") or result.get("name"),
                "metric_name": result.get("metric_name") or "BIC",
                "metric_value": result.get("metric_value"),
                "status": result.get("status") or entry.get("status") or "unknown",
                "param_names": _jsonish(result.get("param_names"), []),
                "provenance": {
                    "client_id": client_id,
                    "iteration": iteration,
                    "result_index": result_index,
                    "timestamp": timestamp,
                },
            }
            row = {
                "source_db": "registry",
                "db_path": None,
                "client_id": client_id,
                "iteration": iteration,
                "result_index": result_index,
                "timestamp": timestamp,
                "name": detail["name"],
                "function_name": result.get("function_name") or result.get("name"),
                "metric_name": detail["metric_name"],
                "metric_value": metric_value,
                "mean_r2": _coerce_float(result.get("mean_r2")),
                "max_r2": _coerce_float(result.get("max_r2")),
                "split": result.get("split") or entry.get("split"),
                "status": detail["status"],
                "param_names": _jsonish(result.get("param_names"), []),
                "code": result.get("code"),
                "dashboard_model_key": f"registry::{client_id}::{iteration}::{result_index}",
                "detail": detail,
            }
            rows.append(row)

    frame = pd.DataFrame(rows)
    desired_columns = [
        "dashboard_model_key",
        "source_db",
        "db_path",
        "client_id",
        "iteration",
        "result_index",
        "timestamp",
        "name",
        "function_name",
        "metric_name",
        "metric_value",
        "mean_r2",
        "max_r2",
        "split",
        "status",
        "param_names",
        "code",
        "detail",
    ]
    for column in desired_columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[desired_columns + [column for column in frame.columns if column not in desired_columns]]


def _rank_model_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    ranked = frame.copy()
    if "dashboard_model_key" not in ranked.columns:
        ranked["dashboard_model_key"] = [
            _model_dashboard_key(row)
            for row in ranked.to_dict(orient="records")
        ]
    ranked["_row_order"] = range(len(ranked))
    ranked["_metric_value"] = ranked["metric_value"].map(_coerce_float) if "metric_value" in ranked.columns else None
    ranked["_rankable"] = [
        _is_rankable_model_row(status, metric_value)
        for status, metric_value in zip(ranked.get("status", pd.Series(dtype=object)), ranked.get("metric_value", pd.Series(dtype=object)), strict=False)
    ]
    ranked["_rank_group"] = ranked["_rankable"].map(lambda value: 0 if bool(value) else 1)
    ranked["_sort_metric"] = ranked["_metric_value"].where(ranked["_rankable"], float("inf"))
    sort_columns = ["_rank_group", "_sort_metric", "_row_order"]
    ranked = ranked.sort_values(by=sort_columns, kind="stable", ignore_index=True)

    display_ranks: list[int | None] = []
    next_rank = 1
    for is_rankable in ranked["_rankable"].tolist():
        if bool(is_rankable):
            display_ranks.append(next_rank)
            next_rank += 1
        else:
            display_ranks.append(None)
    ranked["display_rank"] = display_ranks
    return ranked.drop(columns=["_row_order", "_metric_value", "_rankable", "_rank_group", "_sort_metric"], errors="ignore")


def build_model_comparison_frame(
    summary: pd.DataFrame | None,
    *,
    snapshot: dict[str, Any] | None = None,
) -> pd.DataFrame | None:
    """Normalize diagnostics or registry rows for the models comparison view."""

    if summary is not None and not summary.empty:
        frame = summary.copy()
    elif snapshot is not None:
        frame = _registry_model_rows_from_snapshot(snapshot)
    else:
        return None

    if frame.empty:
        return frame

    if "dashboard_model_key" not in frame.columns:
        frame["dashboard_model_key"] = [
            _model_dashboard_key(row)
            for row in frame.to_dict(orient="records")
        ]
    if "detail" not in frame.columns:
        frame["detail"] = None
    if "source_db" not in frame.columns:
        frame["source_db"] = None
    return _rank_model_rows(frame)


def build_overview_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Build a command-center summary from a registry snapshot."""

    stats = summary_stats(data)
    client_frame = build_client_df(data)
    iteration_frame = build_iteration_df(data)
    global_best = data.get("global_best") or {}
    baseline = data.get("baseline") or {}
    tried_param_sets = data.get("tried_param_sets") or []
    global_best_bic = _coerce_float(global_best.get("metric_value"))

    trajectory_bics = iteration_frame.get("best_bic") if not iteration_frame.empty else pd.Series(dtype=float)
    finite_bics = trajectory_bics.dropna() if hasattr(trajectory_bics, "dropna") else pd.Series(dtype=float)
    finite_bics = finite_bics[finite_bics.map(lambda value: value is not None)] if not finite_bics.empty else finite_bics

    best_row: dict[str, Any] | None = None
    if not iteration_frame.empty:
        ranked = iteration_frame[iteration_frame["best_bic"].notna()].copy()
        if not ranked.empty:
            ranked["_sort_iteration"] = pd.to_numeric(ranked["iteration"], errors="coerce")
            ranked["_sort_client"] = pd.to_numeric(ranked["client_id"], errors="coerce")
            best_row = ranked.sort_values(
                by=["best_bic", "_sort_iteration", "_sort_client"],
                kind="stable",
                ignore_index=True,
            ).iloc[0].to_dict()

    best_bic = global_best_bic
    if best_bic is None and best_row is not None:
        best_bic = _coerce_float(best_row.get("best_bic"))

    baseline_bic = _coerce_float(baseline.get("metric_value"))
    bic_delta = None
    bic_delta_pct = None
    if best_bic is not None and baseline_bic is not None:
        bic_delta = baseline_bic - best_bic
        if baseline_bic != 0:
            bic_delta_pct = (bic_delta / abs(baseline_bic)) * 100

    best_model_name = None
    best_client_id = None
    best_iteration = None
    best_param_names: list[Any] = []
    best_source = "unknown"
    best_model_code = None
    if global_best_bic is not None:
        best_source = "global_best"
        best_model_name = "Global best"
        best_client_id = global_best.get("client_id")
        best_iteration = global_best.get("iteration")
        best_param_names = list(global_best.get("param_names") or [])
        best_model_code = global_best.get("model_code")
    elif best_row is not None:
        best_source = "iteration_history"
        best_model_name = best_row.get("best_model_name")
        best_client_id = best_row.get("client_id")
        best_iteration = best_row.get("iteration")
        best_param_names = list(best_row.get("best_param_names") or [])
    elif best_bic is not None:
        best_model_name = "Global best"

    if best_model_name is None and global_best:
        best_model_name = "Global best"

    if best_client_id is None and global_best:
        best_client_id = global_best.get("client_id")
    if best_iteration is None and global_best:
        best_iteration = global_best.get("iteration")
    if not best_param_names and global_best:
        best_param_names = list(global_best.get("param_names") or [])

    if stats["running"] > 0:
        run_state_label = "Running"
        run_state_tone = "info"
    elif stats["errors"] > 0:
        run_state_label = "Needs attention"
        run_state_tone = "warning"
    elif stats["complete"] > 0:
        run_state_label = "Complete"
        run_state_tone = "success"
    else:
        run_state_label = "Idle"
        run_state_tone = "neutral"

    if stats["errors"] > 0:
        health_label = "Needs attention"
        health_tone = "warning"
    elif best_bic is not None:
        health_label = "Healthy"
        health_tone = "success"
    else:
        health_label = "Waiting"
        health_tone = "neutral"

    latest_client_activity = None
    if not client_frame.empty and "updated" in client_frame.columns:
        updated_series = client_frame["updated"].dropna()
        latest_client_activity = updated_series.iloc[-1] if not updated_series.empty else None

    return {
        **stats,
        "client_count": int(len(client_frame)),
        "running_clients": int(stats["running"]),
        "complete_clients": int(stats["complete"]),
        "error_clients": int(stats["errors"]),
        "recovery_failed_clients": int(stats["recovery_failed"]),
        "iteration_count": int(len(iteration_frame)),
        "model_count": int(finite_bics.shape[0]),
        "trajectory_points": int(finite_bics.shape[0]),
        "param_set_count": int(len(tried_param_sets)),
        "best_bic": best_bic,
        "baseline_bic": baseline_bic,
        "bic_delta": bic_delta,
        "bic_delta_pct": bic_delta_pct,
        "best_model_name": best_model_name,
        "best_client_id": best_client_id,
        "best_iteration": best_iteration,
        "best_param_names": best_param_names,
        "best_model_code": best_model_code,
        "best_source": best_source,
        "best_provenance": global_best if best_source == "global_best" else (best_row or {}),
        "has_global_best": best_bic is not None,
        "has_baseline": baseline_bic is not None,
        "run_state_label": run_state_label,
        "run_state_tone": run_state_tone,
        "health_label": health_label,
        "health_tone": health_tone,
        "latest_client_activity": latest_client_activity,
        "client_frame": client_frame,
        "iteration_frame": iteration_frame,
    }


def _diagnostic_db_paths(results_dir: str | Path) -> list[Path]:
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    unified_path = results_dir / "diagnostics_unified.duckdb"
    if unified_path.exists():
        return [unified_path]

    paths: list[Path] = []
    primary = results_dir / "diagnostics.duckdb"
    if primary.exists():
        paths.append(primary)

    for path in sorted(results_dir.glob("diagnostics_*.duckdb")):
        if path.name != unified_path.name and path not in paths:
            paths.append(path)

    return paths


def _sql_fetch_dataframe(db_path: Path, sql: str, params: Iterable[Any] | None = None) -> pd.DataFrame:
    try:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            cursor = conn.execute(sql, list(params or []))
            return cursor.fetchdf()
    except Exception:
        return pd.DataFrame()


def _sql_fetchone_dict(db_path: Path, sql: str, params: Iterable[Any] | None = None) -> dict[str, Any] | None:
    try:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            cursor = conn.execute(sql, list(params or []))
            row = cursor.fetchone()
            if row is None:
                return None
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
    except Exception:
        return None


def _load_individual_differences(
    db_path: Path,
    model_id: Any,
    split: Any,
) -> dict[str, Any] | None:
    if split is None:
        sql = (
            "SELECT mean_r2, max_r2, best_param, per_param_r2, per_param_detail, split "
            "FROM individual_differences WHERE model_id = ? ORDER BY split LIMIT 1"
        )
        params: list[Any] = [model_id]
    else:
        sql = (
            "SELECT mean_r2, max_r2, best_param, per_param_r2, per_param_detail, split "
            "FROM individual_differences WHERE model_id = ? AND split = ? LIMIT 1"
        )
        params = [model_id, split]

    return _sql_fetchone_dict(db_path, sql, params)


def _jsonish(value: Any, default: Any) -> Any:
    if value is None:
        return default
    return value


def _relative_artifact_path(results_dir: Path, artifact_path: Path) -> str:
    try:
        return str(artifact_path.relative_to(results_dir))
    except ValueError:
        return artifact_path.name


def _parse_name(pattern: re.Pattern[str], artifact_path: Path) -> dict[str, Any]:
    match = pattern.match(artifact_path.name)
    if match is None:
        return {}
    data = match.groupdict()
    parsed: dict[str, Any] = {
        "iteration": int(data["iteration"]),
        "tag": data.get("tag") or "",
    }
    if data.get("run_idx") is not None:
        parsed["run_idx"] = int(data["run_idx"])
    if data.get("participant") is not None:
        parsed["participant"] = data["participant"]
    return parsed


def _artifact_row(
    results_dir: Path,
    artifact_path: Path,
    *,
    kind: str,
    source: str,
    content: str | None = None,
    payload: Any = None,
    raw_text: str | None = None,
    error: str | None = None,
    pattern: re.Pattern[str] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "kind": kind,
        "source": source,
        "path": _relative_artifact_path(results_dir, artifact_path),
        "absolute_path": str(artifact_path),
        "content": content,
        "payload": payload,
        "raw_text": raw_text,
        "error": error,
    }
    if pattern is not None:
        row.update(_parse_name(pattern, artifact_path))
    return row


def _load_text_artifact(results_dir: Path, artifact_path: Path, *, kind: str, source: str) -> dict[str, Any]:
    text = artifact_path.read_text(encoding="utf-8")
    return _artifact_row(results_dir, artifact_path, kind=kind, source=source, content=text, pattern=_TXT_ARTIFACT_NAME_RE)


def _load_json_artifact(results_dir: Path, artifact_path: Path, *, kind: str, source: str, pattern: re.Pattern[str]) -> dict[str, Any]:
    raw_text = artifact_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return _artifact_row(
            results_dir,
            artifact_path,
            kind=kind,
            source=source,
            raw_text=raw_text,
            error=str(exc),
            pattern=pattern,
        )

    row = _artifact_row(results_dir, artifact_path, kind=kind, source=source, payload=payload, raw_text=raw_text, pattern=pattern)
    if kind == "judge_trace":
        row["trace"] = payload.get("trace", payload.get("tool_call_trace")) if isinstance(payload, dict) else None
        row["full_trace"] = payload.get("full_trace") if isinstance(payload, dict) else None
        row["synthesized_feedback"] = payload.get("synthesized_feedback") if isinstance(payload, dict) else None
        row["verdict"] = payload.get("verdict") if isinstance(payload, dict) else None
    return row


def load_feedback_artifacts(results_dir: str | Path, *, limit: int = 25) -> list[dict[str, Any]]:
    """Load read-only feedback and raw LLM inspection artifacts."""

    if limit <= 0:
        return []

    results_dir = Path(results_dir)
    rows: list[dict[str, Any]] = []
    artifact_specs = [
        (results_dir / "feedback", "feedback", "feedback", "iter*_run*.txt", _load_text_artifact, None),
        (results_dir / "models", "model_code", "models", "iter*_run*.txt", _load_text_artifact, None),
        (results_dir / "reviews", "review", "reviews", "*.json", _load_json_artifact, _JSON_REVIEW_NAME_RE),
    ]

    for directory, kind, source, pattern, loader, json_pattern in artifact_specs:
        if not directory.exists():
            continue
        for artifact_path in sorted(directory.glob(pattern)):
            if len(rows) >= limit:
                return rows
            if loader is _load_json_artifact:
                row = loader(results_dir, artifact_path, kind=kind, source=source, pattern=json_pattern)  # type: ignore[arg-type]
            else:
                row = loader(results_dir, artifact_path, kind=kind, source=source)  # type: ignore[misc]
            rows.append(row)
    return rows


def load_judge_trace_artifacts(results_dir: str | Path, *, limit: int = 25) -> list[dict[str, Any]]:
    """Load read-only judge trace artifacts from judge/*.json."""

    if limit <= 0:
        return []

    results_dir = Path(results_dir)
    judge_dir = results_dir / "judge"
    if not judge_dir.exists():
        return []

    rows: list[dict[str, Any]] = []
    for artifact_path in sorted(judge_dir.glob("iter*_run*.json")):
        if len(rows) >= limit:
            break
        row = _load_json_artifact(results_dir, artifact_path, kind="judge_trace", source="judge", pattern=_JSON_TRACE_NAME_RE)
        rows.append(row)
    return rows


def _build_detail_from_row(row: dict[str, Any], db_path: Path) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "model_id": row.get("model_id"),
        "iteration_id": row.get("iteration_id"),
        "run_idx": row.get("run_idx"),
        "iteration": row.get("iteration"),
        "name": row.get("name"),
        "code": row.get("code"),
        "metric_name": row.get("metric_name"),
        "metric_value": row.get("metric_value"),
        "mean_nll": row.get("mean_nll"),
        "split": row.get("split"),
        "status": row.get("status"),
        "param_names": _jsonish(row.get("param_names"), []),
        "source_db": db_path.name,
        "db_path": str(db_path),
    }

    recovery = _sql_fetchone_dict(
        db_path,
        "SELECT passed, mean_r, n_successful, per_param_r, simulation_error "
        "FROM parameter_recovery WHERE model_id = ?",
        [row.get("model_id")],
    )
    if recovery is not None:
        recovery["per_param_r"] = _jsonish(recovery.get("per_param_r"), {})
    detail["parameter_recovery"] = recovery

    individual = _load_individual_differences(db_path, row.get("model_id"), row.get("split"))
    if individual is not None:
        individual["per_param_r2"] = _jsonish(individual.get("per_param_r2"), {})
        individual["per_param_detail"] = _jsonish(individual.get("per_param_detail"), {})
    detail["individual_differences"] = individual

    validation_errors = _sql_fetch_dataframe(
        db_path,
        "SELECT error_type, error_message, error_details FROM validation_errors WHERE model_id = ?",
        [row.get("model_id")],
    )
    if not validation_errors.empty:
        detail["validation_errors"] = validation_errors.to_dict(orient="records")
    else:
        detail["validation_errors"] = []

    ppc_rows = _sql_fetch_dataframe(
        db_path,
        "SELECT participant_id, statistic_name, condition, observed, simulated_mean, "
        "simulated_q025, simulated_q975, n_sims FROM ppc WHERE model_id = ? ORDER BY ppc_id",
        [row.get("model_id")],
    )
    detail["ppc"] = ppc_rows.to_dict(orient="records") if not ppc_rows.empty else []

    block_residual_rows = _sql_fetch_dataframe(
        db_path,
        "SELECT participant_id, block_idx, block_start, block_end, mean_nll_per_trial, n_trials "
        "FROM block_residuals WHERE model_id = ? ORDER BY id",
        [row.get("model_id")],
    )
    detail["block_residuals"] = (
        block_residual_rows.to_dict(orient="records") if not block_residual_rows.empty else []
    )

    return detail


def _models_from_db(db_path: Path) -> list[dict[str, Any]]:
    frame = _sql_fetch_dataframe(
        db_path,
        "SELECT m.model_id, m.iteration_id, it.run_idx, m.iteration, m.name, m.code, "
        "m.metric_name, m.metric_value, m.mean_nll, m.split, m.param_names, m.status, "
        "it.client_id AS iteration_client_id, it.tag AS iteration_tag, "
        "it.timestamp AS iteration_timestamp, it.n_models_proposed, "
        "dif.mean_r2, dif.max_r2, dif.best_param, dif.per_param_r2, dif.per_param_detail, "
        "dif.split AS individual_differences_split "
        "FROM models m "
        "LEFT JOIN iterations it ON it.iteration_id = m.iteration_id "
        "LEFT JOIN individual_differences dif ON dif.model_id = m.model_id AND dif.split = m.split "
        "ORDER BY m.model_id, m.split",
    )
    if frame.empty:
        return []

    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        detail = _build_detail_from_row(row, db_path)
        summary_row = {
            **row,
            "source_db": db_path.name,
            "db_path": str(db_path),
            "dashboard_model_key": f"{db_path.name}::{row.get('model_id')}::{row.get('split')}",
            "detail": detail,
            "param_names": _jsonish(row.get("param_names"), []),
            "per_param_r2": _jsonish(row.get("per_param_r2"), {}),
            "per_param_detail": _jsonish(row.get("per_param_detail"), {}),
        }
        rows.append(summary_row)
    return rows


def load_diagnostics_summary(results_dir: str | Path) -> pd.DataFrame | None:
    """Load a combined diagnostics summary from read-only DuckDB files."""

    rows: list[dict[str, Any]] = []
    for db_path in _diagnostic_db_paths(results_dir):
        rows.extend(_models_from_db(db_path))

    if not rows:
        if _diagnostic_db_paths(results_dir):
            return pd.DataFrame(
                columns=[
                    "model_id",
                    "iteration_id",
                    "run_idx",
                    "iteration",
                    "name",
                    "code",
                    "metric_name",
                    "metric_value",
                    "mean_nll",
                    "split",
                    "param_names",
                    "status",
                    "source_db",
                    "db_path",
                    "dashboard_model_key",
                    "detail",
                ]
            )
        return None

    frame = pd.DataFrame(rows)
    desired_columns = [
        "model_id",
        "iteration_id",
        "run_idx",
        "iteration",
        "name",
        "code",
        "metric_name",
        "metric_value",
        "mean_nll",
        "split",
        "param_names",
        "status",
        "source_db",
        "db_path",
        "dashboard_model_key",
        "detail",
    ]
    for column in desired_columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[desired_columns + [column for column in frame.columns if column not in desired_columns]]


def load_diagnostics_model_rows(results_dir: str | Path, iteration: int) -> list[dict[str, Any]]:
    """Return model rows for a single iteration across all diagnostics stores."""

    summary = load_diagnostics_summary(results_dir)
    if summary is None or summary.empty:
        return []

    iteration_rows = summary[summary["iteration"] == iteration]
    return iteration_rows.to_dict(orient="records")


def load_diagnostics_model_rows_state(results_dir: str | Path, iteration: int) -> DiagnosticsModelRowsState:
    """Return availability state for iteration-scoped diagnostics rows."""

    paths = _diagnostic_db_paths(results_dir)
    rows = load_diagnostics_model_rows(results_dir, iteration)
    return DiagnosticsModelRowsState(available=bool(paths), rows=rows, iteration=iteration)


def load_model_detail(
    results_dir: str | Path,
    model_id: int,
    *,
    db_path: str | Path | None = None,
    split: str | None = None,
) -> dict[str, Any] | None:
    """Load the full detail payload for a single model row."""

    results_dir = Path(results_dir)
    candidate_paths = [Path(db_path)] if db_path is not None else _diagnostic_db_paths(results_dir)

    base_sql = (
        "SELECT m.model_id, m.iteration_id, it.run_idx, m.iteration, m.name, m.code, "
        "m.metric_name, m.metric_value, m.mean_nll, m.split, m.param_names, m.status, "
        "it.client_id AS iteration_client_id, it.tag AS iteration_tag, "
        "it.timestamp AS iteration_timestamp, it.n_models_proposed, "
        "dif.mean_r2, dif.max_r2, dif.best_param, dif.per_param_r2, dif.per_param_detail, "
        "dif.split AS individual_differences_split "
        "FROM models m "
        "LEFT JOIN iterations it ON it.iteration_id = m.iteration_id "
        "LEFT JOIN individual_differences dif ON dif.model_id = m.model_id AND dif.split = m.split "
        "WHERE m.model_id = ?"
    )
    params: list[Any] = [model_id]
    if split is not None:
        base_sql += " AND m.split = ?"
        params.append(split)
    base_sql += " ORDER BY m.model_id, m.split LIMIT 1"

    for path in candidate_paths:
        row = _sql_fetchone_dict(path, base_sql, params)
        if row is not None:
            detail = _build_detail_from_row(row, path)
            if split is not None:
                detail["split"] = split
            return detail

    return None


def normalize_judge_iterations(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return registry judge iterations in iteration order."""

    judge_iterations = data.get("judge_iterations") or {}

    def _sort_key(item: tuple[str, Any]) -> tuple[int, Any]:
        key, _ = item
        key_text = str(key)
        if key_text.lstrip("-").isdigit():
            return (0, int(key_text))
        return (1, key_text)

    rows: list[dict[str, Any]] = []
    for key, value in sorted(judge_iterations.items(), key=_sort_key):
        payload = value or {}
        feedback = payload.get("synthesized_feedback")
        verdict = payload.get("verdict")
        rows.append(
            {
                "iteration": int(key) if str(key).lstrip("-").isdigit() else key,
                "synthesized_feedback": feedback,
                "verdict": verdict,
                "failed": bool(payload.get("failed", False)),
                "timestamp": payload.get("timestamp"),
                "error": payload.get("error"),
                "has_feedback": feedback is not None or verdict is not None,
            }
        )
    return rows
