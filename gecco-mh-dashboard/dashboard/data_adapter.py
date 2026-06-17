from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gecco.coordination import SharedRegistry

# BIC filtering configuration for dashboard display
BIC_PERCENTILE = 95  # Show up to 95th percentile
BIC_ABSOLUTE_CAP = 1000  # Cap displayed BIC at this absolute value


def _cap_bic_outliers(bic_values: list[float | None]) -> float | None:
    """Calculate BIC cap based on 95th percentile or absolute limit."""
    valid_bics = [b for b in bic_values if b is not None and b < float("inf")]
    if not valid_bics:
        return None
    percentile_val = float(np.percentile(valid_bics, BIC_PERCENTILE))
    return min(percentile_val, BIC_ABSOLUTE_CAP)


def _apply_bic_cap(
    rows: list[dict[str, Any]], key: str = "BIC"
) -> list[dict[str, Any]]:
    """Cap BIC values in rows for display. Does not modify the original data."""
    if not rows:
        return rows
    bic_values = [r.get(key) for r in rows]
    cap = _cap_bic_outliers(bic_values)
    if cap is None:
        return rows
    for row in rows:
        bic = row.get(key)
        if bic is not None and bic > cap:
            row[key] = cap
    return rows


def load_registry_snapshot(results_dir: Path) -> dict[str, Any] | None:
    """Load the canonical DuckDB registry snapshot for the dashboard."""
    registry_path = results_dir / "shared_registry.duckdb"
    if not registry_path.exists():
        return None
    try:
        return SharedRegistry.open_existing(registry_path).read()
    except (FileNotFoundError, OSError):
        return None


def _age_from_iso(timestamp: str | None) -> str:
    if not timestamp:
        return "-"

    try:
        dt = datetime.fromisoformat(timestamp)
        age = datetime.now() - dt
        sec = int(age.total_seconds())
        if sec < 60:
            return f"{sec}s"
        if sec < 3600:
            return f"{sec // 60}m"
        return f"{sec // 3600}h {(sec % 3600) // 60}m"
    except (TypeError, ValueError):
        return timestamp


def build_client_df(data: dict[str, Any]) -> pd.DataFrame:
    entries = data.get("client_entries", {})
    rows: list[dict[str, Any]] = []

    def _sort_key(k: str) -> tuple:
        # Numeric IDs first (sorted numerically), then names (alphabetically)
        if str(k).isdigit():
            return (0, int(k), "")
        return (1, 0, str(k))

    for cid in sorted(entries.keys(), key=_sort_key):
        e = entries[cid]
        rows.append(
            {
                "Client": cid,
                "Status": e.get("status", "unknown"),
                "Activity": e.get("activity", "-"),
                "Last Iter": e.get("last_iteration"),
                "Best BIC": e.get("best_metric"),
                "Updated": _age_from_iso(e.get("updated_at")),
            }
        )

    return pd.DataFrame(rows)


def _fill_r2_from_per_param(entry: dict[str, Any]) -> tuple[Any, Any]:
    """Compute max_r2 and best_param from per_param_r2 if not already present."""
    max_r2 = entry.get("max_r2")
    best_param = entry.get("best_param")
    if max_r2 is None:
        per_param = entry.get("per_param_r2") or {}
        if per_param:
            best_param = max(per_param, key=per_param.get)
            max_r2 = per_param[best_param]
    return max_r2, best_param


def build_baseline_row(data: dict[str, Any]) -> pd.DataFrame | None:
    """Build a single-row DataFrame for the baseline model, or None if unavailable."""
    baseline = data.get("baseline") or {}
    if baseline.get("metric_value") is None:
        return None
    max_r2, best_param = _fill_r2_from_per_param(baseline)
    row = {
        "Model": baseline.get("function_name", "baseline_model"),
        "BIC": baseline["metric_value"],
        "Max R²": max_r2,
        "Best Param": best_param,
        "Mean R²": baseline.get("mean_r2"),
        "Params": ", ".join(baseline.get("param_names", [])),
        "Client": "baseline",
        "Iteration": 0,
    }
    return pd.DataFrame([row])


def build_landscape_df(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for entry in data.get("iteration_history", []):
        cid = entry.get("client_id")
        it = entry.get("iteration")
        for r in entry.get("results", []):
            bic = r.get("metric_value")
            mn = r.get("metric_name", "BIC")

            # Determine status
            if mn == "RECOVERY_FAILED":
                status = "recovery_failed"
            elif mn in ("FIT_ERROR", "VALIDATION_ERROR"):
                status = "error"
            elif bic is not None and bic < float("inf"):
                status = "success"
            else:
                status = "error"  # Catch-all for unexpected states

            max_r2, best_param = _fill_r2_from_per_param(r)
            rows.append(
                {
                    "Model": r.get("function_name", "?"),
                    "Status": status,
                    "BIC": bic if bic is not None and bic < float("inf") else None,
                    "Max R²": max_r2,
                    "Best Param": best_param,
                    "Mean R²": r.get("mean_r2"),
                    "Params": ", ".join(r.get("param_names", [])),
                    "Client": cid,
                    "Iteration": it,
                    "Error": r.get("error") or r.get("error_message"),
                    "Recovery R": r.get("recovery_r"),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
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
        )

    rows = _apply_bic_cap(rows)

    # Sort: success first (by BIC), then recovery_failed, then errors
    def sort_key(row):
        status_order = {"success": 0, "recovery_failed": 1, "error": 2}
        bic = row.get("BIC")
        return (
            status_order.get(row["Status"], 2),
            bic if bic is not None else float("inf"),
        )

    rows = sorted(rows, key=sort_key)
    return pd.DataFrame(rows).reset_index(drop=True)


def build_iteration_df(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in data.get("iteration_history", []):
        cid = entry.get("client_id")
        it = entry.get("iteration")
        bics = [
            r.get("metric_value")
            for r in entry.get("results", [])
            if r.get("metric_value") is not None
        ]
        if not bics:
            continue
        rows.append(
            {"Client": str(cid), "Iteration": int(it), "Best BIC": float(min(bics))}
        )

    if not rows:
        return pd.DataFrame(columns=["Client", "Iteration", "Best BIC"])

    rows = _apply_bic_cap(rows, key="Best BIC")
    return pd.DataFrame(rows).sort_values(["Client", "Iteration"])


def build_r2_df(data: dict[str, Any], top_n: int = 8) -> pd.DataFrame:
    candidates: list[dict[str, Any]] = []

    # Include baseline if it has per-param R² data
    baseline = data.get("baseline") or {}
    baseline_per_param = baseline.get("per_param_r2")
    if baseline_per_param:
        candidates.append(
            {
                "Model": baseline.get("function_name", "baseline_model"),
                "Client": "baseline",
                "BIC": baseline.get("metric_value"),
                "Max R²": baseline.get("max_r2"),
                "Mean R²": baseline.get("mean_r2"),
                "per_param": baseline_per_param,
            }
        )

    for entry in data.get("iteration_history", []):
        for r in entry.get("results", []):
            per_param = r.get("per_param_r2")
            if not per_param:
                continue
            candidates.append(
                {
                    "Model": r.get("function_name", "?"),
                    "Client": entry.get("client_id", "?"),
                    "BIC": r.get("metric_value"),
                    "Max R²": r.get("max_r2"),
                    "Mean R²": r.get("mean_r2"),
                    "per_param": per_param,
                }
            )

    if not candidates:
        return pd.DataFrame()

    candidates = sorted(
        candidates, key=lambda x: x["BIC"] if x["BIC"] is not None else float("inf")
    )[:top_n]
    all_params: list[str] = []
    for c in candidates:
        for p in c["per_param"].keys():
            if p not in all_params:
                all_params.append(p)

    rows = []
    for c in candidates:
        row = {
            "Model": c["Model"],
            "Client": c["Client"],
            "BIC": c["BIC"],
            "Max R²": c["Max R²"],
            "Mean R²": c["Mean R²"],
        }
        for p in all_params:
            row[p] = c["per_param"].get(p)
        rows.append(row)

    rows = _apply_bic_cap(rows)
    return pd.DataFrame(rows)


def summary_stats(data: dict[str, Any]) -> dict[str, int]:
    entries = data.get("client_entries", {})
    history = data.get("iteration_history", [])
    total_models = 0
    recovery_failed = 0
    errors = 0
    for h in history:
        for r in h.get("results", []):
            total_models += 1
            mn = r.get("metric_name", "BIC")
            if mn == "RECOVERY_FAILED":
                recovery_failed += 1
            elif mn in ("FIT_ERROR", "VALIDATION_ERROR"):
                errors += 1
    return {
        "n_clients": len(entries),
        "running": sum(1 for e in entries.values() if e.get("status") == "running"),
        "complete": sum(
            1
            for e in entries.values()
            if e.get("status") in ("complete", "complete_no_success")
        ),
        "iterations": len(history),
        "models": total_models,
        "recovery_failed": recovery_failed,
        "errors": errors,
        "failed": recovery_failed + errors,  # Keep for backward compatibility
        "param_combos": len(data.get("tried_param_sets", [])),
    }


# ============================================================
# Results browser helpers
# ============================================================


def list_iterations(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract all (client, iteration) entries from iteration_history, sorted.

    Duplicate (client, iteration) pairs from re-runs are preserved and
    disambiguated with a 'run' index (0-based) and 'history_idx' for lookup.
    """
    seen: list[dict[str, Any]] = []
    dup_counts: dict[tuple, int] = {}
    for idx, entry in enumerate(data.get("iteration_history", [])):
        cid = entry.get("client_id")
        it = entry.get("iteration")
        key = (cid, it)
        run = dup_counts.get(key, 0)
        dup_counts[key] = run + 1
        n_models = len(entry.get("results", []))
        bics = [
            r.get("metric_value")
            for r in entry.get("results", [])
            if r.get("metric_value") is not None
        ]
        seen.append(
            {
                "client_id": cid,
                "iteration": it,
                "run": run,
                "history_idx": idx,
                "n_models": n_models,
                "best_bic": min(bics) if bics else None,
            }
        )
    return sorted(
        seen, key=lambda x: (str(x["client_id"] or ""), x["iteration"] or 0, x["run"])
    )


def get_iteration_results_by_idx(
    data: dict[str, Any], history_idx: int
) -> list[dict[str, Any]]:
    """Get model results by history index (handles duplicates unambiguously)."""
    history = data.get("iteration_history", [])
    if 0 <= history_idx < len(history):
        return history[history_idx].get("results", [])
    return []


def get_model_code(
    data: dict[str, Any], model_name: str, client_id: Any, iteration: Any
) -> str | None:
    """Look up a model's code from iteration_history by name, client, and iteration."""
    for entry in data.get("iteration_history", []):
        if entry.get("client_id") == client_id and entry.get("iteration") == iteration:
            for r in entry.get("results", []):
                if r.get("function_name") == model_name:
                    return r.get("code")
    return None


def load_text_file(results_dir: Path, subdir: str, pattern: str) -> str | None:
    """Try to read a text file matching pattern from results_dir/subdir/. Returns None if not found."""
    target_dir = results_dir / subdir
    if not target_dir.exists():
        return None
    # Try exact match first
    exact = target_dir / pattern
    if exact.exists():
        try:
            return exact.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    # Try glob
    matches = sorted(target_dir.glob(pattern))
    if matches:
        try:
            return matches[0].read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    return None


def load_json_file(results_dir: Path, subdir: str, pattern: str) -> list | dict | None:
    """Try to read a JSON file matching pattern from results_dir/subdir/."""
    text = load_text_file(results_dir, subdir, pattern)
    if text is None:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


_ITER_REGEX = re.compile(r"iter(\d+)(.*?)_run(\d+)\.json")


def list_judge_traces(results_dir: Path) -> list[dict[str, Any]]:
    """Scan judge/ subdirectory for trace files.

    Returns list of dicts sorted by (iteration, run_idx), each containing:
    - iteration: int
    - run_idx: int
    - tag: str
    - timestamp: str (ISO)
    - tool_call_count: int
    - wall_time_seconds: float
    - short_circuit: bool
    - source_iter: int | None (only if short_circuit)
    - file_path: Path (for loading)
    """
    judge_dir = results_dir / "judge"
    if not judge_dir.is_dir():
        return []
    traces: list[dict[str, Any]] = []
    for f in sorted(judge_dir.glob("iter*_run*.json")):
        m = _ITER_REGEX.match(f.name)
        if not m:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        traces.append(
            {
                "iteration": int(m.group(1)),
                "tag": m.group(2),
                "run_idx": int(m.group(3)),
                "timestamp": data.get("timestamp", ""),
                "tool_call_count": data.get("tool_call_count", 0),
                "wall_time_seconds": data.get("wall_time_seconds", 0.0),
                "short_circuit": data.get("short_circuit", False),
                "source_iter": data.get("source_iter"),
                "file_path": f,
            }
        )
    return traces


# ============================================================
# Diagnostics adapter helpers (read-only)
# ============================================================


def _is_diagnostics_db(filename: str) -> bool:
    return filename.startswith("diagnostics") and filename.endswith(".duckdb")


def _open_diagnostics_read_only(db_path: Path):
    """Open a diagnostics DuckDB read-only without initialising schema."""
    import duckdb

    return duckdb.connect(str(db_path), read_only=True)


def find_diagnostics_dbs(results_dir: Path) -> list[Path]:
    """Return sorted list of diagnostics*.duckdb paths in results_dir."""
    if not results_dir.is_dir():
        return []
    return sorted(
        results_dir / f
        for f in sorted(results_dir.iterdir())
        if _is_diagnostics_db(f.name)
    )


def load_diagnostics_summary(results_dir: Path) -> pd.DataFrame | None:
    """Return a split-aware model summary from diagnostics*.duckdb.

    Returns None when no diagnostics DuckDB is found (JSON-only results dirs).
    """
    dbs = find_diagnostics_dbs(results_dir)
    if not dbs:
        return None

    rows: list[dict] = []
    for db_path in dbs:
        conn = None
        try:
            conn = _open_diagnostics_read_only(db_path)
            cursor = conn.execute(
                """
                SELECT
                    m.model_id,
                    m.iteration,
                    m.name,
                    m.metric_name,
                    m.metric_value,
                    m.split,
                    m.status,
                    m.mean_nll,
                    id.mean_r2,
                    id.max_r2,
                    id.best_param,
                    id.per_param_r2,
                    pr.passed AS recovery_passed,
                    pr.mean_r AS recovery_mean_r
                FROM models m
                LEFT JOIN individual_differences id ON m.model_id = id.model_id AND m.split = id.split
                LEFT JOIN parameter_recovery pr ON m.model_id = pr.model_id
                ORDER BY m.model_id
                """
            )
            cols = [d[0] for d in cursor.description]
            for row in cursor.fetchall():
                row_dict = dict(zip(cols, row))
                row_dict["db_path"] = str(db_path)
                row_dict["source_db"] = db_path.name
                row_dict["dashboard_model_key"] = (
                    f"{db_path.name}::{row_dict.get('model_id')}::{row_dict.get('split')}"
                )
                rows.append(row_dict)
        except Exception:
            return None
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    df = pd.DataFrame(rows)
    if "_duckdb_internal" in df.columns:
        df = df.drop(columns=["_duckdb_internal"])
    return df


class DiagnosticsModelRowsState:
    __slots__ = ("available", "rows")

    def __init__(self, available: bool, rows: list[dict[str, Any]]):
        self.available = available
        self.rows = rows


def load_model_detail(
    results_dir: Path,
    model_id: int,
    db_path: str | Path | None = None,
    split: str | None = None,
) -> dict | None:
    """Load full detail for a specific diagnostics model.

    When db_path is provided, the lookup is restricted to that diagnostics DB
    so colliding model_id values across sharded databases remain stable.
    """
    dbs = [Path(db_path)] if db_path is not None else find_diagnostics_dbs(results_dir)
    if not dbs:
        return None

    for db_file in dbs:
        db_file = Path(db_file)
        if not db_file.is_absolute():
            db_file = results_dir / db_file
        conn = _open_diagnostics_read_only(db_file)
        try:
            query = """
                SELECT
                    m.model_id,
                    m.iteration,
                    m.name,
                    m.code,
                    m.metric_name,
                    m.metric_value,
                    m.split,
                    m.status,
                    m.param_names,
                    m.mean_nll,
                    id.mean_r2,
                    id.max_r2,
                    id.best_param,
                    id.per_param_r2,
                    pr.passed AS recovery_passed,
                    pr.mean_r AS recovery_mean_r,
                    pr.per_param_r AS recovery_per_param_r,
                    pr.simulation_error
                FROM models m
                LEFT JOIN individual_differences id ON m.model_id = id.model_id AND m.split = id.split
                LEFT JOIN parameter_recovery pr ON m.model_id = pr.model_id
                WHERE m.model_id = ?
            """
            params: list[Any] = [model_id]
            if split is not None:
                query += " AND m.split = ?"
                params.append(split)
            cursor = conn.execute(
                query,
                params,
            )
            row = cursor.fetchone()
            if row:
                cols = [d[0] for d in cursor.description]
                detail = dict(zip(cols, row))
                detail["db_path"] = str(db_file)
                detail["source_db"] = db_file.name
                detail["dashboard_model_key"] = (
                    f"{db_file.name}::{detail.get('model_id')}::{detail.get('split')}"
                )
                return detail
        finally:
            conn.close()

    return None


def load_diagnostics_model_rows(
    results_dir: Path, iteration: int | None = None
) -> list[dict[str, Any]]:
    """Prepare diagnostics-backed model rows with stable identity and details."""
    return load_diagnostics_model_rows_state(results_dir, iteration).rows


def load_diagnostics_model_rows_state(
    results_dir: Path, iteration: int | None = None
) -> DiagnosticsModelRowsState:
    """Return diagnostics availability plus resolved model rows.

    available=True means diagnostics DuckDBs were found and read successfully,
    even if no rows match the current iteration/filter selection.
    """
    summary = load_diagnostics_summary(results_dir)
    if summary is None:
        return DiagnosticsModelRowsState(available=False, rows=[])

    if iteration is not None:
        summary = summary[summary["iteration"] == iteration]

    if summary.empty:
        return DiagnosticsModelRowsState(available=True, rows=[])

    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        row_dict = row.to_dict()
        detail = load_model_detail(
            results_dir,
            int(row_dict["model_id"]),
            db_path=row_dict.get("db_path"),
            split=row_dict.get("split"),
        )
        row_dict["detail"] = detail
        rows.append(row_dict)
    return DiagnosticsModelRowsState(available=True, rows=rows)


# ============================================================
# Judge registry adapter helpers
# ============================================================


def normalize_judge_iterations(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise registry judge_iterations into display rows.

    Works entirely from the DuckDB registry snapshot — no JSON trace files required.
    """
    judge_data = data.get("judge_iterations", {})
    rows: list[dict[str, Any]] = []
    for iter_key, entry in judge_data.items():
        try:
            iteration = int(iter_key)
        except (ValueError, TypeError):
            continue
        feedback = entry.get("synthesized_feedback")
        rows.append(
            {
                "iteration": iteration,
                "failed": bool(entry.get("failed", False)),
                "has_feedback": feedback is not None,
                "verdict": entry.get("verdict"),
                "error": entry.get("error"),
                "timestamp": entry.get("timestamp"),
            }
        )
    return sorted(rows, key=lambda r: r["iteration"])


def load_judge_trace(
    results_dir: Path,
    iteration: int,
    run_idx: int,
    tag: str = "",
) -> dict[str, Any] | None:
    """Load and parse a specific judge trace JSON file by iteration/run_idx/tag."""
    judge_dir = results_dir / "judge"
    fname = judge_dir / f"iter{iteration}{tag}_run{run_idx}.json"
    if not fname.exists():
        return None
    try:
        return json.loads(fname.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
