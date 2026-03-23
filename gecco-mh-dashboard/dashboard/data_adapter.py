from __future__ import annotations

import fcntl
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


def _apply_bic_cap(rows: list[dict[str, Any]], key: str = "BIC") -> list[dict[str, Any]]:
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


def _read_json_locked(path: Path) -> dict[str, Any] | None:
    """Read JSON under shared lock. Returns None when unavailable/unreadable."""
    if not path.exists():
        return None

    try:
        with open(path, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return None


def load_registry_snapshot(results_dir: Path) -> dict[str, Any] | None:
    return _read_json_locked(results_dir / "shared_registry.json")


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

    def _sort_key(k: str) -> int:
        return int(k) if str(k).isdigit() else 9999

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


def build_landscape_df(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in data.get("iteration_history", []):
        cid = entry.get("client_id")
        it = entry.get("iteration")
        for r in entry.get("results", []):
            bic = r.get("metric_value")
            if bic is None:
                continue
            rows.append(
                {
                    "Model": r.get("function_name", "?"),
                    "BIC": bic,
                    "Max R²": r.get("max_r2"),
                    "Best Param": r.get("best_param"),
                    "Mean R²": r.get("mean_r2"),
                    "Params": ", ".join(r.get("param_names", [])),
                    "Client": cid,
                    "Iteration": it,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Model", "BIC", "Max R²", "Best Param", "Mean R²", "Params", "Client", "Iteration"])

    rows = _apply_bic_cap(rows)
    return pd.DataFrame(rows).sort_values("BIC", ascending=True).reset_index(drop=True)


def build_iteration_df(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in data.get("iteration_history", []):
        cid = entry.get("client_id")
        it = entry.get("iteration")
        bics = [r.get("metric_value") for r in entry.get("results", []) if r.get("metric_value") is not None]
        if not bics:
            continue
        rows.append({"Client": str(cid), "Iteration": int(it), "Best BIC": float(min(bics))})

    if not rows:
        return pd.DataFrame(columns=["Client", "Iteration", "Best BIC"])

    rows = _apply_bic_cap(rows, key="Best BIC")
    return pd.DataFrame(rows).sort_values(["Client", "Iteration"])


def build_r2_df(data: dict[str, Any], top_n: int = 8) -> pd.DataFrame:
    candidates: list[dict[str, Any]] = []
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

    candidates = sorted(candidates, key=lambda x: (x["BIC"] if x["BIC"] is not None else float("inf")))[:top_n]
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
    failed_models = 0
    for h in history:
        for r in h.get("results", []):
            total_models += 1
            mn = r.get("metric_name", "BIC")
            if mn in ("RECOVERY_FAILED", "FIT_ERROR"):
                failed_models += 1
    return {
        "n_clients": len(entries),
        "running": sum(1 for e in entries.values() if e.get("status") == "running"),
        "complete": sum(1 for e in entries.values() if e.get("status") == "complete"),
        "iterations": len(history),
        "models": total_models,
        "failed": failed_models,
        "param_combos": len(data.get("tried_param_sets", [])),
    }


# ============================================================
# Results browser helpers
# ============================================================

def list_iterations(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract unique (client, iteration) pairs from iteration_history, sorted."""
    seen: list[dict[str, Any]] = []
    for entry in data.get("iteration_history", []):
        cid = entry.get("client_id")
        it = entry.get("iteration")
        n_models = len(entry.get("results", []))
        bics = [r.get("metric_value") for r in entry.get("results", [])
                if r.get("metric_value") is not None]
        seen.append({
            "client_id": cid,
            "iteration": it,
            "n_models": n_models,
            "best_bic": min(bics) if bics else None,
        })
    return sorted(seen, key=lambda x: (x["client_id"] or 0, x["iteration"] or 0))


def get_iteration_results(data: dict[str, Any], client_id: Any, iteration: int) -> list[dict[str, Any]]:
    """Get model results for a specific (client, iteration)."""
    for entry in data.get("iteration_history", []):
        if entry.get("client_id") == client_id and entry.get("iteration") == iteration:
            return entry.get("results", [])
    return []


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
