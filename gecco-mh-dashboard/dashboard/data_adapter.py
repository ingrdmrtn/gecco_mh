from __future__ import annotations

import fcntl
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


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
                    "Mean R²": r.get("mean_r2"),
                    "Params": ", ".join(r.get("param_names", [])),
                    "Client": cid,
                    "Iteration": it,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Model", "BIC", "Mean R²", "Params", "Client", "Iteration"])

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
            "Mean R²": c["Mean R²"],
        }
        for p in all_params:
            row[p] = c["per_param"].get(p)
        rows.append(row)

    return pd.DataFrame(rows)


def summary_stats(data: dict[str, Any]) -> dict[str, int]:
    entries = data.get("client_entries", {})
    history = data.get("iteration_history", [])
    total_models = sum(len(h.get("results", [])) for h in history)
    return {
        "n_clients": len(entries),
        "running": sum(1 for e in entries.values() if e.get("status") == "running"),
        "complete": sum(1 for e in entries.values() if e.get("status") == "complete"),
        "iterations": len(history),
        "models": total_models,
        "param_combos": len(data.get("tried_param_sets", [])),
    }
