"""
Read-only query tools for the ToolUsingJudge.

Each function takes a :class:`DiagnosticStore` as its first argument and
returns a plain Python dict / list that can be JSON-serialised and fed back
to the LLM.

The companion ``TOOL_SCHEMAS`` list exposes these as OpenAI-style function
schemas so they can be passed directly to ``client.chat.completions.create``
as ``tools=[...]``.
"""

from __future__ import annotations

import json
from typing import Any

from gecco.diagnostic_store.store import DiagnosticStore


# ======================================================================
# Helper
# ======================================================================

def _parse_json_col(value: Any) -> Any:
    """Parse a JSON string column into a Python object, or return as-is."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def _hydrate(row: dict) -> dict:
    """Parse any JSON-valued columns in *row* in place."""
    json_cols = {"param_names", "params", "per_param_r", "per_param_r2",
                 "per_param_detail", "error_details"}
    return {k: _parse_json_col(v) if k in json_cols else v for k, v in row.items()}


def _summarize_per_param_r(data: Any) -> Any:
    """Collapse a per-param r dict into a compact summary (worst 5 highlighted)."""
    if not isinstance(data, dict) or not data:
        return data
    items = sorted(data.items(), key=lambda x: x[1] if x[1] is not None else 1.0)
    vals = [v for _, v in items if v is not None]
    return {
        "n_params": len(items),
        "mean_r": sum(vals) / len(vals) if vals else None,
        "min_r": min(vals) if vals else None,
        "max_r": max(vals) if vals else None,
        "worst_params": [{"name": k, "r": v} for k, v in items[:5]],
    }


def _summarize_per_param_r2(data: Any) -> Any:
    """Collapse a per-param R² dict into a compact summary (best 5 highlighted)."""
    if not isinstance(data, dict) or not data:
        return data
    items = sorted(
        data.items(),
        key=lambda x: x[1] if x[1] is not None else -1.0,
        reverse=True,
    )
    vals = [v for _, v in items if v is not None]
    return {
        "n_params": len(items),
        "mean_r2": sum(vals) / len(vals) if vals else None,
        "min_r2": min(vals) if vals else None,
        "max_r2": max(vals) if vals else None,
        "best_params": [{"name": k, "r2": v} for k, v in items[:5]],
    }


def _hydrate_for_judge(row: dict) -> dict:
    """Hydrate a row and replace large per-param fields with compact summaries.

    Drops ``per_param_detail`` entirely (verbose coefficient tables that the
    judge does not need).
    """
    result = _hydrate(row)
    if "per_param_r" in result:
        result["per_param_r"] = _summarize_per_param_r(result["per_param_r"])
    if "per_param_r2" in result:
        result["per_param_r2"] = _summarize_per_param_r2(result["per_param_r2"])
    result.pop("per_param_detail", None)
    return result


# ======================================================================
# Tool implementations
# ======================================================================

def list_iterations(store: DiagnosticStore, run_idx: int | None = None,
                    limit: int = 50) -> list[dict]:
    """Enumerate iterations with summary counts.

    Returns one row per iteration ordered by iteration index.
    """
    if run_idx is not None:
        sql = """
            SELECT i.iteration, i.run_idx, i.tag, i.timestamp,
                   i.n_models_proposed,
                   COUNT(CASE WHEN m.status = 'ok' THEN 1 END) AS n_ok,
                   MIN(CASE WHEN m.status = 'ok' THEN m.metric_value END) AS best_metric
            FROM iterations i
            LEFT JOIN models m ON m.iteration_id = i.iteration_id
            WHERE i.run_idx = ?
            GROUP BY i.iteration, i.run_idx, i.tag, i.timestamp, i.n_models_proposed
            ORDER BY i.iteration
            LIMIT ?
        """
        return store.fetchall(sql, [run_idx, limit])
    else:
        sql = """
            SELECT i.iteration, i.run_idx, i.tag, i.timestamp,
                   i.n_models_proposed,
                   COUNT(CASE WHEN m.status = 'ok' THEN 1 END) AS n_ok,
                   MIN(CASE WHEN m.status = 'ok' THEN m.metric_value END) AS best_metric
            FROM iterations i
            LEFT JOIN models m ON m.iteration_id = i.iteration_id
            GROUP BY i.iteration, i.run_idx, i.tag, i.timestamp, i.n_models_proposed
            ORDER BY i.iteration
            LIMIT ?
        """
        return store.fetchall(sql, [limit])


def get_best_models(store: DiagnosticStore, k: int = 5,
                    metric: str = "BIC") -> list[dict]:
    """Return the top-*k* models ordered by metric_value ascending."""
    sql = """
        SELECT m.model_id, m.run_idx, m.iteration, m.name,
               m.metric_name, m.metric_value, m.param_names, m.status
        FROM models m
        WHERE m.status = 'ok'
          AND m.metric_value IS NOT NULL
        ORDER BY m.metric_value ASC
        LIMIT ?
    """
    return [_hydrate(r) for r in store.fetchall(sql, [k])]


def get_model(store: DiagnosticStore, model_id: int,
              include_code: bool = True) -> dict | None:
    """Return the full record for a single model.

    Parameters
    ----------
    model_id:
        The model to retrieve.
    include_code:
        When True (default) the ``code`` field is included and truncated to
        6000 chars.  Pass ``include_code=false`` when you only need metadata
        (name, BIC, params) to save tokens.
    """
    sql = """
        SELECT m.*, i.iteration, i.run_idx, i.tag
        FROM models m
        JOIN iterations i ON i.iteration_id = m.iteration_id
        WHERE m.model_id = ?
    """
    row = store.fetchone(sql, [model_id])
    if row is None:
        return None
    result = _hydrate(row)
    if "code" in result and result["code"]:
        if not include_code:
            del result["code"]
        else:
            code = str(result["code"])
            if len(code) > 6000:
                result["code"] = code[:6000] + "\n... [truncated]"
    return result


def get_per_participant_fit(store: DiagnosticStore,
                            model_id: int,
                            full: bool = False) -> dict | list[dict]:
    """Return per-participant BIC and parameter estimates for a model.

    By default returns an aggregated summary:
    - overall BIC distribution (mean/std/min/max/quantiles)
    - per-parameter summary (mean/std/q025/q50/q975 across participants)
    - top-5 best-fitting participants (lowest BIC) and worst-5 (highest BIC)

    Set ``full=True`` to get raw per-participant rows (capped at 50).
    """
    sql = """
        SELECT participant_idx, bic, n_trials, params
        FROM model_participants
        WHERE model_id = ?
        ORDER BY bic ASC
    """
    rows = store.fetchall(sql, [model_id])
    if not rows:
        return {"model_id": model_id, "n_participants": 0}

    if full:
        return [_hydrate(r) for r in rows[:50]]

    import numpy as np  # core GeCCo dependency

    n = len(rows)
    bic_values = [r["bic"] for r in rows if r["bic"] is not None]
    bic_arr = np.array(bic_values, dtype=float) if bic_values else None

    # Collect per-parameter values across all participants
    all_params: dict[str, list[float]] = {}
    for row in rows:
        params = _parse_json_col(row["params"]) or {}
        for k, v in params.items():
            if v is not None:
                all_params.setdefault(k, []).append(float(v))

    param_summary = {}
    for pname, vals in all_params.items():
        arr = np.array(vals, dtype=float)
        param_summary[pname] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "q025": float(np.quantile(arr, 0.025)),
            "q50": float(np.median(arr)),
            "q975": float(np.quantile(arr, 0.975)),
        }

    # rows are sorted BIC ASC: first 5 = best, last 5 = worst
    best_5 = [_hydrate(r) for r in rows[:5]]
    worst_5 = [_hydrate(r) for r in rows[-5:]]

    result: dict = {
        "model_id": model_id,
        "n_participants": n,
        "param_summary": param_summary,
        "best_5_participants": best_5,
        "worst_5_participants": worst_5,
    }
    if bic_arr is not None and len(bic_arr):
        result["bic_mean"] = float(np.mean(bic_arr))
        result["bic_std"] = float(np.std(bic_arr))
        result["bic_min"] = float(np.min(bic_arr))
        result["bic_max"] = float(np.max(bic_arr))
        result["bic_quantiles"] = {
            "q05": float(np.quantile(bic_arr, 0.05)),
            "q25": float(np.quantile(bic_arr, 0.25)),
            "q50": float(np.median(bic_arr)),
            "q75": float(np.quantile(bic_arr, 0.75)),
            "q95": float(np.quantile(bic_arr, 0.95)),
        }
    return result


class DiagnosticNotAvailableError(RuntimeError):
    """Raised when a diagnostic tool is called but the data
    hasn't been computed (typically because the config option
    is disabled)."""


def _check_model_exists(store: DiagnosticStore, model_id: int) -> None:
    """Raise ValueError if model_id does not exist in the models table."""
    row = store.fetchone(
        "SELECT 1 FROM models WHERE model_id = ?", [model_id]
    )
    if row is None:
        raise ValueError(
            f"model_id={model_id} does not exist in the database. "
            "Ensure the model ID is valid."
        )


def get_recovery(store: DiagnosticStore, model_id: int) -> dict:
    """Return parameter recovery diagnostics for a model.

    ``per_param_r`` is summarised to ``{n_params, mean_r, min_r, max_r,
    worst_params}``.  ``per_param_detail`` (verbose coefficient tables) is
    omitted entirely.

    Raises:
        ValueError: If model_id does not exist.
        DiagnosticNotAvailableError: If parameter recovery data has not
            been computed for this model.  Enable via
            ``parameter_recovery.enabled: true`` in config.
    """
    _check_model_exists(store, model_id)
    sql = """
        SELECT * FROM parameter_recovery WHERE model_id = ?
    """
    row = store.fetchone(sql, [model_id])
    if row is None:
        raise DiagnosticNotAvailableError(
            f"Parameter recovery data not found for model_id={model_id}. "
            "This diagnostic requires 'parameter_recovery.enabled: true' in the "
            "config. Add this section to enable parameter identifiability checks."
        )
    return _hydrate_for_judge(row)


def get_individual_differences(store: DiagnosticStore,
                                model_id: int) -> dict:
    """Return individual differences R² and predictor coefficients.

    ``per_param_r2`` is summarised to ``{n_params, mean_r2, min_r2, max_r2,
    best_params}``.  ``per_param_detail`` (verbose coefficient tables) is
    omitted entirely.

    Raises:
        ValueError: If model_id does not exist.
        DiagnosticNotAvailableError: If individual-differences data has not
            been computed.  Requires covariates to be configured in the
            data section.
    """
    _check_model_exists(store, model_id)
    sql = """
        SELECT * FROM individual_differences WHERE model_id = ?
    """
    row = store.fetchone(sql, [model_id])
    if row is None:
        raise DiagnosticNotAvailableError(
            f"Individual differences data not found for model_id={model_id}. "
            "This diagnostic requires covariates to be configured in the data "
            "section (e.g., 'data.covariates: [anxiety_score, age]'). "
            "Without covariates, individual differences analysis cannot be "
            "performed."
        )
    return _hydrate_for_judge(row)


def get_ppc(store: DiagnosticStore, model_id: int,
            statistic: str | None = None,
            condition: str | None = None,
            participant_detail: bool = False) -> list[dict]:
    """Return PPC statistics for a model.

    By default returns one aggregated row per (statistic, condition) pair:
    ``n_participants``, ``n_outside_95ci``, ``frac_outside_95ci``,
    ``mean_observed``, ``mean_simulated_mean``, ``mean_abs_zscore``.
    Rows are sorted by ``frac_outside_95ci`` descending so the worst-fitting
    statistics appear first.

    Set ``participant_detail=True`` to instead get raw per-participant rows
    for the *single worst* (statistic, condition) combination (cap 50 rows).
    ``statistic`` and ``condition`` filters are applied before selecting the
    worst group, so you can use them to pin a specific group.

    Raises:
        ValueError: If model_id does not exist.
        DiagnosticNotAvailableError: If PPC data has not been computed for
            this model.  Enable via ``judge.ppc.enabled: true`` in config.
    """
    _check_model_exists(store, model_id)
    # Existence check — needed because aggregate queries silently return []
    # when no rows exist, giving no signal that the diagnostic wasn't run.
    check_sql = "SELECT 1 FROM ppc WHERE model_id = ? LIMIT 1"
    if store.fetchone(check_sql, [model_id]) is None:
        raise DiagnosticNotAvailableError(
            f"PPC data not found for model_id={model_id}. "
            "This diagnostic requires 'judge.ppc.enabled: true' in the config. "
            "Add this section under 'judge:' to enable posterior predictive checks."
        )

    qparams: list[Any] = [model_id]
    filters = "model_id = ?"
    if statistic:
        filters += " AND statistic_name = ?"
        qparams.append(statistic)
    if condition:
        filters += " AND condition = ?"
        qparams.append(condition)

    if not participant_detail:
        sql = f"""
            SELECT
                statistic_name,
                condition,
                COUNT(DISTINCT participant_id)                              AS n_participants,
                SUM(CASE WHEN observed < simulated_q025
                              OR observed > simulated_q975 THEN 1 ELSE 0 END) AS n_outside_95ci,
                AVG(CASE WHEN observed < simulated_q025
                              OR observed > simulated_q975 THEN 1.0 ELSE 0.0 END) AS frac_outside_95ci,
                AVG(observed)                                               AS mean_observed,
                AVG(simulated_mean)                                         AS mean_simulated_mean,
                AVG(ABS((observed - simulated_mean)
                    / NULLIF((simulated_q975 - simulated_q025) / 3.92, 0))) AS mean_abs_zscore
            FROM ppc
            WHERE {filters}
            GROUP BY statistic_name, condition
            ORDER BY frac_outside_95ci DESC
        """
        return store.fetchall(sql, qparams)

    # participant_detail: find worst group, return raw rows (cap 50)
    worst_group_sql = f"""
        SELECT statistic_name, condition,
               AVG(CASE WHEN observed < simulated_q025
                             OR observed > simulated_q975 THEN 1.0 ELSE 0.0 END) AS frac
        FROM ppc
        WHERE {filters}
        GROUP BY statistic_name, condition
        ORDER BY frac DESC
        LIMIT 1
    """
    worst = store.fetchone(worst_group_sql, qparams)
    if worst is None:
        return []
    detail_params: list[Any] = [model_id, worst["statistic_name"], worst["condition"]]
    detail_sql = """
        SELECT participant_id, statistic_name, condition,
               observed, simulated_mean, simulated_q025, simulated_q975, n_sims,
               CASE WHEN observed < simulated_q025 OR observed > simulated_q975
                    THEN 1 ELSE 0
               END AS outside_95ci
        FROM ppc
        WHERE model_id = ?
          AND statistic_name = ?
          AND condition = ?
        ORDER BY participant_id
        LIMIT 50
    """
    return store.fetchall(detail_sql, detail_params)


def get_block_residuals(store: DiagnosticStore, model_id: int) -> dict:
    """Return aggregated block-level NLL residual summaries for a model.

    Useful for identifying phases of the task where the model's NLL per
    trial is especially high.

    Raises:
        ValueError: If model_id does not exist.
        DiagnosticNotAvailableError: If block residual data has not been
            computed.  Enable via ``judge.block_residuals.enabled: true``
            in config (auto-enabled when PPC is enabled).
    """
    _check_model_exists(store, model_id)
    # Existence check — needed because the current implementation returns
    # {n_participants: 0, blocks: []} for missing data, which could be
    # confused with a legitimate empty result.
    check_sql = "SELECT 1 FROM block_residuals WHERE model_id = ? LIMIT 1"
    if store.fetchone(check_sql, [model_id]) is None:
        raise DiagnosticNotAvailableError(
            f"Block residual data not found for model_id={model_id}. "
            "This diagnostic requires 'judge.block_residuals.enabled: true' "
            "in the config (or it is auto-enabled when 'judge.ppc.enabled: true'). "
            "Add this section under 'judge:' to enable block-level residual "
            "analysis."
        )

    participant_row = store.fetchone(
        "SELECT COUNT(DISTINCT participant_id) AS n_participants "
        "FROM block_residuals WHERE model_id = ?",
        [model_id],
    )
    summary_rows = store.fetchall(
        """
        SELECT
            block_idx,
            MIN(block_start) AS block_start,
            MAX(block_end) AS block_end,
            AVG(mean_nll_per_trial) AS mean_nll_per_trial_mean,
            STDDEV_SAMP(mean_nll_per_trial) AS mean_nll_per_trial_std,
            MIN(mean_nll_per_trial) AS mean_nll_per_trial_min,
            MAX(mean_nll_per_trial) AS mean_nll_per_trial_max,
            COUNT(DISTINCT participant_id) AS n_participants
        FROM block_residuals
        WHERE model_id = ?
        GROUP BY block_idx
        ORDER BY block_idx
        """,
        [model_id],
    )
    return {
        "model_id": model_id,
        "n_participants": int((participant_row or {}).get("n_participants") or 0),
        "blocks": summary_rows,
    }


def get_participant_best_models(
    store: DiagnosticStore,
    run_idx: int | None = None,
    top_k_models: int | None = None,
) -> dict:
    """Return the best-fitting model for each participant and heterogeneity summary."""
    params: list[Any] = []
    run_filter = ""
    if run_idx is not None:
        run_filter = "AND m.run_idx = ?"
        params.append(run_idx)

    sql = f"""
        WITH ranked AS (
            SELECT
                mp.participant_idx,
                m.model_id,
                m.name,
                m.iteration,
                m.run_idx,
                mp.bic,
                ROW_NUMBER() OVER (
                    PARTITION BY mp.participant_idx
                    ORDER BY mp.bic ASC, m.iteration ASC, m.model_id ASC
                ) AS rn
            FROM model_participants mp
            JOIN models m ON mp.model_id = m.model_id
            WHERE m.status = 'ok'
              AND mp.bic IS NOT NULL
              {run_filter}
        )
        SELECT participant_idx, model_id, name, iteration, run_idx, bic
        FROM ranked
        WHERE rn = 1
        ORDER BY participant_idx
    """
    rows = store.fetchall(sql, params)
    participants = [
        {
            "participant_idx": row["participant_idx"],
            "best_model_id": row["model_id"],
            "best_model_name": row["name"],
            "best_bic": row["bic"],
            "iteration": row["iteration"],
            "run_idx": row["run_idx"],
        }
        for row in rows
    ]

    model_counts: dict[str, int] = {}
    for row in rows:
        model_counts[row["name"]] = model_counts.get(row["name"], 0) + 1

    n_participants = len(rows)
    max_count = max(model_counts.values(), default=0)
    modal_model = None
    if model_counts:
        modal_model = sorted(
            model_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]

    if top_k_models is not None and top_k_models > 0:
        sorted_counts = sorted(
            model_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_k_models]
        model_counts_out = dict(sorted_counts)
    else:
        model_counts_out = dict(sorted(model_counts.items()))

    heterogeneity_index = (
        1.0 - (max_count / n_participants)
        if n_participants > 0 else 0.0
    )

    return {
        "participants": participants,
        "summary": {
            "n_participants": n_participants,
            "n_unique_models": len(model_counts),
            "model_counts": model_counts_out,
            "modal_model": modal_model,
            "heterogeneity_index": heterogeneity_index,
        },
    }


def compare_models(store: DiagnosticStore,
                   model_ids: list[int]) -> list[dict]:
    """Side-by-side comparison of key metrics for the given model IDs."""
    if not model_ids:
        return []
    placeholders = ",".join(["?"] * len(model_ids))
    sql = f"""
        SELECT
            m.model_id, m.name, m.iteration, m.metric_name, m.metric_value,
            m.param_names, m.status,
            pr.mean_r AS recovery_mean_r, pr.passed AS recovery_passed,
            id.mean_r2 AS id_mean_r2, id.max_r2 AS id_max_r2,
            id.best_param AS id_best_param
        FROM models m
        LEFT JOIN parameter_recovery pr ON pr.model_id = m.model_id
        LEFT JOIN individual_differences id ON id.model_id = m.model_id
        WHERE m.model_id IN ({placeholders})
        ORDER BY m.metric_value ASC NULLS LAST
    """
    return [_hydrate(r) for r in store.fetchall(sql, model_ids)]


def get_parameter_distribution(store: DiagnosticStore, model_id: int,
                                param_name: str) -> dict:
    """Return cross-participant distribution statistics for one parameter."""
    sql = """
        SELECT params FROM model_participants WHERE model_id = ?
    """
    rows = store.fetchall(sql, [model_id])
    values = []
    for row in rows:
        params = _parse_json_col(row["params"]) or {}
        if param_name in params:
            v = params[param_name]
            if v is not None:
                values.append(float(v))

    if not values:
        return {"param_name": param_name, "n": 0, "values": []}

    import numpy as np  # local import — numpy is a core GeCCo dependency
    arr = np.array(values)
    return {
        "param_name": param_name,
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q025": float(np.quantile(arr, 0.025)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "q75": float(np.quantile(arr, 0.75)),
        "q975": float(np.quantile(arr, 0.975)),
        "max": float(np.max(arr)),
    }


def search_models(store: DiagnosticStore, code_contains: str | None = None,
                  param_contains: str | None = None,
                  metric_lt: float | None = None,
                  limit: int = 20) -> list[dict]:
    """Filter models by code substring, parameter name, or metric threshold."""
    filters = ["m.status = 'ok'"]
    params: list[Any] = []

    if code_contains:
        filters.append("m.code LIKE ?")
        params.append(f"%{code_contains}%")

    if param_contains:
        filters.append("m.param_names LIKE ?")
        params.append(f"%{param_contains}%")

    if metric_lt is not None:
        filters.append("m.metric_value < ?")
        params.append(metric_lt)

    where = " AND ".join(filters)
    params.append(limit)

    sql = f"""
        SELECT m.model_id, m.run_idx, m.iteration, m.name,
               m.metric_value, m.param_names, m.status
        FROM models m
        WHERE {where}
        ORDER BY m.metric_value ASC NULLS LAST
        LIMIT ?
    """
    return [_hydrate(r) for r in store.fetchall(sql, params)]


def get_bic_trajectory(store: DiagnosticStore,
                       run_idx: int | None = None) -> list[dict]:
    """Return the best metric value per iteration (time series)."""
    if run_idx is not None:
        sql = """
            SELECT m.iteration, MIN(m.metric_value) AS best_metric,
                   COUNT(*) AS n_models_total,
                   COUNT(CASE WHEN m.status = 'ok' THEN 1 END) AS n_ok
            FROM models m
            WHERE m.run_idx = ?
            GROUP BY m.iteration
            ORDER BY m.iteration
        """
        return store.fetchall(sql, [run_idx])
    else:
        sql = """
            SELECT m.iteration, MIN(m.metric_value) AS best_metric,
                   COUNT(*) AS n_models_total,
                   COUNT(CASE WHEN m.status = 'ok' THEN 1 END) AS n_ok
            FROM models m
            GROUP BY m.iteration
            ORDER BY m.iteration
        """
        return store.fetchall(sql)


# ======================================================================
# OpenAI-style tool schemas
# ======================================================================

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_iterations",
            "description": (
                "Enumerate all iterations recorded in the diagnostic store, "
                "with summary counts of models proposed, successfully fit, "
                "and the best metric per iteration. "
                "NOTE: run_idx is the client/run identifier (e.g. 0), NOT the "
                "iteration number. Omit run_idx to query across all runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run_idx": {
                        "type": "integer",
                        "description": (
                            "Filter to a specific run index (the client/run identifier, "
                            "NOT the iteration number). Omit to return all runs."
                        )
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of rows to return (default 50).",
                        "default": 50
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_best_models",
            "description": (
                "Return the top-k models by BIC (ascending) across all iterations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "k": {
                        "type": "integer",
                        "description": "Number of models to return (default 5).",
                        "default": 5
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_model",
            "description": (
                "Retrieve the full record for a single model, including its "
                "parameters, metric value, and iteration.  By default the model "
                "code is included (truncated to 6000 chars).  Pass "
                "include_code=false when you only need metadata (name, BIC, "
                "param list) to save tokens."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "integer",
                        "description": "The model_id from the diagnostic store."
                    },
                    "include_code": {
                        "type": "boolean",
                        "description": (
                            "Whether to include the model code field (default true). "
                            "Use false when you only need metadata."
                        ),
                        "default": True
                    }
                },
                "required": ["model_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_per_participant_fit",
            "description": (
                "Return a fit summary for a given model_id.  By default returns "
                "an aggregated summary: overall BIC distribution "
                "(mean/std/min/max/quantiles), per-parameter summary across "
                "participants (mean/std/q025/q50/q975), and the top-5 "
                "best-fitting and worst-5 worst-fitting participant rows.  "
                "Set full=true to get raw per-participant rows instead (capped "
                "at 50 rows)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "integer",
                        "description": "The model_id to look up."
                    },
                    "full": {
                        "type": "boolean",
                        "description": (
                            "If true, return raw per-participant rows (capped at 50) "
                            "instead of the aggregated summary.  Default false."
                        ),
                        "default": False
                    }
                },
                "required": ["model_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recovery",
            "description": (
                "Return parameter recovery diagnostics for a model: overall "
                "pass/fail, mean Pearson r, and a compact per-parameter r "
                "summary (n_params, mean_r, min_r, max_r, worst 5 params by r). "
                "Verbose coefficient tables are omitted. "
                "WARNING: Returns an error if parameter recovery is not enabled "
                "in the config (parameter_recovery.enabled). Only call this tool "
                "when you know recovery diagnostics have been computed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "integer",
                        "description": "The model_id to look up."
                    }
                },
                "required": ["model_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_individual_differences",
            "description": (
                "Return individual-differences regression results for a model: "
                "mean and max R², best parameter, and a compact per-parameter "
                "R² summary (n_params, mean_r2, min_r2, max_r2, best 5 params "
                "by R²).  Verbose coefficient tables are omitted. "
                "WARNING: Returns an error if covariates are not configured "
                "in the data section. Only call this tool when you know "
                "individual differences have been computed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "integer",
                        "description": "The model_id to look up."
                    }
                },
                "required": ["model_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ppc",
            "description": (
                "Return posterior predictive check (PPC) statistics for a model. "
                "By default returns one aggregated row per (statistic, condition) "
                "pair: n_participants, n_outside_95ci, frac_outside_95ci, "
                "mean_observed, mean_simulated_mean, mean_abs_zscore.  Rows are "
                "sorted by frac_outside_95ci descending so the worst-fitting "
                "statistics appear first.  "
                "Set participant_detail=true to get raw per-participant rows "
                "for the single worst (statistic, condition) group (capped at 50 "
                "rows); combine with statistic/condition filters to target a "
                "specific group. "
                "WARNING: Returns an error if PPC is not enabled in the config "
                "(judge.ppc.enabled). Only call this tool when you know PPC "
                "diagnostics have been computed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "integer",
                        "description": "The model_id to look up."
                    },
                    "statistic": {
                        "type": "string",
                        "description": "Optional: filter to a specific statistic name."
                    },
                    "condition": {
                        "type": "string",
                        "description": "Optional: filter to a specific condition label."
                    },
                    "participant_detail": {
                        "type": "boolean",
                        "description": (
                            "If true, return raw per-participant rows for the worst "
                            "group instead of the aggregated summary (cap 50 rows). "
                            "Default false."
                        ),
                        "default": False
                    }
                },
                "required": ["model_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_block_residuals",
            "description": (
                "Return block-level residual summaries for a model, aggregated "
                "across participants. Useful for identifying phases of the task "
                "where the model's NLL per trial is especially high. "
                "WARNING: Returns an error if block residuals are not enabled "
                "in the config (judge.block_residuals.enabled). Only call this "
                "tool when you know block residuals have been computed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "integer",
                        "description": "The model_id to look up."
                    }
                },
                "required": ["model_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_models",
            "description": (
                "Side-by-side comparison of key metrics (BIC, recovery, individual "
                "differences) for a list of model IDs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of model_id values to compare."
                    }
                },
                "required": ["model_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_participant_best_models",
            "description": (
                "For each participant, return the best-fitting model by BIC and "
                "summarise model heterogeneity across participants."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run_idx": {
                        "type": "integer",
                        "description": "Optional: restrict the query to one run index."
                    },
                    "top_k_models": {
                        "type": "integer",
                        "description": "Optional: limit summary model_counts to the top-k most frequent models."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_parameter_distribution",
            "description": (
                "Return descriptive statistics (mean, SD, quantiles) of the "
                "cross-participant distribution of a single parameter for a model."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "integer",
                        "description": "The model_id to query."
                    },
                    "param_name": {
                        "type": "string",
                        "description": "Name of the parameter."
                    }
                },
                "required": ["model_id", "param_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_models",
            "description": (
                "Filter models by code substring, parameter name presence, or "
                "metric threshold.  Returns model_id, name, iteration, and metric value."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code_contains": {
                        "type": "string",
                        "description": "Substring that must appear in the model code."
                    },
                    "param_contains": {
                        "type": "string",
                        "description": "Parameter name substring to search for."
                    },
                    "metric_lt": {
                        "type": "number",
                        "description": "Return only models with metric_value < this threshold."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default 20).",
                        "default": 20
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_bic_trajectory",
            "description": (
                "Return the best BIC per iteration as a time series.  "
                "Useful for assessing convergence and improvement rate. "
                "NOTE: run_idx is the client/run identifier (e.g. 0), NOT the "
                "iteration number. Omit run_idx to get the trajectory across all runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run_idx": {
                        "type": "integer",
                        "description": (
                            "Filter to a specific run index (the client/run identifier, "
                            "NOT the iteration number). Omit to return all runs."
                        )
                    }
                },
                "required": []
            }
        }
    },
]


# ======================================================================
# Tool dispatcher
# ======================================================================

_TOOL_FUNCTIONS = {
    "list_iterations": list_iterations,
    "get_best_models": get_best_models,
    "get_model": get_model,
    "get_per_participant_fit": get_per_participant_fit,
    "get_recovery": get_recovery,
    "get_individual_differences": get_individual_differences,
    "get_ppc": get_ppc,
    "get_block_residuals": get_block_residuals,
    "compare_models": compare_models,
    "get_participant_best_models": get_participant_best_models,
    "get_parameter_distribution": get_parameter_distribution,
    "search_models": search_models,
    "get_bic_trajectory": get_bic_trajectory,
}


def dispatch_tool(store: DiagnosticStore, tool_name: str, args: dict) -> Any:
    """Call the tool function matching *tool_name* with *args*.

    Returns the result, or a dict ``{"error": "..."}`` if the tool is
    unknown or raises an exception.
    """
    fn = _TOOL_FUNCTIONS.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        return fn(store, **args)
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}
