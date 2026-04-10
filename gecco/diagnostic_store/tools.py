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


def get_model(store: DiagnosticStore, model_id: int) -> dict | None:
    """Return the full record for a single model (including code)."""
    sql = """
        SELECT m.*, i.iteration, i.run_idx, i.tag
        FROM models m
        JOIN iterations i ON i.iteration_id = m.iteration_id
        WHERE m.model_id = ?
    """
    row = store.fetchone(sql, [model_id])
    return _hydrate(row) if row else None


def get_per_participant_fit(store: DiagnosticStore,
                            model_id: int) -> list[dict]:
    """Return per-participant BIC and parameter estimates for a model."""
    sql = """
        SELECT participant_idx, bic, n_trials, params
        FROM model_participants
        WHERE model_id = ?
        ORDER BY participant_idx
    """
    return [_hydrate(r) for r in store.fetchall(sql, [model_id])]


def get_recovery(store: DiagnosticStore, model_id: int) -> dict | None:
    """Return parameter recovery diagnostics for a model."""
    sql = """
        SELECT * FROM parameter_recovery WHERE model_id = ?
    """
    row = store.fetchone(sql, [model_id])
    return _hydrate(row) if row else None


def get_individual_differences(store: DiagnosticStore,
                                model_id: int) -> dict | None:
    """Return individual differences R² and predictor coefficients."""
    sql = """
        SELECT * FROM individual_differences WHERE model_id = ?
    """
    row = store.fetchone(sql, [model_id])
    return _hydrate(row) if row else None


def get_ppc(store: DiagnosticStore, model_id: int,
            statistic: str | None = None,
            condition: str | None = None) -> list[dict]:
    """Return PPC records for a model, optionally filtered."""
    params: list[Any] = [model_id]
    filters = "model_id = ?"
    if statistic:
        filters += " AND statistic_name = ?"
        params.append(statistic)
    if condition:
        filters += " AND condition = ?"
        params.append(condition)
    sql = f"""
        SELECT participant_id, statistic_name, condition,
               observed, simulated_mean, simulated_q025, simulated_q975, n_sims,
               CASE
                   WHEN observed < simulated_q025 OR observed > simulated_q975
                   THEN 1 ELSE 0
               END AS outside_95ci
        FROM ppc
        WHERE {filters}
        ORDER BY statistic_name, condition, participant_id
    """
    return store.fetchall(sql, params)


def get_block_residuals(store: DiagnosticStore, model_id: int) -> dict:
    """Return aggregated block-level NLL residual summaries for a model."""
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
                "and the best metric per iteration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run_idx": {
                        "type": "integer",
                        "description": "Filter to a specific run index (omit for all runs)."
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
                "Retrieve the full record for a single model, including its code, "
                "parameters, metric value, and iteration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "integer",
                        "description": "The model_id from the diagnostic store."
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
                "Return per-participant BIC values and fitted parameter estimates "
                "for a given model_id.  Useful for diagnosing heterogeneity in fit quality."
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
            "name": "get_recovery",
            "description": (
                "Return parameter recovery diagnostics for a model: overall pass/fail, "
                "mean Pearson r, and per-parameter r values."
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
                "mean and max R², best parameter, and per-parameter R² with "
                "predictor coefficients."
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
                "Return posterior predictive check (PPC) statistics for a model.  "
                "Each row reports an observed statistic value, the simulated mean, "
                "and 95% predictive interval.  The 'outside_95ci' flag marks "
                "observed values outside the predictive interval."
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
                "where the model's NLL per trial is especially high."
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
                "Useful for assessing convergence and improvement rate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run_idx": {
                        "type": "integer",
                        "description": "Filter to a specific run index (omit for all)."
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
