"""Baseline model fitting for GeCCo."""

from __future__ import annotations

from typing import Any

import numpy as np

from gecco.coordination import SharedRegistry
from gecco.utils import TimestampedConsole, log as _log

console = TimestampedConsole()


def fit_baseline_if_needed(
    *,
    cfg: Any,
    df_train,
    registry: SharedRegistry,
    id_eval_data=None,
) -> dict[str, Any] | None:
    """Fit and persist the baseline model when the registry is missing it.

    Args:
        cfg: Full experiment configuration.
        df_train: Training data used to fit the baseline.
        registry: Shared DuckDB-backed runtime registry.
        id_eval_data: Optional pre-loaded individual-differences data.

    Returns:
        The stored baseline result, or ``None`` when no baseline model is
        configured or fitting fails.
    """
    baseline_cfg = getattr(cfg, "baseline", None)
    baseline_code = getattr(baseline_cfg, "model", None) if baseline_cfg else None
    if not baseline_code:
        baseline_code = getattr(cfg.llm, "template_model", None)
    if not baseline_code:
        _log(
            "[GeCCo] No baseline model or template_model in config — skipping baseline"
        )
        return None

    try:
        existing_baseline = registry.read().get("baseline")
        if existing_baseline is not None:
            _log("[GeCCo] Loading cached baseline from shared registry")
            return existing_baseline

        def _baseline_from_row(row) -> dict[str, Any]:
            return {
                "function_name": row[0],
                "metric_name": row[1],
                "metric_value": row[2],
                "param_names": registry._from_json_value(row[3]) or [],
                "eval_metrics": registry._from_json_value(row[4]) or [],
                "mean_r2": row[5],
                "max_r2": row[6],
                "best_param": row[7],
                "per_param_r2": registry._from_json_value(row[8]) or {},
                "code": row[9],
                "val_mean_nll": row[10],
            }

        def _fit_or_load(conn) -> dict[str, Any] | None:
            row = conn.execute(
                "SELECT function_name, metric_name, metric_value, param_names, eval_metrics, "
                "mean_r2, max_r2, best_param, per_param_r2, code, val_mean_nll "
                "FROM runtime_baseline WHERE singleton = 1"
            ).fetchone()
            if row is not None:
                _log("[GeCCo] Loading cached baseline from shared registry")
                return _baseline_from_row(row)

            import re

            func_match = re.search(r"def\s+(\w+)\s*\(", baseline_code)
            func_name = func_match.group(1) if func_match else "cognitive_model"

            _log(f"[GeCCo] Fitting baseline model ({func_name})...")
            console.print(f"[bold]Fitting baseline model ({func_name})...[/]")

            from gecco.offline_evaluation.fit_generated_models import (
                run_fit_hierarchical as run_fit,
            )

            fit_res = run_fit(
                df_train, baseline_code, cfg=cfg, expected_func_name=func_name
            )

            result: dict[str, Any] = {
                "function_name": "baseline_model",
                "metric_name": fit_res["metric_name"],
                "metric_value": float(fit_res["metric_value"]),
                "param_names": fit_res["param_names"],
                "code": baseline_code,
                "eval_metrics": [float(v) for v in fit_res.get("eval_metrics", [])],
                "participant_n_trials": fit_res.get("participant_n_trials", []),
            }

            param_values = fit_res.get("parameter_values", [])
            if param_values:
                result["parameter_values"] = [
                    values.tolist() if isinstance(values, np.ndarray) else list(values)
                    for values in param_values
                ]

            if id_eval_data is not None:
                try:
                    from gecco.offline_evaluation.individual_differences import (
                        evaluate_individual_differences,
                    )

                    id_results = evaluate_individual_differences(
                        fit_res, df_train, cfg, id_data=id_eval_data
                    )
                    result["individual_differences"] = {
                        "mean_r2": id_results.get("mean_r2"),
                        "max_r2": id_results.get("max_r2"),
                        "best_param": id_results.get("best_param"),
                        "per_param_r2": id_results.get("per_param_r2"),
                        "summary_text": id_results.get("summary_text", ""),
                    }
                except Exception as exc:
                    _log(f"[GeCCo] Baseline individual differences eval failed: {exc}")

            id_res = result.get("individual_differences") or {}
            conn.execute(
                "INSERT OR REPLACE INTO runtime_baseline "
                "(singleton, function_name, metric_name, metric_value, param_names, eval_metrics, "
                "mean_r2, max_r2, best_param, per_param_r2, code, val_mean_nll) "
                "VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    result.get("function_name", "baseline_model"),
                    result.get("metric_name", "BIC"),
                    result.get("metric_value"),
                    registry._to_json_text(result.get("param_names", [])),
                    registry._to_json_text(result.get("eval_metrics", [])),
                    id_res.get("mean_r2"),
                    id_res.get("max_r2"),
                    id_res.get("best_param"),
                    registry._to_json_text(id_res.get("per_param_r2", {})),
                    result.get("code"),
                    result.get("val_mean_nll"),
                ],
            )

            _log(
                "[GeCCo] Baseline saved to shared registry: "
                f"{fit_res['metric_name']} = {fit_res['metric_value']:.2f}"
            )

            console.print(
                f"  [bold green]Baseline {fit_res['metric_name']}:[/] "
                f"[cyan]{fit_res['metric_value']:.2f}[/] "
                f"(params: {', '.join(fit_res['param_names'])})"
            )
            return result

        return registry._with_connection(
            write=True,
            operation="fit-baseline",
            callback=_fit_or_load,
        )

    except Exception as exc:
        _log(f"[GeCCo] Error fitting baseline: {exc}")
        console.print(f"[bold red]Error fitting baseline:[/] {exc}")
        return None
