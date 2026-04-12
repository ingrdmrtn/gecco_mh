"""
Baseline model fitting for GeCCo.

Fits the template model from the config as a baseline comparator.
Uses file locking so only one distributed client does the fitting;
others wait and load the cached result.
"""

import fcntl
import json
import numpy as np
from pathlib import Path

from rich.console import Console

from gecco.utils import log as _log

console = Console()


def fit_baseline_if_needed(
    baseline_path, cfg, df_train, registry=None, id_eval_data=None
):
    """
    Fit the template model as a baseline if results don't already exist.

    Uses file locking so only one distributed client does the fitting;
    others block until the result is ready, then load it.

    Parameters
    ----------
    baseline_path : Path
        Where to save/load baseline results (e.g. results/task/baseline.json).
    cfg : object
        Full experiment configuration.
    df_train : pd.DataFrame
        Training data to fit the baseline on.
    registry : SharedRegistry, optional
        If provided, baseline results are written to the shared registry.
    id_eval_data : pd.DataFrame, optional
        Pre-loaded individual differences data for R² evaluation.

    Returns
    -------
    dict or None
        Baseline result dict, or None if no template model is configured.
    """
    # Prefer explicit baseline.model; fall back to llm.template_model
    baseline_cfg = getattr(cfg, "baseline", None)
    baseline_code = getattr(baseline_cfg, "model", None) if baseline_cfg else None
    if not baseline_code:
        baseline_code = getattr(cfg.llm, "template_model", None)
    if not baseline_code:
        _log(
            "[GeCCo] No baseline model or template_model in config — skipping baseline"
        )
        return None

    baseline_path = Path(baseline_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    # Fast path: already fitted
    if baseline_path.exists():
        _log("[GeCCo] Loading cached baseline from disk")
        with open(baseline_path, "r") as f:
            result = json.load(f)
        # Ensure registry has baseline too
        if registry is not None:
            registry.set_baseline(result)
        return result

    # Acquire exclusive lock — only one client fits
    lock_path = baseline_path.with_suffix(".lock")
    _log("[GeCCo] Acquiring baseline lock...")
    lock_fd = open(lock_path, "w")
    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)

    try:
        # Double-check after acquiring lock (another client may have finished)
        if baseline_path.exists():
            _log("[GeCCo] Baseline fitted by another client — loading")
            with open(baseline_path, "r") as f:
                result = json.load(f)
            if registry is not None:
                registry.set_baseline(result)
            return result

        # Extract function name from the baseline code
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

        result = {
            "function_name": "baseline_model",
            "metric_name": fit_res["metric_name"],
            "metric_value": float(fit_res["metric_value"]),
            "param_names": fit_res["param_names"],
            "code": baseline_code,
            "eval_metrics": [float(v) for v in fit_res.get("eval_metrics", [])],
            "participant_n_trials": fit_res.get("participant_n_trials", []),
        }

        # Serialize parameter values (list of arrays → list of lists)
        param_values = fit_res.get("parameter_values", [])
        if param_values:
            result["parameter_values"] = [
                v.tolist() if isinstance(v, np.ndarray) else list(v)
                for v in param_values
            ]

        # Individual differences evaluation
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
            except Exception as e:
                _log(f"[GeCCo] Baseline individual differences eval failed: {e}")

        # Save to disk
        with open(baseline_path, "w") as f:
            json.dump(result, f, indent=2)
        _log(
            f"[GeCCo] Baseline saved: {fit_res['metric_name']} = {fit_res['metric_value']:.2f}"
        )

        # Write to shared registry
        if registry is not None:
            registry.set_baseline(result)

        console.print(
            f"  [bold green]Baseline {fit_res['metric_name']}:[/] "
            f"[cyan]{fit_res['metric_value']:.2f}[/] "
            f"(params: {', '.join(fit_res['param_names'])})"
        )

        return result

    except Exception as e:
        _log(f"[GeCCo] Error fitting baseline: {e}")
        console.print(f"[bold red]Error fitting baseline:[/] {e}")
        return None

    finally:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        lock_fd.close()
