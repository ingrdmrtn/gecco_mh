#!/usr/bin/env python
"""
Test-fit a candidate cognitive model exactly as GeCCo would.

Usage examples:

  # Fit a model from a Python file:
  python scripts/test_fit_model.py --config two_step_factors.yaml --code model.py

  # Fit a model from the shared registry (by name or index):
  python scripts/test_fit_model.py --config two_step_factors.yaml --registry --model-name dual_lr_perseveration
  python scripts/test_fit_model.py --config two_step_factors.yaml --registry --model-index 0

  # Fit on test split instead of eval (default):
  python scripts/test_fit_model.py --config two_step_factors.yaml --code model.py --split test

  # Use plain MLE instead of hierarchical fitting:
  python scripts/test_fit_model.py --config two_step_factors.yaml --code model.py --fit-type mle

  # Override function name (default: cognitive_model1):
  python scripts/test_fit_model.py --config two_step_factors.yaml --code model.py --func-name cognitive_model2
"""

import os, sys
import argparse
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.schema import load_config
from gecco.prepare_data.io import load_data, split_by_participant
from gecco.offline_evaluation.fit_generated_models import run_fit, run_fit_hierarchical

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def load_code_from_file(path: str) -> str:
    return Path(path).read_text()


def load_code_from_registry(results_dir: Path, model_name: str = None,
                            model_index: int = None) -> str:
    """Extract model code from the shared registry."""
    registry_path = results_dir / "shared_registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    with open(registry_path) as f:
        data = json.load(f)

    # Collect all models from iteration history
    all_models = []
    for entry in data.get("iteration_history", []):
        for r in entry.get("results", []):
            code = r.get("code", "")
            if code and r.get("metric_name") not in ("FIT_ERROR", "RECOVERY_FAILED"):
                all_models.append({
                    "name": r.get("function_name", "unknown"),
                    "metric_value": r.get("metric_value", float("inf")),
                    "client_id": entry.get("client_id"),
                    "iteration": entry.get("iteration"),
                    "code": code,
                })

    if not all_models:
        raise ValueError("No models with code found in registry")

    if model_name:
        matches = [m for m in all_models if m["name"] == model_name]
        if not matches:
            available = sorted(set(m["name"] for m in all_models))
            raise ValueError(
                f"Model '{model_name}' not found. Available: {available}"
            )
        # Take the best-scoring match
        model = min(matches, key=lambda m: m["metric_value"])
    elif model_index is not None:
        # Sort by metric value and pick by index
        all_models.sort(key=lambda m: m["metric_value"])
        if model_index >= len(all_models):
            raise ValueError(
                f"Index {model_index} out of range (0-{len(all_models)-1})"
            )
        model = all_models[model_index]
    else:
        # Default: best model
        model = min(all_models, key=lambda m: m["metric_value"])

    console.print(
        f"[dim]Selected model:[/] [bold]{model['name']}[/] "
        f"(client={model['client_id']}, iter={model['iteration']}, "
        f"metric={model['metric_value']:.2f})"
    )
    return model["code"]


def get_eval_test_split(df, df_prompt, cfg):
    """Replicate the eval/test split from run_gecco_distributed.py."""
    data_cfg = cfg.data
    eval_test_proportion = getattr(cfg.evaluation, "eval_test_split", 0.7)
    non_prompt_ids = sorted(
        set(df[data_cfg.id_column].unique())
        - set(df_prompt[data_cfg.id_column].unique())
    )
    np.random.seed(getattr(cfg.evaluation, "split_seed", 42))
    np.random.shuffle(non_prompt_ids)
    split_idx = int(len(non_prompt_ids) * eval_test_proportion)
    eval_ids = non_prompt_ids[:split_idx]
    test_ids = non_prompt_ids[split_idx:]
    df_eval = df[df[data_cfg.id_column].isin(eval_ids)]
    df_test = df[df[data_cfg.id_column].isin(test_ids)]
    return df_eval, df_test


def main():
    parser = argparse.ArgumentParser(
        description="Test-fit a candidate cognitive model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Config YAML file name (in config/)")
    parser.add_argument("--code", type=str, default=None,
                        help="Path to a .py file containing the model code")
    parser.add_argument("--registry", action="store_true",
                        help="Load model from the shared registry")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Model name to find in registry")
    parser.add_argument("--model-index", type=int, default=None,
                        help="Model index in registry (sorted by metric, 0=best)")
    parser.add_argument("--func-name", type=str, default="cognitive_model1",
                        help="Function name to extract from code")
    parser.add_argument("--split", type=str, default="eval",
                        choices=["eval", "test"],
                        help="Data split to fit on (default: eval)")
    parser.add_argument("--fit-type", type=str, default=None,
                        choices=["mle", "hierarchical"],
                        help="Fitting method (default: from config)")
    args = parser.parse_args()

    if not args.code and not args.registry:
        parser.error("Provide either --code <file> or --registry")

    # --- Load config ---
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "config" / args.config)

    # --- Load data ---
    data_cfg = cfg.data
    df = load_data(data_cfg.path, data_cfg.input_columns)
    splits = split_by_participant(df, data_cfg.id_column, data_cfg.splits)
    df_prompt = splits["prompt"]
    df_eval, df_test = get_eval_test_split(df, df_prompt, cfg)

    df_fit = df_eval if args.split == "eval" else df_test

    split_table = Table(title="Data Split", show_header=True, header_style="bold")
    split_table.add_column("Split")
    split_table.add_column("Participants", justify="right")
    split_table.add_row("Prompt", str(len(df_prompt[data_cfg.id_column].unique())))
    split_table.add_row(
        f"{'→ ' if args.split == 'eval' else ''}Eval",
        str(len(df_eval[data_cfg.id_column].unique())),
    )
    split_table.add_row(
        f"{'→ ' if args.split == 'test' else ''}Test",
        str(len(df_test[data_cfg.id_column].unique())),
    )
    console.print(split_table)

    # --- Load model code ---
    if args.code:
        code = load_code_from_file(args.code)
        console.print(f"[dim]Loaded code from:[/] {args.code}")
    else:
        results_dir = (
            project_root / "results" / cfg.task.name
            if getattr(cfg.evaluation, "fit_type", "group") != "individual"
            else project_root / "results" / f"{cfg.task.name}_individual"
        )
        code = load_code_from_registry(
            results_dir,
            model_name=args.model_name,
            model_index=args.model_index,
        )

    # --- Show the code ---
    console.print(Panel(code, title="Model Code", border_style="blue"))

    # --- Fit ---
    fit_type = args.fit_type or "hierarchical"
    if fit_type == "mle":
        fit_func = run_fit
        console.print("[dim]Using MLE fitting[/]")
    else:
        fit_func = run_fit_hierarchical
        console.print("[dim]Using hierarchical (HBI) fitting[/]")

    import time
    t0 = time.time()
    try:
        result = fit_func(df_fit, code, cfg=cfg, expected_func_name=args.func_name)
    except Exception as e:
        console.print(f"[bold red]Fit failed:[/] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    elapsed = time.time() - t0

    # --- Display results ---
    metric_name = result["metric_name"]
    metric_value = result["metric_value"]
    param_names = result["param_names"]
    n_participants = len(df_fit[data_cfg.id_column].unique())

    results_table = Table(title="Fit Results", show_header=True, header_style="bold")
    results_table.add_column("Metric")
    results_table.add_column("Value", justify="right")
    results_table.add_row(f"Mean {metric_name}", f"{metric_value:.4f}")
    results_table.add_row("Parameters", ", ".join(param_names))
    results_table.add_row("N parameters", str(len(param_names)))

    eval_metrics = result.get("eval_metrics", [])
    if eval_metrics:
        results_table.add_row(f"Median {metric_name}", f"{np.median(eval_metrics):.4f}")
        results_table.add_row(f"Std {metric_name}", f"{np.std(eval_metrics):.4f}")
        results_table.add_row(f"Min {metric_name}", f"{np.min(eval_metrics):.4f}")
        results_table.add_row(f"Max {metric_name}", f"{np.max(eval_metrics):.4f}")

    results_table.add_section()
    results_table.add_row("Total time", f"{elapsed:.1f}s")
    results_table.add_row("Time per participant", f"{elapsed / n_participants:.2f}s")

    console.print(results_table)


if __name__ == "__main__":
    main()
