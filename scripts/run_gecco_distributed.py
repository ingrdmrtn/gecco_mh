# scripts/run_gecco_distributed.py
#
# Distributed GeCCo client — designed to run as a SLURM job array task.
# Multiple instances query a shared vLLM server and coordinate via a
# shared JSON registry on the filesystem.

import os, sys, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path
from config.schema import load_config
from gecco.offline_evaluation.fit_generated_models import run_fit_hierarchical as run_fit
from gecco.prepare_data.io import load_data, split_by_participant
from gecco.prepare_data.data2text import get_data2text_function
from gecco.load_llms.model_loader import load_llm
from gecco.run_gecco import GeCCoModelSearch
from gecco.prompt_builder.prompt import PromptBuilderWrapper
from gecco.coordination import SharedRegistry, apply_client_profile
import pandas as pd
import json
import argparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Distributed GeCCo client")
    parser.add_argument('--config', type=str, required=True, help='Config YAML file name')
    parser.add_argument('--client-id', type=int, default=None,
                        help='Client ID (defaults to $SLURM_ARRAY_TASK_ID)')
    parser.add_argument('--client-profile', type=str, default=None,
                        help='Named client profile from config clients: section')
    parser.add_argument('--vllm-url', type=str, default=None,
                        help='vLLM server URL (overrides $VLLM_BASE_URL)')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: use a small local model with 1 run and 1 iteration')
    args = parser.parse_args()

    # --- Set vLLM URL if provided (before load_llm reads it) ---
    if args.vllm_url:
        os.environ["VLLM_BASE_URL"] = args.vllm_url

    # --- Resolve client ID ---
    # Use profile name as client ID when available (more readable in dashboard),
    # fall back to numeric SLURM task ID or explicit --client-id.
    numeric_id = args.client_id
    if numeric_id is None:
        numeric_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    client_id = args.client_profile if args.client_profile else numeric_id

    console.print(Panel(
        f"[bold]Client ID:[/] {client_id}\n"
        f"[bold]Profile:[/] {args.client_profile or 'default'}\n"
        f"[bold]vLLM URL:[/] {os.environ.get('VLLM_BASE_URL', '(not set)')}",
        title="Distributed GeCCo Client",
        style="blue",
    ))

    # --- Load configuration & data ---
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "config" / args.config)

    # --- Apply client profile overrides ---
    if args.client_profile:
        apply_client_profile(cfg, args.client_profile)

    # --- Test mode ---
    if args.test:
        cfg.llm.provider = "qwen"
        cfg.llm.base_model = "Qwen/Qwen2.5-1.5B-Instruct"
        cfg.loop.max_independent_runs = 1
        cfg.loop.max_iterations = 1
        console.print(Panel("[bold yellow]TEST MODE[/]\nUsing Qwen 1.5B, 1 run, 1 iteration", style="yellow"))

    data_cfg = cfg.data
    metadata = getattr(getattr(cfg, "metadata", None), "flag", False)
    max_independent_runs = cfg.loop.max_independent_runs

    df = load_data(data_cfg.path, data_cfg.input_columns)
    splits = split_by_participant(df, data_cfg.id_column, data_cfg.splits)
    df_prompt = splits["prompt"]

    # --- Split eval/test by proportion of remaining participants ---
    eval_test_proportion = getattr(cfg.evaluation, "eval_test_split", 0.7)
    non_prompt_ids = sorted(set(df[data_cfg.id_column].unique()) - set(df_prompt[data_cfg.id_column].unique()))
    np.random.seed(getattr(cfg.evaluation, "split_seed", 42))
    np.random.shuffle(non_prompt_ids)
    split_idx = int(len(non_prompt_ids) * eval_test_proportion)
    eval_ids = non_prompt_ids[:split_idx]
    test_ids = non_prompt_ids[split_idx:]
    df_eval = df[df[data_cfg.id_column].isin(eval_ids)]
    df_test = df[df[data_cfg.id_column].isin(test_ids)]

    # Data split summary
    split_table = Table(title="Data Split", show_header=True, header_style="bold")
    split_table.add_column("Split")
    split_table.add_column("Participants", justify="right")
    split_table.add_row("Prompt", str(len(df_prompt[data_cfg.id_column].unique())))
    split_table.add_row("Eval", str(len(eval_ids)))
    split_table.add_row("Test", str(len(test_ids)))
    console.print(split_table)

    if getattr(cfg.loop, "early_stopping", "False") == "True":
        df_baselines = load_data(data_cfg.path)
        splits_baselines = split_by_participant(df_baselines, data_cfg.id_column, data_cfg.splits)
        df_prompt_splits, df_eval_splits = splits_baselines["prompt"], splits_baselines["eval"]
        baseline_bic = np.mean(df_eval_splits.baseline_bic)
    else:
        baseline_bic = None

    # --- Convert data to narrative text for the LLM ---
    data2text = get_data2text_function(data_cfg.data2text_function)
    data_text = data2text(
        df_prompt,
        id_col=data_cfg.id_column,
        template=data_cfg.narrative_template,
        fit_type=getattr(cfg.evaluation, "fit_type", "group"),
        metadata=getattr(cfg.metadata, "narrative_template", None) if metadata else None,
        max_trials=getattr(data_cfg, "max_prompt_trials", None),
        value_mappings=getattr(data_cfg, "value_mappings", None)
    )

    # --- Build prompt builder ---
    prompt_builder = PromptBuilderWrapper(cfg, data_text, df_prompt)

    # --- Load LLM ---
    model, tokenizer = load_llm(
        cfg.llm.provider,
        cfg.llm.base_model,
        base_url=getattr(cfg.llm, "base_url", None),
    )

    # --- Create shared registry ---
    results_dir = (
        project_root / "results" / cfg.task.name
        if getattr(cfg.evaluation, "fit_type", "group") != "individual"
        else project_root / "results" / f"{cfg.task.name}_individual"
    )
    registry = SharedRegistry(results_dir / "shared_registry.json")

    # --- Fit baseline model (once, with locking) ---
    from gecco.baseline import fit_baseline_if_needed

    id_eval_data = None
    if hasattr(cfg, 'individual_differences_eval'):
        from gecco.offline_evaluation.individual_differences import load_id_data
        id_eval_data = load_id_data(cfg)

    baseline_path = results_dir / "baseline.json"
    baseline_result = fit_baseline_if_needed(
        baseline_path, cfg, df_eval, registry=registry,
        id_eval_data=id_eval_data,
    )
    if baseline_result:
        console.print(
            f"[dim]Baseline {baseline_result['metric_name']}: "
            f"{baseline_result['metric_value']:.2f}[/]"
        )

    # --- Run GeCCo iterative model search ---
    search = GeCCoModelSearch(
        model, tokenizer, cfg, df_eval, prompt_builder,
        client_id=client_id, shared_registry=registry,
    )
    global_best_model = None
    global_best_bic = np.inf
    global_best_params = None

    for r in range(max_independent_runs):
        console.rule(f"[bold]Client {client_id} — Run {r+1}/{max_independent_runs}")

        best_model, best_bic, best_params = search.run_n_shots(r, baseline_bic=baseline_bic)
        best_iter = search.best_iter

        console.print(
            Panel(
                f"[bold]Best BIC:[/] [cyan]{best_bic:.2f}[/]\n"
                f"[bold]Parameters:[/] {', '.join(best_params)}",
                title=f"Client {client_id} — Run {r} Complete",
                style="green",
            )
        )

        if getattr(cfg.llm, "do_simulation", "False") == "True":
            from gecco.prompt_builder.simulation_prompt import simulation_prompt

            simulation_prompt_text = simulation_prompt(best_model, cfg)
            simulation_text = search.generate(model, tokenizer, simulation_prompt_text)
            simulation_dir = search.results_dir / "simulation"
            simulation_dir.mkdir(parents=True, exist_ok=True)
            simulation_file = simulation_dir / f"simulation_model_client{client_id}_run{r}.txt"
            with open(simulation_file, "w") as f:
                f.write(simulation_text)

        # fit the best model to test data
        console.print("[dim]Fitting best model to test data...[/]")
        try:
            func_name = f"cognitive_model{best_iter}"
            fit_res = run_fit(df_test, best_model, cfg=cfg, expected_func_name=func_name)
            mean_metric = float(fit_res["metric_value"])
            metric_name = fit_res["metric_name"]
            params = fit_res["param_names"]

            console.print(
                f"  [bold]{func_name}[/]: mean {metric_name} = [cyan]{mean_metric:.2f}[/]"
            )

            # save best model bic
            results_dir = search.results_dir
            results_dir.mkdir(parents=True, exist_ok=True)
            best_bic_file = results_dir / "bics" / f"best_bic_on_test_client{client_id}_run{r}.json"
            with open(best_bic_file, "w") as f:
                json.dump({"mean_"+metric_name: mean_metric, "individual_"+metric_name: fit_res['eval_metrics']}, f)

            # save best model params
            param_df = pd.DataFrame(fit_res["parameter_values"], columns=params)
            param_dir = results_dir / "parameters"
            param_dir.mkdir(parents=True, exist_ok=True)
            param_file = param_dir / f"best_params_on_test_client{client_id}_run{r}.csv"
            param_df.to_csv(param_file, index=False)

        except Exception as e:
            console.print(f"[bold red]Error fitting {func_name}:[/] {e}")

        if best_bic < global_best_bic:
            global_best_bic = best_bic
            global_best_model = best_model
            global_best_params = best_params
            global_best_iter = best_iter

    # --- Final results ---
    console.rule(f"[bold blue]Client {client_id} — GeCCo Search Complete")
    console.print(f"  Best mean BIC: [bold cyan]{global_best_bic:.2f}[/]")


if __name__ == "__main__":
    main()
