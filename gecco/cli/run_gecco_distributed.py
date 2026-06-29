"""Internal CLI route for distributed worker execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rich.panel import Panel
from rich.table import Table

from config.schema import load_config
from gecco.coordination import SharedRegistry, apply_client_profile
from gecco.cli.config_paths import resolve_config_path, results_dir_for_config
from gecco.load_llms.model_loader import load_llm
from gecco.offline_evaluation.fit_generated_models import (
    run_fit_hierarchical as run_fit,
)
from gecco.prepare_data.data2text import get_data2text_function
from gecco.prepare_data.io import load_data, split_by_participant
from gecco.prompt_builder.prompt import PromptBuilderWrapper
from gecco.run_gecco import GeCCoModelSearch
from gecco.sentry_init import init_sentry
from gecco.tempdirs import configure_temp_dirs
from gecco.utils import TimestampedConsole


console = TimestampedConsole()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _split_prompt_train_test(df, data_cfg, evaluation_cfg):
    """Split participants into prompt, train, and test partitions."""
    splits = split_by_participant(df, data_cfg.id_column, data_cfg.splits)
    df_prompt = splits["prompt"]

    non_prompt_ids = sorted(
        set(df[data_cfg.id_column].unique()) - set(df_prompt[data_cfg.id_column].unique())
    )
    shuffled_ids = list(non_prompt_ids)
    np.random.default_rng(getattr(evaluation_cfg, "split_seed", 42)).shuffle(shuffled_ids)

    train_ratio = getattr(evaluation_cfg, "train_ratio", 0.7)
    n_train = int(len(shuffled_ids) * train_ratio)
    train_ids = shuffled_ids[:n_train]
    test_ids = shuffled_ids[n_train:]

    df_train = df[df[data_cfg.id_column].isin(train_ids)]
    df_test = df[df[data_cfg.id_column].isin(test_ids)]
    return df_prompt, df_train, df_test, train_ids, test_ids


def register_parser(subparsers) -> argparse.ArgumentParser:
    """Register the internal distributed worker subcommand."""
    parser = subparsers.add_parser(
        "distributed-client", help=argparse.SUPPRESS, description=argparse.SUPPRESS
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--client-id", type=int, default=None)
    parser.add_argument("--client-profile", type=str, default=None)
    parser.add_argument("--vllm-url", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--test", action="store_true")
    parser.set_defaults(handler=main)
    return parser


def run_distributed_client(
    *,
    config: str,
    client_id: int | None = None,
    client_profile: str | None = None,
    vllm_url: str | None = None,
    results_dir: str | None = None,
    test: bool = False,
) -> int | None:
    """Run a distributed GeCCo worker client."""
    configure_temp_dirs(PROJECT_ROOT, prefix="GeCCo")

    if vllm_url:
        os.environ["VLLM_BASE_URL"] = vllm_url

    numeric_id = client_id
    if numeric_id is None:
        numeric_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    resolved_client_id = client_profile if client_profile else numeric_id

    console.print(
        Panel(
            f"[bold]Client ID:[/] {resolved_client_id}\n"
            f"[bold]Profile:[/] {client_profile or 'default'}\n"
            f"[bold]Results Dir:[/] {results_dir or '(from config path)'}\n"
            f"[bold]vLLM URL:[/] {os.environ.get('VLLM_BASE_URL', '(not set)')}",
            title="Distributed GeCCo Client",
            style="blue",
        )
    )

    cfg = load_config(resolve_config_path(config, project_root=PROJECT_ROOT))

    init_sentry(
        cfg=cfg,
        task_name=cfg.task.name,
        client_id=resolved_client_id,
        config_name=config,
    )

    if client_profile:
        apply_client_profile(cfg, client_profile)

    cmg_cfg = getattr(cfg, "centralized_model_generation", None)
    cmg_enabled = cmg_cfg is not None and getattr(cmg_cfg, "enabled", False)
    is_cmg_generator = (
        cmg_enabled
        and client_profile is not None
        and str(client_profile) == str(getattr(cmg_cfg, "generator_client", ""))
    )

    if test:
        original_provider = cfg.llm.provider
        original_model = cfg.llm.base_model
        cfg.loop.max_independent_runs = 1
        cfg.loop.max_iterations = 1
        console.print(
            Panel(
                f"[bold yellow]TEST MODE[/]\n"
                f"Provider: {original_provider}\n"
                f"Model: {original_model}\n"
                f"Runs: 1, Iterations: 1",
                style="yellow",
            )
        )

    data_cfg = cfg.data
    metadata = getattr(getattr(cfg, "metadata", None), "flag", False)
    max_independent_runs = cfg.loop.max_independent_runs

    if results_dir:
        resolved_results_dir = Path(results_dir)
        if not resolved_results_dir.is_absolute():
            resolved_results_dir = PROJECT_ROOT / resolved_results_dir
    else:
        resolved_results_dir = results_dir_for_config(
            config,
            project_root=PROJECT_ROOT,
            fit_type=getattr(cfg.evaluation, "fit_type", "group"),
        )
    registry = SharedRegistry(resolved_results_dir / "shared_registry.duckdb")

    try:
        df = load_data(data_cfg.path, data_cfg.input_columns)
        df_prompt, df_train, df_test, train_ids, test_ids = _split_prompt_train_test(
            df, data_cfg, cfg.evaluation
        )

        split_table = Table(title="Data Split", show_header=True, header_style="bold")
        split_table.add_column("Split")
        split_table.add_column("Participants", justify="right")
        split_table.add_row("Prompt", str(len(df_prompt[data_cfg.id_column].unique())))
        split_table.add_row("Train", str(len(train_ids)))
        split_table.add_row("Test", str(len(test_ids)))
        console.print(split_table)

        if getattr(cfg.loop, "early_stopping", "False") == "True":
            df_baselines = load_data(data_cfg.path)
            splits_baselines = split_by_participant(
                df_baselines, data_cfg.id_column, data_cfg.splits
            )
            df_train_splits = splits_baselines["train"]
            baseline_bic = np.mean(df_train_splits.baseline_bic)
        else:
            baseline_bic = None

        data2text = get_data2text_function(data_cfg.data2text_function)
        data_text = data2text(
            df_prompt,
            id_col=data_cfg.id_column,
            template=data_cfg.narrative_template,
            fit_type=getattr(cfg.evaluation, "fit_type", "group"),
            metadata=getattr(cfg.metadata, "narrative_template", None) if metadata else None,
            max_trials=getattr(data_cfg, "max_prompt_trials", None),
            value_mappings=getattr(data_cfg, "value_mappings", None),
        )

        prompt_builder = PromptBuilderWrapper(cfg, data_text, df_prompt)
        model, tokenizer = load_llm(
            cfg.llm.provider,
            cfg.llm.base_model,
            base_url=getattr(cfg.llm, "base_url", None),
        )
    except Exception as exc:
        abort_reason = f"{exc.__class__.__name__}: {exc}"
        console.print(
            f"[red]Distributed client failed; publishing shared abort: {abort_reason}[/]"
        )
        try:
            registry.request_abort(
                client_id=resolved_client_id,
                reason=abort_reason,
                iteration=None,
            )
            registry.set_client_status(resolved_client_id, status="failed")
        except Exception as abort_exc:
            console.print(f"[red]Failed to publish shared abort: {abort_exc}[/]")
        raise
    search: GeCCoModelSearch | None = None
    try:
        from gecco.baseline import fit_baseline_if_needed

        id_eval_data = None
        if hasattr(cfg, "individual_differences_eval"):
            from gecco.offline_evaluation.individual_differences import load_id_data

            id_eval_data = load_id_data(cfg)

        baseline_result = fit_baseline_if_needed(
            cfg=cfg,
            df_train=df_train,
            registry=registry,
            id_eval_data=id_eval_data,
        )
        if baseline_result:
            console.print(
                f"[dim]Baseline {baseline_result['metric_name']}: "
                f"{baseline_result['metric_value']:.2f}[/]"
            )

        search = GeCCoModelSearch(
            model,
            tokenizer,
            cfg,
            df_train,
            prompt_builder,
            client_id=resolved_client_id,
            shared_registry=registry,
            config_path=config,
            results_dir=resolved_results_dir,
        )

        global_best_bic = np.inf

        for run_index in range(max_independent_runs):
            console.rule(
                f"[bold]Client {resolved_client_id} — Run {run_index + 1}/{max_independent_runs}"
            )

            best_model, best_bic, best_params = search.run_n_shots(
                run_index, baseline_bic=baseline_bic
            )
            best_iter = search.best_iter

            if is_cmg_generator:
                console.print(
                    "[green]CMG generator run complete; "
                    "skipping evaluator-only simulation and test fitting.[/]"
                )
                continue

            console.print(
                Panel(
                    f"[bold]Best BIC:[/] [cyan]{best_bic:.2f}[/]\n"
                    f"[bold]Parameters:[/] {', '.join(best_params)}",
                    title=f"Client {resolved_client_id} — Run {run_index} Complete",
                    style="green",
                )
            )

            if getattr(cfg.llm, "do_simulation", "False") == "True":
                from gecco.prompt_builder.simulation_prompt import simulation_prompt

                simulation_prompt_text = simulation_prompt(best_model, cfg)
                simulation_text = search.generate(model, tokenizer, simulation_prompt_text)
                simulation_dir = search.results_dir / "simulation"
                simulation_dir.mkdir(parents=True, exist_ok=True)
                simulation_file = (
                    simulation_dir
                    / f"simulation_model_client{resolved_client_id}_run{run_index}.txt"
                )
                with simulation_file.open("w", encoding="utf-8") as file_obj:
                    file_obj.write(simulation_text)

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

                search.results_dir.mkdir(parents=True, exist_ok=True)
                best_bic_file = (
                    search.results_dir
                    / "bics"
                    / f"best_bic_on_test_client{resolved_client_id}_run{run_index}.json"
                )
                with best_bic_file.open("w", encoding="utf-8") as file_obj:
                    json.dump(
                        {
                            "mean_" + metric_name: mean_metric,
                            "individual_" + metric_name: fit_res["eval_metrics"],
                            "mean_NLL": fit_res["mean_nll"],
                            "individual_NLL": fit_res["per_participant_nll"],
                        },
                        file_obj,
                    )

                param_df = pd.DataFrame(fit_res["parameter_values"], columns=params)
                param_dir = search.results_dir / "parameters"
                param_dir.mkdir(parents=True, exist_ok=True)
                param_file = (
                    param_dir
                    / f"best_params_on_test_client{resolved_client_id}_run{run_index}.csv"
                )
                param_df.to_csv(param_file, index=False)

            except Exception as exc:
                console.print(f"[bold red]Error fitting {func_name}:[/] {exc}")

            if best_bic < global_best_bic:
                global_best_bic = best_bic

        console.rule(f"[bold blue]Client {resolved_client_id} — GeCCo Search Complete")
        console.print(f"  Best mean BIC: [bold cyan]{global_best_bic:.2f}[/]")
        return None
    except Exception as exc:
        abort_reason = f"{exc.__class__.__name__}: {exc}"
        console.print(
            f"[red]Distributed client failed; publishing shared abort: {abort_reason}[/]"
        )
        try:
            registry.request_abort(
                client_id=resolved_client_id,
                reason=abort_reason,
                iteration=search.best_iter if search is not None and search.best_iter >= 0 else None,
            )
            registry.set_client_status(resolved_client_id, status="failed")
        except Exception as abort_exc:
            console.print(f"[red]Failed to publish shared abort: {abort_exc}[/]")
        raise
    finally:
        if search is not None:
            search.close()


def main(args: argparse.Namespace) -> int | None:
    """Run the distributed client command from parsed CLI arguments."""
    return run_distributed_client(
        config=args.config,
        client_id=args.client_id,
        client_profile=args.client_profile,
        vllm_url=args.vllm_url,
        results_dir=args.results_dir,
        test=args.test,
    )
