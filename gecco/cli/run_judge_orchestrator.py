"""CLI route for the judge orchestrator."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from rich.panel import Panel

from config.schema import load_config
from gecco.construct_feedback.orchestrated import run_orchestrated_judge_pipeline
from gecco.construct_feedback.tool_judge import ToolUsingJudge
from gecco.coordination import SharedRegistry
from gecco.diagnostic_store.rebuild import rebuild_from_artifacts
from gecco.load_llms.model_loader import load_llm
from gecco.prepare_data.data2text import get_data2text_function
from gecco.prepare_data.io import load_data, split_by_participant
from gecco.prompt_builder.prompt import PromptBuilderWrapper
from gecco.sentry_init import init_sentry
from gecco.tempdirs import configure_temp_dirs
from gecco.utils import TimestampedConsole


console = TimestampedConsole()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def register_parser(subparsers) -> argparse.ArgumentParser:
    """Register the orchestrator subcommand."""
    parser = subparsers.add_parser("orchestrate", help="Run the central judge orchestrator")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--vllm-url", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--n-clients", type=int, default=None)
    parser.set_defaults(handler=main)
    return parser


def run_orchestrator(
    *,
    config: str,
    vllm_url: str | None = None,
    results_dir: str | None = None,
    n_clients: int | None = None,
) -> int | None:
    """Run the centralised judge orchestrator."""
    configure_temp_dirs(PROJECT_ROOT, prefix="Orchestrator")

    if vllm_url:
        os.environ["VLLM_BASE_URL"] = vllm_url

    console.print(
        Panel(
            f"[bold]Config:[/] {config}\n"
            f"[bold]vLLM URL:[/] {os.environ.get('VLLM_BASE_URL', '(not set)')}\n"
            f"[bold]Results Dir:[/] {results_dir or '(from task name)'}\n"
            f"[bold]N Clients:[/] {n_clients or '(from config)'}",
            title="Centralized Judge Orchestrator",
            style="cyan",
        )
    )

    cfg = load_config(PROJECT_ROOT / "config" / config)

    init_sentry(cfg=cfg, task_name=cfg.task.name, config_name=config)

    resolved_results_dir = Path(results_dir) if results_dir else Path("results") / cfg.task.name

    cmg_cfg = getattr(cfg, "centralized_model_generation", None)
    cmg_enabled = cmg_cfg is not None and getattr(cmg_cfg, "enabled", False)

    if cmg_enabled:
        generator_client = str(getattr(cmg_cfg, "generator_client", ""))
        if not generator_client:
            console.print(
                "[red]Error: centralized_model_generation.generator_client is required[/]"
            )
            raise SystemExit(1)
        if generator_client.isdigit() or generator_client.lstrip("-").isdigit():
            console.print(
                "[red]Error: centralized_model_generation.generator_client must be a "
                "named profile, not a numeric evaluator ID[/]"
            )
            raise SystemExit(1)
        resolved_n_clients = getattr(cmg_cfg, "n_models", None)
        if not isinstance(resolved_n_clients, int) or resolved_n_clients <= 0:
            console.print(
                "[red]Error: centralized_model_generation.n_models must be a positive integer[/]"
            )
            raise SystemExit(1)
    elif n_clients:
        resolved_n_clients = n_clients
    elif hasattr(cfg, "loop") and hasattr(cfg.loop, "n_clients") and cfg.loop.n_clients:
        resolved_n_clients = cfg.loop.n_clients
    else:
        console.print(
            "[red]Error: n_clients not specified in config.loop.n_clients and not "
            "provided via --n-clients[/]"
        )
        raise SystemExit(1)

    registry_path = resolved_results_dir / "shared_registry.json"
    registry = SharedRegistry(str(registry_path))

    console.print("[cyan]Loading LLM model...[/]")
    model, tokenizer = load_llm(cfg.llm.provider, cfg.llm.base_model)

    console.print("[cyan]Loading data...[/]")
    data_cfg = cfg.data
    df = load_data(data_cfg.path, data_cfg.input_columns)
    splits = split_by_participant(df, data_cfg.id_column, data_cfg.splits)
    df_prompt = splits["prompt"]

    data2text = get_data2text_function(data_cfg.data2text_function)
    data_text = data2text(
        df_prompt,
        id_col=data_cfg.id_column,
        template=data_cfg.narrative_template,
        value_mappings=getattr(data_cfg, "value_mappings", None),
    )
    prompt_builder = PromptBuilderWrapper(cfg, data_text, df_prompt)
    _ = prompt_builder

    max_iterations = getattr(cfg.loop, "max_iterations", 10)
    barrier_timeout = (
        getattr(getattr(cfg.judge, "barrier", None), "orchestrator_wait_seconds", 1800)
        if hasattr(cfg, "judge")
        else 1800
    )
    retry_buffer = (
        getattr(getattr(cfg.judge, "barrier", None), "retry_wait_seconds", 300)
        if hasattr(cfg, "judge")
        else 300
    )

    console.print(
        Panel(
            f"[bold]Max Iterations:[/] {max_iterations}\n"
            f"[bold]N Clients Expected:[/] {resolved_n_clients}\n"
            f"[bold]Barrier Timeout:[/] {barrier_timeout}s\n"
            f"[bold]Results Dir:[/] {resolved_results_dir}",
            style="cyan",
        )
    )

    for iteration in range(max_iterations):
        console.rule(f"[bold]Judge Orchestrator - Iteration {iteration}")

        console.print(
            f"[cyan]Waiting for {resolved_n_clients} clients to complete iteration {iteration}...[/]"
        )
        count = registry.wait_for_clients_complete(
            iteration=iteration,
            n_expected=resolved_n_clients,
            timeout_seconds=barrier_timeout + retry_buffer,
            poll_seconds=5.0,
        )
        console.print(f"[green]Iteration {iteration} complete: {count} clients[/]")

        clients_with_models = registry.count_clients_with_models(iteration)
        console.print(f"[dim]Clients with runnable models: {clients_with_models}[/]")
        if clients_with_models == 0:
            console.print("[yellow]No clients produced runnable models, skipping judge[/]")
            if cmg_enabled:
                generator_name = getattr(cmg_cfg, "generator_client", "generator")
                fallback_feedback = {
                    generator_name: "All evaluator-assigned models failed syntax validation "
                    "after retries. Review error messages and try a different approach."
                }
            else:
                fallback_feedback = {
                    "default": "All models failed syntax validation after retries. "
                    "Review error messages and try a different approach."
                }
            registry.set_judge_feedback(
                iteration=iteration,
                synthesized_feedback=fallback_feedback,
                verdict_payload={"skipped": True, "reason": "all_syntax_failures"},
            )
            continue

        console.print("[cyan]Updating unified diagnostic store...[/]")
        try:
            unified_store = rebuild_from_artifacts(
                results_dir=str(resolved_results_dir),
                db_path=str(resolved_results_dir / "diagnostics_unified.duckdb"),
                overwrite=False,
                iterations=[iteration],
            )
        except Exception as exc:
            console.print(
                f"[red]Failed to update diagnostic store incrementally: {exc}[/]\n"
                f"[yellow]Falling back to full rebuild of all artifacts...[/]"
            )
            try:
                unified_store = rebuild_from_artifacts(
                    results_dir=str(resolved_results_dir),
                    db_path=str(resolved_results_dir / "diagnostics_unified.duckdb"),
                    overwrite=True,
                )
                console.print("[green]Full rebuild successful[/]")
            except Exception as fallback_exc:
                console.print(
                    f"[red]Fallback rebuild also failed: {fallback_exc}[/]\n"
                    f"[red]Writing failure entry to registry; clients will halt.[/]"
                )
                registry.set_judge_failure(
                    iteration=iteration,
                    error=f"Diagnostic store rebuild failed: {fallback_exc}",
                )
                continue

        console.print("[cyan]Running centralized judge...[/]")
        max_judge_retries = 2
        registry_snapshot = registry.read()
        global_best = registry_snapshot.get("global_best") or {}
        for attempt in range(max_judge_retries + 1):
            try:
                judge = ToolUsingJudge(
                    cfg=cfg,
                    diagnostic_store=unified_store,
                    model=model,
                    tokenizer=tokenizer,
                    results_dir=resolved_results_dir,
                )

                artifact = run_orchestrated_judge_pipeline(
                    judge=judge,
                    cfg=cfg,
                    results_dir=resolved_results_dir,
                    iteration=iteration,
                    run_idx=0,
                    tag="_orchestrator",
                    best_model=global_best.get("model_code"),
                    best_metric=global_best.get("metric_value"),
                    recovery_failures=None,
                    prev_had_success=True,
                )
                trace_file = (
                    resolved_results_dir
                    / "judge"
                    / f"iter{iteration}_orchestrator_run0.json"
                )
                console.print(f"[cyan]Trace saved to {trace_file}[/]")

                verdict_payload = {
                    "iteration": iteration,
                    "n_clients": count,
                    "timestamp": time.time(),
                }
                registry.set_judge_feedback(
                    iteration=iteration,
                    synthesized_feedback=artifact.synthesized_feedback,
                    verdict_payload=verdict_payload,
                )

                console.print(
                    f"[green]Judge complete for iteration {iteration}; shared feedback written to registry[/]"
                )
                break

            except Exception as exc:
                if attempt < max_judge_retries:
                    console.print(
                        f"  [yellow]Judge attempt {attempt + 1} failed: {exc}. Retrying...[/]"
                    )
                    time.sleep(10)
                else:
                    console.print(
                        f"  [red]Judge failed after {max_judge_retries + 1} attempts: {exc}[/]\n"
                        f"  [red]Writing failure entry to registry; clients will halt.[/]"
                    )
                    registry.set_judge_failure(iteration=iteration, error=str(exc))

    console.rule("[green]Orchestrator Complete")
    return None


def main(args: argparse.Namespace) -> int | None:
    """Run the orchestrator command from parsed CLI arguments."""
    return run_orchestrator(
        config=args.config,
        vllm_url=args.vllm_url,
        results_dir=args.results_dir,
        n_clients=args.n_clients,
    )
