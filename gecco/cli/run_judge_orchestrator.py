"""CLI route for the judge orchestrator."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from rich.panel import Panel

from config.schema import load_config
from gecco.construct_feedback.orchestrated import (
    build_feedback_artifact,
    persist_feedback_artifact,
    run_orchestrated_judge_pipeline,
)
from gecco.construct_feedback.tool_judge import ToolUsingJudge
from gecco.cli.config_paths import resolve_config_path, results_dir_for_config
from gecco.coordination import SharedRegistry
from gecco.diagnostic_store.store import DiagnosticStore
from gecco.load_llms.model_loader import load_llm
from gecco.prepare_data.data2text import get_data2text_function
from gecco.prepare_data.io import load_data, split_by_participant
from gecco.prompt_builder.prompt import PromptBuilderWrapper
from gecco.sentry_init import capture_operational_error, flush_sentry_events, init_sentry
from gecco.tempdirs import configure_temp_dirs
from gecco.utils import TimestampedConsole


console = TimestampedConsole()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DUCKDB_IMPORT_RETRY_ATTEMPTS = 5
_DUCKDB_IMPORT_RETRY_BASE_SECONDS = 0.5
_DUCKDB_IMPORT_RETRY_MAX_SECONDS = 4.0
_DUCKDB_LOCK_ERROR_MARKERS = ("Could not set lock", "Conflicting lock")


def _is_duckdb_lock_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker.lower() in message for marker in _DUCKDB_LOCK_ERROR_MARKERS)


def _import_source_db_with_retries(store: DiagnosticStore, source_path: Path) -> None:
    for attempt in range(_DUCKDB_IMPORT_RETRY_ATTEMPTS):
        try:
            store.import_from_source_db(source_path)
            return
        except Exception as exc:
            if not _is_duckdb_lock_error(exc) or attempt == _DUCKDB_IMPORT_RETRY_ATTEMPTS - 1:
                raise
            backoff_seconds = min(
                _DUCKDB_IMPORT_RETRY_BASE_SECONDS * (2**attempt),
                _DUCKDB_IMPORT_RETRY_MAX_SECONDS,
            )
            time.sleep(backoff_seconds)


def _build_judge_store_from_duckdb_sources(results_dir: Path) -> DiagnosticStore:
    """Build the judge evidence store from existing diagnostic DuckDB files."""
    unified_path = results_dir / "diagnostics_unified.duckdb"
    source_paths = sorted(
        path
        for path in results_dir.glob("diagnostics*.duckdb")
        if path.name != unified_path.name
    )

    if source_paths:
        if unified_path.exists():
            unified_path.unlink()
        lock_path = Path(f"{unified_path}.lock")
        if lock_path.exists():
            lock_path.unlink()
        store = DiagnosticStore(unified_path)
        for source_path in source_paths:
            _import_source_db_with_retries(store, source_path)
        return store

    if unified_path.exists():
        return DiagnosticStore(unified_path)

    raise FileNotFoundError(
        f"No diagnostics DuckDB files found in {results_dir}; expected diagnostics*.duckdb"
    )


def register_parser(subparsers) -> argparse.ArgumentParser:
    """Register the orchestrator subcommand."""
    parser = subparsers.add_parser(
        "judge-orchestrate", help=argparse.SUPPRESS, description=argparse.SUPPRESS
    )
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
    had_failure = False
    configure_temp_dirs(PROJECT_ROOT, prefix="Orchestrator")

    if vllm_url:
        os.environ["VLLM_BASE_URL"] = vllm_url

    console.print(
        Panel(
            f"[bold]Config:[/] {config}\n"
            f"[bold]vLLM URL:[/] {os.environ.get('VLLM_BASE_URL', '(not set)')}\n"
            f"[bold]Results Dir:[/] {results_dir or '(from config path)'}\n"
            f"[bold]N Clients:[/] {n_clients or '(from config)'}",
            title="Centralized Judge Orchestrator",
            style="cyan",
        )
    )

    cfg = load_config(resolve_config_path(config, project_root=PROJECT_ROOT))

    sentry_connected = init_sentry(cfg=cfg, task_name=cfg.task.name, config_name=config)
    console.print(
        "[green]Sentry monitoring: connected[/]"
        if sentry_connected
        else "[yellow]Sentry monitoring: disabled (SENTRY_DSN not set)[/]"
    )

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

    registry_path = resolved_results_dir / "shared_registry.duckdb"
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

        registry.raise_if_aborted()

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
            artifact = build_feedback_artifact(
                iteration=iteration,
                run_idx=0,
                tag="_orchestrator",
                analysis_data={
                    "trace": [],
                    "full_trace": [],
                    "best_bic": None,
                    "wall_time": 0.0,
                    "short_circuit": True,
                    "metadata": {
                        "shortcut_reason": "no_runnable_models",
                        "n_clients": count,
                        "clients_with_models": clients_with_models,
                    },
                },
                synthesized_feedback=fallback_feedback,
                verdict_payloads=[],
                best_model=None,
                best_metric=None,
                include_best_model_code=False,
            )
            persist_feedback_artifact(artifact=artifact, results_dir=resolved_results_dir)
            registry.set_judge_feedback(
                iteration=iteration,
                synthesized_feedback=artifact.synthesized_feedback,
                verdict_payload={"skipped": True, "reason": "all_syntax_failures"},
            )
            continue

        console.print("[cyan]Loading unified diagnostic store from DuckDB sources...[/]")
        try:
            unified_store = _build_judge_store_from_duckdb_sources(resolved_results_dir)
        except Exception as exc:
            console.print(
                f"[red]Failed to load diagnostic DuckDB sources: {exc}[/]\n"
                f"[red]Writing failure entry to registry; clients will halt.[/]"
            )
            capture_operational_error(
                exc,
                component="judge_orchestrator",
                operation="load_duckdb_store",
                iteration=iteration,
                results_dir=str(resolved_results_dir),
            )
            registry.set_judge_failure(
                iteration=iteration,
                error=f"Diagnostic DuckDB load failed: {exc}",
            )
            had_failure = True
            break

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
                    capture_operational_error(
                        exc,
                        component="judge_orchestrator",
                        operation="final_judge_retry",
                        iteration=iteration,
                        results_dir=str(resolved_results_dir),
                        attempt=attempt + 1,
                        max_attempts=max_judge_retries + 1,
                        exception_type=type(exc).__name__,
                    )
                    flush_sentry_events()
                    registry.set_judge_failure(iteration=iteration, error=str(exc))
                    had_failure = True
                    break

        if had_failure:
            break

    console.rule("[green]Orchestrator Complete")
    return 1 if had_failure else None


def main(args: argparse.Namespace) -> int | None:
    """Run the orchestrator command from parsed CLI arguments."""
    return run_orchestrator(
        config=args.config,
        vllm_url=args.vllm_url,
        results_dir=args.results_dir,
        n_clients=args.n_clients,
    )
