"""CLI route for the judge orchestrator."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.panel import Panel

from config.schema import load_config
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
        for attempt in range(max_judge_retries + 1):
            try:
                judge = ToolUsingJudge(
                    cfg=cfg,
                    diagnostic_store=unified_store,
                    model=model,
                    tokenizer=tokenizer,
                    results_dir=resolved_results_dir,
                )

                judge_start_time = time.time()
                analysis_data = judge.get_feedback_analysis(
                    iteration=iteration,
                    run_idx=0,
                    tag="_orchestrator",
                    best_model=None,
                    best_metric=None,
                    recovery_failures=None,
                    prev_had_success=True,
                )

                if analysis_data.get("short_circuit"):
                    if cmg_enabled:
                        generator_name = getattr(cmg_cfg, "generator_client", "generator")
                        synthesized_feedback = {
                            generator_name: analysis_data["analysis_text"]
                        }
                    else:
                        synthesized_feedback = {"default": analysis_data["analysis_text"]}
                    last_verdict_dict = {}
                    all_recommendations = []
                else:
                    synthesized_feedback = {}
                    last_verdict_dict = {}
                    all_recommendations = []
                    clients = getattr(cfg, "clients", {}) or {}
                    if not isinstance(clients, dict):
                        clients = {
                            name: getattr(clients, name)
                            for name in vars(clients).keys()
                            if not name.startswith("_")
                        }

                    if cmg_enabled:
                        generator_name = getattr(cmg_cfg, "generator_client", "generator")
                        persona_config = clients.get(generator_name) if clients else None
                        persona_suffix = ""
                        if persona_config and hasattr(persona_config, "llm"):
                            persona_suffix = getattr(
                                persona_config.llm, "feedback_guidance", None
                            ) or getattr(persona_config.llm, "system_prompt_suffix", "")
                        feedback_text, verdict_dict = judge.synthesize_for_persona(
                            analysis_data,
                            persona_name=generator_name,
                            persona_suffix=persona_suffix,
                            persona_config=persona_config,
                        )
                        synthesized_feedback[generator_name] = feedback_text
                        last_verdict_dict = verdict_dict
                        if verdict_dict.get("key_recommendations"):
                            all_recommendations = verdict_dict["key_recommendations"]
                    elif clients:
                        for persona_name, persona_config in clients.items():
                            persona_suffix = ""
                            if persona_config and hasattr(persona_config, "llm"):
                                persona_suffix = getattr(
                                    persona_config.llm, "feedback_guidance", None
                                ) or getattr(persona_config.llm, "system_prompt_suffix", "")

                            feedback_text, verdict_dict = judge.synthesize_for_persona(
                                analysis_data,
                                persona_name=persona_name,
                                persona_suffix=persona_suffix,
                                persona_config=persona_config,
                            )
                            synthesized_feedback[persona_name] = feedback_text
                            last_verdict_dict = verdict_dict
                            if verdict_dict.get("key_recommendations"):
                                all_recommendations.extend(
                                    verdict_dict["key_recommendations"]
                                )
                    else:
                        feedback_text, verdict_dict = judge.synthesize_for_persona(
                            analysis_data,
                            persona_name="default",
                            persona_suffix="",
                        )
                        synthesized_feedback["default"] = feedback_text
                        last_verdict_dict = verdict_dict
                        if verdict_dict.get("key_recommendations"):
                            all_recommendations = verdict_dict["key_recommendations"]

                total_wall_time = time.time() - judge_start_time

                judge_dir = resolved_results_dir / "judge"
                judge_dir.mkdir(parents=True, exist_ok=True)

                seen_recs = set()
                unique_recommendations = []
                for recommendation in all_recommendations:
                    if recommendation not in seen_recs:
                        seen_recs.add(recommendation)
                        unique_recommendations.append(recommendation)

                raw_per_angle = (
                    last_verdict_dict.get("per_angle", []) if last_verdict_dict else []
                )
                per_angle = [
                    angle.model_dump() if hasattr(angle, "model_dump") else angle
                    for angle in raw_per_angle
                ]

                trace_payload = {
                    "iteration": iteration,
                    "run_idx": 0,
                    "tag": "_orchestrator",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "tool_call_count": len(analysis_data.get("trace", [])),
                    "wall_time_seconds": total_wall_time,
                    "best_bic": analysis_data.get("best_bic"),
                    "tool_call_trace": analysis_data.get("trace", []),
                    "full_trace": analysis_data.get("full_trace", []),
                    "per_angle": per_angle,
                    "key_recommendations": unique_recommendations[:5],
                    "synthesized_feedback": synthesized_feedback,
                    "stuck_search": analysis_data.get("is_stuck", False),
                    "personas": list(synthesized_feedback.keys()),
                }
                if analysis_data.get("short_circuit"):
                    trace_payload["short_circuit"] = True

                trace_file = judge_dir / f"iter{iteration}_orchestrator_run0.json"
                with trace_file.open("w", encoding="utf-8") as file_obj:
                    json.dump(trace_payload, file_obj, indent=2, default=str)

                console.print(f"[cyan]Trace saved to {trace_file}[/]")

                verdict_payload = {
                    "iteration": iteration,
                    "n_clients": count,
                    "timestamp": time.time(),
                }
                registry.set_judge_feedback(
                    iteration=iteration,
                    synthesized_feedback=synthesized_feedback,
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
