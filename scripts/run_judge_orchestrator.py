#!/usr/bin/env python3
"""
Centralized judge orchestrator for distributed GeCCo runs.

This process:
1. Waits for all clients to complete each iteration
2. Rebuilds the unified diagnostic store from all clients' artifacts
3. Runs a single centralized ToolUsingJudge pass on the unified data
4. Writes the shared verdict to the registry for all clients to consume

Clients wait for this shared feedback instead of each running their own judge.
This reduces total judge cost from N × cost to 1 × cost per iteration.
"""

import os
import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path
import argparse
from rich.console import Console
from rich.panel import Panel

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.schema import load_config
from gecco.coordination import SharedRegistry
from gecco.diagnostic_store.rebuild import rebuild_from_artifacts
from gecco.construct_feedback.tool_judge import ToolUsingJudge
from gecco.load_llms.model_loader import load_llm
from gecco.prepare_data.io import load_data, split_by_participant
from gecco.prepare_data.data2text import get_data2text_function
from gecco.prompt_builder.prompt import PromptBuilderWrapper
from gecco.sentry_init import init_sentry
from gecco.utils import TimestampedConsole

console = TimestampedConsole()


def main():
    parser = argparse.ArgumentParser(description="Centralized judge orchestrator")
    parser.add_argument(
        "--config", type=str, required=True, help="Config YAML file name"
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=None,
        help="vLLM server URL (overrides $VLLM_BASE_URL)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory (overrides task name from config)",
    )
    parser.add_argument(
        "--n-clients",
        type=int,
        default=None,
        help="Number of clients expected (overrides config)",
    )
    args = parser.parse_args()

    # --- Set vLLM URL if provided (before load_llm reads it) ---
    if args.vllm_url:
        os.environ["VLLM_BASE_URL"] = args.vllm_url

    console.print(
        Panel(
            f"[bold]Config:[/] {args.config}\n"
            f"[bold]vLLM URL:[/] {os.environ.get('VLLM_BASE_URL', '(not set)')}\n"
            f"[bold]Results Dir:[/] {args.results_dir or '(from task name)'}\n"
            f"[bold]N Clients:[/] {args.n_clients or '(from config)'}",
            title="Centralized Judge Orchestrator",
            style="cyan",
        )
    )

    # --- Load configuration ---
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "config" / args.config)

    # --- Initialize Sentry ---
    init_sentry(
        cfg=cfg,
        task_name=cfg.task.name,
        config_name=args.config,
    )

    # --- Determine results directory ---
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path("results") / cfg.task.name

    # --- Determine number of clients ---
    cmg_cfg = getattr(cfg, "centralized_model_generation", None)
    cmg_enabled = cmg_cfg is not None and getattr(cmg_cfg, "enabled", False)

    if cmg_enabled:
        generator_client = str(getattr(cmg_cfg, "generator_client", ""))
        if not generator_client:
            console.print(
                "[red]Error: centralized_model_generation.generator_client is required[/]"
            )
            sys.exit(1)
        if generator_client.isdigit() or generator_client.lstrip("-").isdigit():
            console.print(
                "[red]Error: centralized_model_generation.generator_client must be "
                "a named profile, not a numeric evaluator ID[/]"
            )
            sys.exit(1)
        n_clients = getattr(cmg_cfg, "n_models", None)
        if not isinstance(n_clients, int) or n_clients <= 0:
            console.print(
                "[red]Error: centralized_model_generation.n_models must be a positive integer[/]"
            )
            sys.exit(1)
    elif args.n_clients:
        n_clients = args.n_clients
    elif hasattr(cfg, "loop") and hasattr(cfg.loop, "n_clients") and cfg.loop.n_clients:
        n_clients = cfg.loop.n_clients
    else:
        console.print(
            "[red]Error: n_clients not specified in config.loop.n_clients "
            "and not provided via --n-clients[/]"
        )
        sys.exit(1)

    # --- Create/open shared registry ---
    registry_path = results_dir / "shared_registry.json"
    registry = SharedRegistry(str(registry_path))

    # --- Load model and tokenizer (for ToolUsingJudge) ---
    console.print("[cyan]Loading LLM model...[/]")
    model, tokenizer = load_llm(cfg.llm.provider, cfg.llm.base_model)

    # --- Load data (for prompt building, though not strictly needed for judge) ---
    console.print("[cyan]Loading data...[/]")
    data_cfg = cfg.data
    df = load_data(data_cfg.path, data_cfg.input_columns)
    splits = split_by_participant(df, data_cfg.id_column, data_cfg.splits)
    df_prompt = splits["prompt"]

    # Convert data to narrative
    data2text = get_data2text_function(data_cfg.data2text_function)
    data_text = data2text(
        df_prompt,
        id_col=data_cfg.id_column,
        template=data_cfg.narrative_template,
        value_mappings=getattr(data_cfg, "value_mappings", None),
    )

    # Build prompt (for judge context)
    prompt_builder = PromptBuilderWrapper(cfg, data_text, df_prompt)

    # --- Determine max iterations and run loop ---
    max_iterations = getattr(cfg.loop, "max_iterations", 10)
    barrier_timeout = (
        getattr(getattr(cfg.judge, "barrier", None), "orchestrator_wait_seconds", 1800)
        if hasattr(cfg, "judge")
        else 1800
    )
    # Extra timeout buffer to account for client retries
    retry_buffer = (
        getattr(getattr(cfg.judge, "barrier", None), "retry_wait_seconds", 300)
        if hasattr(cfg, "judge")
        else 300
    )

    console.print(
        Panel(
            f"[bold]Max Iterations:[/] {max_iterations}\n"
            f"[bold]N Clients Expected:[/] {n_clients}\n"
            f"[bold]Barrier Timeout:[/] {barrier_timeout}s\n"
            f"[bold]Results Dir:[/] {results_dir}",
            style="cyan",
        )
    )

    # --- Main orchestrator loop ---
    for it in range(max_iterations):
        console.rule(f"[bold]Judge Orchestrator - Iteration {it}")

        # --- Wait for all clients to complete iteration it ---
        console.print(
            f"[cyan]Waiting for {n_clients} clients to complete iteration {it}...[/]"
        )
        # Use wait_for_clients_complete to account for retry scenarios
        count = registry.wait_for_clients_complete(
            iteration=it,
            n_expected=n_clients,
            timeout_seconds=barrier_timeout + retry_buffer,
            poll_seconds=5.0,
        )
        console.print(f"[green]Iteration {it} complete: {count} clients[/]")

        # --- Check if any clients produced runnable models ---
        clients_with_models = registry.count_clients_with_models(it)
        console.print(f"[dim]Clients with runnable models: {clients_with_models}[/]")
        if clients_with_models == 0:
            console.print(
                "[yellow]No clients produced runnable models, skipping judge[/]"
            )
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
                iteration=it,
                synthesized_feedback=fallback_feedback,
                verdict_payload={"skipped": True, "reason": "all_syntax_failures"},
            )
            continue

        # --- Rebuild unified diagnostic store from all clients' artifacts ---
        console.print("[cyan]Updating unified diagnostic store...[/]")
        try:
            # Use incremental rebuild: only process current iteration, keep existing DB
            unified_store = rebuild_from_artifacts(
                results_dir=str(results_dir),
                db_path=str(results_dir / "diagnostics_unified.duckdb"),
                overwrite=False,  # Safe on first run and safe after restart
                iterations=[it],
            )
        except Exception as e:
            console.print(
                f"[red]Failed to update diagnostic store incrementally: {e}[/]\n"
                f"[yellow]Falling back to full rebuild of all artifacts...[/]"
            )
            # Fallback: delete and rebuild everything
            try:
                unified_store = rebuild_from_artifacts(
                    results_dir=str(results_dir),
                    db_path=str(results_dir / "diagnostics_unified.duckdb"),
                    overwrite=True,
                )
                console.print("[green]Full rebuild successful[/]")
            except Exception as fallback_e:
                console.print(
                    f"[red]Fallback rebuild also failed: {fallback_e}[/]\n"
                    f"[red]Writing failure entry to registry; clients will halt.[/]"
                )
                registry.set_judge_failure(
                    iteration=it,
                    error=f"Diagnostic store rebuild failed: {fallback_e}",
                )
                continue

        # --- Check if no_tools lesion is active ---
        no_tools_lesion_active = (
            getattr(cfg, "judge", None) is not None
            and getattr(cfg.judge, "lesion", None) is not None
            and getattr(cfg.judge.lesion, "enabled", False)
            and getattr(cfg.judge.lesion, "lesion_type", None) == "no_tools"
        )

        # --- Instantiate and run ToolUsingJudge on unified store ---
        console.print("[cyan]Running centralized judge...[/]")
        MAX_JUDGE_RETRIES = 2
        last_judge_error = None
        for attempt in range(MAX_JUDGE_RETRIES + 1):
            try:
                judge = ToolUsingJudge(
                    cfg=cfg,
                    diagnostic_store=unified_store,
                    model=model,
                    tokenizer=tokenizer,
                    results_dir=results_dir,
                )

                # Disable tool loop when no_tools lesion is active
                if no_tools_lesion_active:
                    console.print(
                        "[cyan]no_tools lesion active — disabling tool loop, "
                        "using pre-computed context only[/]"
                    )
                    judge._tool_loop = None

                # R2: Run analysis once, then synthesize for each persona
                # Start timer for wall time tracking
                judge_start_time = time.time()

                analysis_data = judge.get_feedback_analysis(
                    iteration=it,
                    run_idx=0,
                    tag="_orchestrator",
                    best_model=None,  # Not used by judge
                    best_metric=None,
                    recovery_failures=None,
                    prev_had_success=True,
                )

                # Check for short-circuit
                if analysis_data.get("short_circuit"):
                    # Short-circuit: reuse previous verdict for all personas
                    if cmg_enabled:
                        generator_name = getattr(cmg_cfg, "generator_client", "generator")
                        synthesized_feedback = {generator_name: analysis_data["analysis_text"]}
                    else:
                        synthesized_feedback = {"default": analysis_data["analysis_text"]}
                    last_verdict_dict = {}
                    all_recommendations = []
                else:
                    # R2: Synthesize for each persona in cfg.clients
                    synthesized_feedback = {}
                    last_verdict_dict = {}
                    all_recommendations = []
                    clients = getattr(cfg, "clients", {})

                    if cmg_enabled:
                        # CMG mode: synthesize for the generator persona only
                        generator_name = getattr(cmg_cfg, "generator_client", "generator")
                        persona_config = getattr(clients, generator_name, None) if clients else None
                        persona_suffix = ""
                        if persona_config and hasattr(persona_config, "llm"):
                            persona_suffix = getattr(
                                persona_config.llm, "feedback_guidance", None
                            ) or getattr(
                                persona_config.llm, "system_prompt_suffix", ""
                            )
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
                        # Enumerate personas from config
                        for persona_name in vars(clients).keys():
                            if persona_name.startswith("_"):
                                continue  # Skip private attributes

                            persona_config = getattr(clients, persona_name, None)
                            persona_suffix = ""
                            if persona_config and hasattr(persona_config, "llm"):
                                # Prefer feedback_guidance if present; fall back to system_prompt_suffix
                                persona_suffix = getattr(
                                    persona_config.llm, "feedback_guidance", None
                                ) or getattr(
                                    persona_config.llm, "system_prompt_suffix", ""
                                )

                            feedback_text, verdict_dict = judge.synthesize_for_persona(
                                analysis_data,
                                persona_name=persona_name,
                                persona_suffix=persona_suffix,
                                persona_config=persona_config,
                            )
                            synthesized_feedback[persona_name] = feedback_text
                            last_verdict_dict = verdict_dict
                            # Collect recommendations from this persona
                            if verdict_dict.get("key_recommendations"):
                                all_recommendations.extend(
                                    verdict_dict["key_recommendations"]
                                )
                    else:
                        # No clients defined; use default persona
                        feedback_text, verdict_dict = judge.synthesize_for_persona(
                            analysis_data,
                            persona_name="default",
                            persona_suffix="",
                        )
                        synthesized_feedback["default"] = feedback_text
                        last_verdict_dict = verdict_dict
                        if verdict_dict.get("key_recommendations"):
                            all_recommendations = verdict_dict["key_recommendations"]

                # Calculate total wall time
                total_wall_time = time.time() - judge_start_time

                # --- Write trace file to results/<task>/judge/ ---
                judge_dir = results_dir / "judge"
                judge_dir.mkdir(parents=True, exist_ok=True)

                # Deduplicate recommendations across personas
                seen_recs = set()
                unique_recommendations = []
                for rec in all_recommendations:
                    if rec not in seen_recs:
                        seen_recs.add(rec)
                        unique_recommendations.append(rec)

                # Get per_angle from last verdict (they're similar across personas)
                raw_per_angle = (
                    last_verdict_dict.get("per_angle", []) if last_verdict_dict else []
                )
                # Ensure per_angle entries are plain dicts (Pydantic models need .model_dump())
                per_angle = [
                    a.model_dump() if hasattr(a, "model_dump") else a
                    for a in raw_per_angle
                ]

                trace_payload = {
                    "iteration": it,
                    "run_idx": 0,
                    "tag": "_orchestrator",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "tool_call_count": len(analysis_data.get("trace", [])),
                    "wall_time_seconds": total_wall_time,
                    "best_bic": analysis_data.get("best_bic"),
                    "tool_call_trace": analysis_data.get("trace", []),
                    "full_trace": analysis_data.get("full_trace", []),
                    "per_angle": per_angle,
                    "key_recommendations": unique_recommendations[
                        :5
                    ],  # Top 5 deduplicated
                    "synthesized_feedback": synthesized_feedback,  # Dict keyed by persona name
                    "stuck_search": analysis_data.get("is_stuck", False),
                    "personas": list(synthesized_feedback.keys()),
                }

                # Add short_circuit flag if present
                if analysis_data.get("short_circuit"):
                    trace_payload["short_circuit"] = True

                trace_file = judge_dir / f"iter{it}_orchestrator_run0.json"
                with open(trace_file, "w") as f:
                    json.dump(trace_payload, f, indent=2, default=str)

                console.print(f"[cyan]Trace saved to {trace_file}[/]")

                # --- Write shared feedback to registry ---
                verdict_payload = {
                    "iteration": it,
                    "n_clients": count,
                    "timestamp": time.time(),
                }

                registry.set_judge_feedback(
                    iteration=it,
                    synthesized_feedback=synthesized_feedback,
                    verdict_payload=verdict_payload,
                )

                console.print(
                    f"[green]Judge complete for iteration {it}; "
                    f"shared feedback written to registry[/]"
                )
                break  # Success — exit retry loop

            except Exception as e:
                last_judge_error = e
                if attempt < MAX_JUDGE_RETRIES:
                    console.print(
                        f"  [yellow]Judge attempt {attempt + 1} failed: {e}. Retrying...[/]"
                    )
                    time.sleep(10)
                else:
                    console.print(
                        f"  [red]Judge failed after {MAX_JUDGE_RETRIES + 1} attempts: {e}[/]\n"
                        f"  [red]Writing failure entry to registry; clients will halt.[/]"
                    )
                    registry.set_judge_failure(iteration=it, error=str(e))

    console.rule("[green]Orchestrator Complete")


if __name__ == "__main__":
    main()
