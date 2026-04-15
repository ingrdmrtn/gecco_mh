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
    if args.n_clients:
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
        count = registry.wait_for_iteration(
            iteration=it,
            n_expected=n_clients,
            timeout_seconds=barrier_timeout,
            poll_seconds=5.0,
        )
        console.print(f"[green]Iteration {it} complete: {count} clients[/]")

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
                    f"[yellow]Skipping judge for iteration {it}[/]"
                )
                continue

        # --- Instantiate and run ToolUsingJudge on unified store ---
        console.print("[cyan]Running centralized judge...[/]")
        try:
            judge = ToolUsingJudge(
                cfg=cfg,
                diagnostic_store=unified_store,
                model=model,
                tokenizer=tokenizer,
                results_dir=results_dir,
            )

            # R2: Run analysis once, then synthesize for each persona
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
                synthesized_feedback = {"default": analysis_data["analysis_text"]}
            else:
                # R2: Synthesize for each persona in cfg.clients
                synthesized_feedback = {}
                clients = getattr(cfg, "clients", {})

                if clients:
                    # Enumerate personas from config
                    for persona_name in vars(clients).keys():
                        if persona_name.startswith("_"):
                            continue  # Skip private attributes

                        persona_config = getattr(clients, persona_name, None)
                        persona_suffix = ""

                        if persona_config and hasattr(persona_config, "llm"):
                            persona_suffix = getattr(
                                persona_config.llm,
                                "system_prompt_suffix",
                                ""
                            )

                        feedback_text, _ = judge.synthesize_for_persona(
                            analysis_data,
                            persona_name=persona_name,
                            persona_suffix=persona_suffix,
                        )
                        synthesized_feedback[persona_name] = feedback_text
                else:
                    # No clients defined; use default persona
                    feedback_text, _ = judge.synthesize_for_persona(
                        analysis_data,
                        persona_name="default",
                        persona_suffix="",
                    )
                    synthesized_feedback["default"] = feedback_text

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

        except Exception as e:
            console.print(
                f"[red]Judge failed for iteration {it}: {e}[/]\n"
                f"[yellow]Clients will fall back to local judge[/]"
            )
            continue

    console.rule("[green]Orchestrator Complete")


if __name__ == "__main__":
    main()
