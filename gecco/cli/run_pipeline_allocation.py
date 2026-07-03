"""Internal CLI route for the per-pipeline allocation runner.

Started by ``bash/run_pipeline_allocation.sh`` inside a SLURM job allocation.
Resolves the config, starts all clients concurrently as subprocesses, waits for
them, then runs the judge/orchestrator and test-evaluation stages in order.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config.schema import get_judge_mode, load_config
from gecco.cli.config_paths import resolve_config_path
from gecco.sentry_init import init_sentry
from gecco.tempdirs import configure_temp_dirs
from gecco.utils import TimestampedConsole


console = TimestampedConsole()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def register_parser(subparsers) -> argparse.ArgumentParser:
    """Register the internal pipeline-allocation subcommand."""
    parser = subparsers.add_parser(
        "pipeline-allocation", help=argparse.SUPPRESS, description=argparse.SUPPRESS
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--profiles-csv", type=str, default=None)
    parser.add_argument("--vllm-url", type=str, default=None)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.set_defaults(handler=main)
    return parser


def _resolve_profiles(cfg, profiles_csv: str | None) -> list[str]:
    """Return the list of client profile names to launch."""
    if profiles_csv:
        return [p for p in profiles_csv.split(",") if p]
    clients = getattr(cfg, "clients", {}) or {}
    return list(clients.keys())


def _needs_orchestrator(cfg) -> bool:
    """Check whether the config requires a judge orchestrator."""
    judge = getattr(cfg, "judge", None)
    if judge is None:
        return False
    return get_judge_mode(cfg) != "off"


def _needs_test_evaluation(cfg) -> bool:
    """Check whether test evaluation should run."""
    cmg = getattr(cfg, "centralized_model_generation", None)
    if cmg and getattr(cmg, "enabled", False):
        return getattr(cmg, "run_final_evaluation", True)
    return True


def run_pipeline_allocation(
    *,
    config: str,
    profiles_csv: str | None = None,
    vllm_url: str | None = None,
    results_dir: str = "",
) -> int:
    """Run one full pipeline inside a SLURM job allocation.

    Stages
    ------
    1. Start all client subprocesses concurrently.
    2. Wait for all clients; if any fails, return nonzero.
    3. Run judge/orchestrator if the config requires it.
    4. Run test evaluation after successful client/judge stages.
    """
    configure_temp_dirs(PROJECT_ROOT, prefix="pipeline-allocation")

    # Resolve config
    config_path = resolve_config_path(config, project_root=PROJECT_ROOT)
    if not config_path.exists():
        console.print(f"[red]ERROR: Config not found: {config_path}[/]")
        return 1

    cfg = load_config(config_path)

    # Reject CMG configs explicitly at the runner level — CMG has its own
    # multi-stage sbatch plan (generator + evaluators + orchestrator) that
    # does not map to a single allocation job.  This check fires even when
    # *both* ``clients`` and ``--profiles-csv`` are present.
    cmg_cfg = getattr(cfg, "centralized_model_generation", None)
    if cmg_cfg and getattr(cmg_cfg, "enabled", False):
        console.print(
            "[red]ERROR: Centralized model generation (CMG) configs are not "
            "supported in pipeline-allocation mode. Use the default distributed "
            "or distributed-batch launcher for CMG pipelines.[/]"
        )
        return 1

    init_sentry(
        cfg=cfg,
        task_name=cfg.task.name,
        config_name=config,
    )

    # Resolve results directory
    resolved_results_dir = Path(results_dir)
    if not resolved_results_dir.is_absolute():
        resolved_results_dir = PROJECT_ROOT / resolved_results_dir

    # Resolve client profiles
    profiles = _resolve_profiles(cfg, profiles_csv)
    if not profiles:
        console.print("[red]ERROR: No client profiles found in config or --profiles-csv[/]")
        return 1

    clients_text = ", ".join(profiles) if profiles else "(none)"
    console.print(f"[cyan]Pipeline allocation: config={config}[/]")
    console.print(f"[cyan]  profiles=[{clients_text}][/]")
    console.print(f"[cyan]  results-dir={resolved_results_dir}[/]")
    if vllm_url:
        console.print(f"[cyan]  vllm-url={vllm_url}[/]")

    # Build the base command
    python_cmd = [sys.executable, "-m", "gecco", "internal"]

    vllm_args = ["--vllm-url", vllm_url] if vllm_url else []
    results_args = ["--results-dir", str(results_dir)]

    # ── Stage 1: Start all clients concurrently ──
    processes: list[subprocess.Popen] = []
    for profile in profiles:
        client_cmd = python_cmd + [
            "distributed-client",
            "--config", config,
            "--client-profile", profile,
        ] + vllm_args + results_args
        console.print(f"  [dim]Starting client {profile}...[/]")
        proc = subprocess.Popen(
            client_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(proc)

    console.print(f"[cyan]Started {len(processes)} client(s), waiting for completion...[/]")

    # ── Stage 2: Wait for all clients ──
    client_failed = False
    for proc, profile in zip(processes, profiles):
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            console.print(f"[red]Client {profile} failed (exit code {proc.returncode})[/]")
            if stderr:
                console.print(f"[red]stderr: {stderr.strip()}[/]")
            client_failed = True
        else:
            console.print(f"[green]Client {profile} completed successfully[/]")

    if client_failed:
        console.print("[red]Pipeline allocation failed: one or more clients failed[/]")
        return 1

    # ── Stage 3: Run judge/orchestrator if required ──
    needs_orchestrator = _needs_orchestrator(cfg)
    if needs_orchestrator:
        n_clients = getattr(cfg.loop, "n_clients", len(profiles))
        console.print("[cyan]Starting judge orchestrator...[/]")
        judge_cmd = python_cmd + [
            "judge-orchestrate",
            "--config", config,
            "--n-clients", str(n_clients),
        ] + vllm_args + results_args
        proc = subprocess.Popen(
            judge_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            console.print(f"[red]Judge orchestrator failed (exit code {proc.returncode})[/]")
            if stderr:
                console.print(f"[red]stderr: {stderr.strip()}[/]")
            return 1
        console.print("[green]Judge orchestrator completed successfully[/]")

    # ── Stage 4: Run test evaluation ──
    needs_test_eval = _needs_test_evaluation(cfg)
    if needs_test_eval:
        console.print("[cyan]Starting test evaluation...[/]")
        eval_cmd = python_cmd + [
            "test-evaluation",
            "--config", config,
            "--results-dir", str(results_dir),
            "--write-store",
        ]
        proc = subprocess.Popen(
            eval_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            console.print(f"[red]Test evaluation failed (exit code {proc.returncode})[/]")
            if stderr:
                console.print(f"[red]stderr: {stderr.strip()}[/]")
            return 1
        console.print("[green]Test evaluation completed successfully[/]")

    console.print("[green]Pipeline allocation complete[/]")
    return 0


def main(args: argparse.Namespace) -> int:
    """Run the pipeline allocation command from parsed CLI arguments."""
    return run_pipeline_allocation(
        config=args.config,
        profiles_csv=args.profiles_csv,
        vllm_url=args.vllm_url,
        results_dir=args.results_dir,
    )
