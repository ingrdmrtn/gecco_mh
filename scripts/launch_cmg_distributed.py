#!/usr/bin/env python3
"""
Launch a centralized model generation (CMG) distributed GeCCo search.

Launches one generator client, N evaluator clients (numeric client IDs 0..N-1),
and the orchestrated judge.

Usage:
    python scripts/launch_cmg_distributed.py --config two_step_factors_cmg.yaml

    # Dry run (show commands without executing)
    python scripts/launch_cmg_distributed.py --config two_step_factors_cmg.yaml --dry-run

    # SLURM mode (generator + evaluator array + orchestrator)
    python scripts/launch_cmg_distributed.py --config two_step_factors_cmg.yaml --slurm
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.schema import load_config
from rich.console import Console
from rich.panel import Panel


def _get_slurm_defaults(config_path):
    """Read slurm.cpus_per_task, slurm.partition, and slurm.mem_per_task from config."""
    import yaml

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    slurm = raw.get("slurm", {})
    return {
        "cpus_per_task": slurm.get("cpus_per_task"),
        "partition": slurm.get("partition"),
        "mem_per_task": slurm.get("mem_per_task"),
    }

console = Console()


def run_cmd(cmd, dry_run=False):
    """Run a shell command, or just print it if dry_run."""
    print(f"  $ {cmd}")
    if dry_run:
        return None
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        sys.exit(1)
    output = result.stdout.strip()
    if output.startswith("Submitted batch job "):
        return output.split()[-1]
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Launch CMG distributed GeCCo search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config YAML filename (relative to config/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Submit SLURM jobs instead of running locally",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=None,
        help="vLLM server URL (passed to all jobs)",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default=None,
        help="Conda environment to activate",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default=None,
        help="SLURM partition (e.g. gpu, cpu)",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=None,
        help="CPUs per evaluator task (SLURM; overrides config slurm.cpus_per_task, default: 48)",
    )
    parser.add_argument(
        "--mem",
        type=str,
        default=None,
        help="Memory per node, e.g. 64G (overrides config slurm.mem_per_task)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config" / args.config

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)
    cmg_cfg = getattr(cfg, "centralized_model_generation", None)

    if cmg_cfg is None or not getattr(cmg_cfg, "enabled", False):
        print("ERROR: CMG not enabled in config (centralized_model_generation.enabled: true)")
        sys.exit(1)

    generator_client = str(getattr(cmg_cfg, "generator_client", ""))
    if not generator_client:
        print("ERROR: centralized_model_generation.generator_client is required")
        sys.exit(1)
    if generator_client.isdigit() or generator_client.lstrip("-").isdigit():
        print("ERROR: centralized_model_generation.generator_client must be "
              "a named profile, not a numeric evaluator ID")
        sys.exit(1)

    n_models = getattr(cmg_cfg, "n_models", None)
    if not isinstance(n_models, int) or n_models <= 0:
        print("ERROR: centralized_model_generation.n_models must be a positive integer")
        sys.exit(1)

    # Resolve SLURM defaults from config (CLI overrides)
    slurm_defaults = _get_slurm_defaults(config_path)
    cpus_per_task = args.cpus_per_task or slurm_defaults.get("cpus_per_task") or 48
    partition = args.partition or slurm_defaults.get("partition")
    partition_flag = f"--partition={partition}" if partition else ""
    mem = args.mem or slurm_defaults.get("mem_per_task")
    mem_flag = f"--mem={mem}" if mem else ""

    task_name = getattr(cfg.task, "name", "unknown")
    config_name = args.config

    vllm_url = args.vllm_url or os.environ.get("VLLM_BASE_URL", "")
    vllm_url_arg = f'--vllm-url "{vllm_url}"' if vllm_url else ""

    console.print(
        Panel(
            f"[bold]Config:[/] {config_name}\n"
            f"[bold]Task:[/] {task_name}\n"
            f"[bold]Generator Client:[/] {generator_client}\n"
            f"[bold]Evaluators:[/] {n_models} (IDs 0..{n_models - 1})\n"
            f"[bold]vLLM URL:[/] {vllm_url or '(not set)'}\n"
            f"[bold]Mode:[/] {'SLURM' if args.slurm else 'local process'}",
            title="CMG Distributed Launch Plan",
            style="green",
        )
    )

    script_dir = project_root / "scripts"
    base_cmd = f"python {script_dir / 'run_gecco_distributed.py'} --config {config_name}"

    print()
    print("=" * 60)
    print("LAUNCH PLAN")
    print("=" * 60)

    # 1. Generator command
    gen_cmd = f'{base_cmd} --client-profile {generator_client} {vllm_url_arg}'
    print(f"\n[Generator] {generator_client}:")
    print(f"  {gen_cmd}")

    # 2. Evaluator commands
    print(f"\n[Evaluators] {n_models} clients:")
    for i in range(n_models):
        eval_cmd = f'{base_cmd} --client-id {i} {vllm_url_arg}'
        print(f"  Evaluator {i}: {eval_cmd}")

    # 3. Judge orchestrator command
    orch_cmd = f'python {script_dir / "run_judge_orchestrator.py"} --config {config_name} {vllm_url_arg}'
    print(f"\n[Judge Orchestrator]:")
    print(f"  {orch_cmd}")

    print()

    if args.dry_run and not args.slurm:
        print("[Dry run] Commands printed above — not executing.")
        return

    if args.slurm:
        # Build conda prefix for --wrap commands
        conda_prefix = (
            f"conda run -n {args.conda_env} "
            if args.conda_env
            else ""
        )

        # --------------------------------------------------------
        # 1. Submit generator as a single non-array job
        #    Uses --wrap to call Python directly (no shell wrapper),
        #    passing --client-profile (NOT --client-id).
        # --------------------------------------------------------
        print("Submitting generator job...")
        gen_python_cmd = (
            f"python {script_dir / 'run_gecco_distributed.py'} "
            f"--config {config_name} "
            f"--client-profile {generator_client} "
            f"{vllm_url_arg}"
        )
        gen_job_cmd = (
            f"sbatch "
            f"--job-name=gecco-cmg-generator "
            f"--cpus-per-task={cpus_per_task} "
            f"{partition_flag} "
            f"{mem_flag} "
            f"--output=logs/gecco-cmg-generator-%j.out "
            f"--error=logs/gecco-cmg-generator-%j.err "
            f'--wrap="{conda_prefix}{gen_python_cmd}"'
        )
        gen_job_id = run_cmd(gen_job_cmd, dry_run=args.dry_run)

        # --------------------------------------------------------
        # 2. Submit evaluators as an array 0..n_models-1
        #    Uses the existing shell wrapper which passes
        #    --client-id "$SLURM_ARRAY_TASK_ID".
        #    Empty profiles_csv means no profile → numeric ID.
        # --------------------------------------------------------
        print("Submitting evaluator array job...")
        conda_arg = f'"{args.conda_env}"' if args.conda_env else '""'
        eval_job_cmd = (
            f"sbatch "
            f"--array=0-{n_models - 1} "
            f"--job-name=gecco-cmg-evaluator "
            f"--cpus-per-task={cpus_per_task} "
            f"{partition_flag} "
            f"{mem_flag} "
            f"--output=logs/gecco-cmg-evaluator-%A_%a.out "
            f"--error=logs/gecco-cmg-evaluator-%A_%a.err "
            f'{project_root / "bash/run_gecco_distributed.sh"} '
            f'"{config_name}" "" "{vllm_url}" {conda_arg}'
        )
        run_cmd(eval_job_cmd, dry_run=args.dry_run)

        # --------------------------------------------------------
        # 3. Submit orchestrator
        # --------------------------------------------------------
        print("Submitting orchestrator job...")
        orch_job_cmd = (
            f"sbatch "
            f"--job-name=gecco-cmg-orchestrator "
            f"--cpus-per-task=8 "
            f"{partition_flag} "
            f"--mem=16G "
            f"--output=logs/gecco-cmg-orchestrator-%j.out "
            f"--error=logs/gecco-cmg-orchestrator-%j.err "
            f'{project_root / "bash/run_judge_orchestrator.sh"} '
            f'"{config_name}" "{vllm_url}" "{n_models}" {conda_arg}'
        )
        run_cmd(orch_job_cmd, dry_run=args.dry_run)

        if not args.dry_run:
            print(f"\nGenerator job ID: {gen_job_id}")
        else:
            print("\n[Dry run] No jobs were submitted.")
    else:
        print("[Local mode] Please run each command in a separate terminal or use a process manager.")
        print("\nExample (tmux):")
        print(f"  tmux new-session -d -s gen '{gen_cmd}'")
        for i in range(n_models):
            eval_cmd = f'{base_cmd} --client-id {i} {vllm_url_arg}'
            print(f"  tmux new-window -t gen -n eval{i} '{eval_cmd}'")
        print(f"  tmux new-window -t gen -n judge '{orch_cmd}'")


if __name__ == "__main__":
    main()
