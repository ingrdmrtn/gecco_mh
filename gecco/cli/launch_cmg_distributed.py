"""CLI route for CMG distributed launch."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel

from config.schema import load_config
from gecco.cli.launcher_utils import (
    LaunchCommand,
    LaunchExecutor,
    LaunchPlan,
    SubmissionResult,
)


console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _print_final_eval_result(result: SubmissionResult) -> None:
    if result.job_id:
        print(f"  Final evaluation job ID: {result.job_id}")
    print()


def register_parser(subparsers) -> argparse.ArgumentParser:
    """Register the CMG launcher subcommand."""
    parser = subparsers.add_parser(
        "cmg-distributed", help="Launch a CMG distributed GeCCo run"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--vllm-url", type=str, default=None)
    parser.add_argument("--conda-env", type=str, default=None)
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--cpus-per-task", type=int, default=None)
    parser.add_argument("--mem", type=str, default=None)
    parser.add_argument(
        "--run-final-eval",
        dest="run_final_eval",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.set_defaults(handler=main)
    return parser


def _get_slurm_defaults(config_path):
    """Read SLURM defaults from config."""
    with config_path.open("r", encoding="utf-8") as file_obj:
        raw = yaml.safe_load(file_obj)
    slurm = raw.get("slurm", {})
    return {
        "cpus_per_task": slurm.get("cpus_per_task"),
        "partition": slurm.get("partition"),
        "mem_per_task": slurm.get("mem_per_task"),
    }


def _get_results_dir(project_root, cfg):
    task_name = getattr(cfg.task, "name", "unknown")
    fit_type = getattr(cfg.evaluation, "fit_type", "group")
    if fit_type == "individual":
        return project_root / "results" / f"{task_name}_individual"
    return project_root / "results" / task_name
def run_cmg_distributed_launcher(
    *,
    config: str,
    dry_run: bool = False,
    local: bool = False,
    vllm_url: str | None = None,
    conda_env: str | None = None,
    partition: str | None = None,
    cpus_per_task: int | None = None,
    mem: str | None = None,
    run_final_eval: bool | None = None,
) -> int | None:
    """Launch a centralised model generation distributed GeCCo search."""
    config_path = PROJECT_ROOT / "config" / config

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        raise SystemExit(1)

    cfg = load_config(config_path)
    cmg_cfg = getattr(cfg, "centralized_model_generation", None)

    if cmg_cfg is None or not getattr(cmg_cfg, "enabled", False):
        print(
            "ERROR: CMG not enabled in config (centralized_model_generation.enabled: true)"
        )
        raise SystemExit(1)

    generator_client = str(getattr(cmg_cfg, "generator_client", ""))
    if not generator_client:
        print("ERROR: centralized_model_generation.generator_client is required")
        raise SystemExit(1)
    if generator_client.isdigit() or generator_client.lstrip("-").isdigit():
        print(
            "ERROR: centralized_model_generation.generator_client must be a named "
            "profile, not a numeric evaluator ID"
        )
        raise SystemExit(1)

    n_models = getattr(cmg_cfg, "n_models", None)
    if not isinstance(n_models, int) or n_models <= 0:
        print("ERROR: centralized_model_generation.n_models must be a positive integer")
        raise SystemExit(1)

    final_eval_enabled = getattr(cmg_cfg, "run_final_evaluation", True)
    if run_final_eval is not None:
        final_eval_enabled = run_final_eval

    conda_arg = f'"{conda_env}"' if conda_env else '""'
    executor = LaunchExecutor()

    slurm_defaults = _get_slurm_defaults(config_path)
    resolved_cpus_per_task = cpus_per_task or slurm_defaults.get("cpus_per_task") or 48
    resolved_partition = partition or slurm_defaults.get("partition")
    partition_flag = f"--partition={resolved_partition}" if resolved_partition else ""
    resolved_mem = mem or slurm_defaults.get("mem_per_task")
    mem_flag = f"--mem={resolved_mem}" if resolved_mem else ""

    task_name = getattr(cfg.task, "name", "unknown")
    results_dir = _get_results_dir(PROJECT_ROOT, cfg)
    results_dir_rel = results_dir.relative_to(PROJECT_ROOT)

    resolved_vllm_url = vllm_url or os.environ.get("VLLM_BASE_URL", "")
    vllm_url_arg = f'--vllm-url "{resolved_vllm_url}"' if resolved_vllm_url else ""

    console.print(
        Panel(
            f"[bold]Config:[/] {config}\n"
            f"[bold]Task:[/] {task_name}\n"
            f"[bold]Generator Client:[/] {generator_client}\n"
            f"[bold]Evaluators:[/] {n_models} (IDs 0..{n_models - 1})\n"
            f"[bold]Final Eval:[/] {'enabled' if final_eval_enabled else 'disabled'}\n"
            f"[bold]vLLM URL:[/] {resolved_vllm_url or '(not set)'}\n"
            f"[bold]Mode:[/] {'local process' if local else 'SLURM'}",
            title="CMG Distributed Launch Plan",
            style="green",
        )
    )

    base_cmd = f"python -m gecco internal distributed-client --config {config}"

    print()
    print("=" * 60)
    print("LAUNCH PLAN")
    print("=" * 60)

    gen_cmd = f"{base_cmd} --client-profile {generator_client} {vllm_url_arg}"
    print(f"\n[Generator] {generator_client}:")
    print(f"  {gen_cmd}")

    print(f"\n[Evaluators] {n_models} clients:")
    for index in range(n_models):
        eval_cmd = f"{base_cmd} --client-id {index} {vllm_url_arg}"
        print(f"  Evaluator {index}: {eval_cmd}")

    orch_cmd = f"python -m gecco judge orchestrate --config {config} {vllm_url_arg}"
    print("\n[Judge Orchestrator]:")
    print(f"  {orch_cmd}")

    if final_eval_enabled:
        final_eval_cmd = (
            f"sbatch --job-name=gecco-cmg-test-eval --cpus-per-task=8 "
            f"{partition_flag} --mem=16G "
            f"--output=logs/gecco-cmg-test-eval-%j.out "
            f"--error=logs/gecco-cmg-test-eval-%j.err "
            f'{PROJECT_ROOT / "bash/run_test_evaluation.sh"} '
            f'"{config}" "{results_dir_rel}" {conda_arg}'
        )
        print("\n[Final Test Evaluation]:")
        print(f"  {final_eval_cmd}")

    print()

    if dry_run and local:
        print("[Dry run] Commands printed above — not executing.")
        return None

    if not local:
        print("Submitting generator job...")
        final_eval_dep_fallback = "--dependency=afterok:<generator_job_id>:<evaluator_job_id>:<orchestrator_job_id>"
        plan_commands = [
            LaunchCommand(
                label="generator",
                command=(
                    f"sbatch "
                    f"--job-name=gecco-cmg-generator "
                    f"--cpus-per-task={resolved_cpus_per_task} "
                    f"{partition_flag} "
                    f"{mem_flag} "
                    f"--output=logs/gecco-cmg-generator-%j.out "
                    f"--error=logs/gecco-cmg-generator-%j.err "
                    f'{PROJECT_ROOT / "bash/run_cmg_generator.sh"} '
                    f'"{config}" "{generator_client}" "{resolved_vllm_url}" {conda_arg}'
                ),
            ),
            LaunchCommand(
                label="evaluator",
                command=(
                    f"sbatch "
                    f"--array=0-{n_models - 1} "
                    f"--job-name=gecco-cmg-evaluator "
                    f"--cpus-per-task={resolved_cpus_per_task} "
                    f"{partition_flag} "
                    f"{mem_flag} "
                    f"--output=logs/gecco-cmg-evaluator-%A_%a.out "
                    f"--error=logs/gecco-cmg-evaluator-%A_%a.err "
                    f'{PROJECT_ROOT / "bash/run_gecco_distributed.sh"} '
                    f'"{config}" "" "{resolved_vllm_url}" {conda_arg}'
                ),
            ),
            LaunchCommand(
                label="orchestrator",
                command=(
                    f"sbatch "
                    f"--job-name=gecco-cmg-orchestrator "
                    f"--cpus-per-task=8 "
                    f"{partition_flag} "
                    f"--mem=16G "
                    f"--output=logs/gecco-cmg-orchestrator-%j.out "
                    f"--error=logs/gecco-cmg-orchestrator-%j.err "
                    f'{PROJECT_ROOT / "bash/run_judge_orchestrator.sh"} '
                    f'"{config}" "{resolved_vllm_url}" "{n_models}" {conda_arg}'
                ),
            ),
        ]
        if final_eval_enabled:
            plan_commands.append(
                LaunchCommand(
                    label="final_eval",
                    command=(
                        f"sbatch {{dependency}} --cpus-per-task=8 {partition_flag} --mem=16G "
                        f'{PROJECT_ROOT / "bash/run_test_evaluation.sh"} '
                        f'"{config}" "{results_dir_rel}" {conda_arg}'
                    ),
                    dependency_labels=("generator", "evaluator", "orchestrator"),
                    dependency_fallback=final_eval_dep_fallback,
                )
            )
        submission_results = executor.execute(
            LaunchPlan(commands=tuple(plan_commands)),
            dry_run=dry_run,
            on_result=_print_final_eval_result,
        )
        gen_result = submission_results[0]
        gen_job_id = gen_result.job_id
        if not dry_run:
            print(f"\nGenerator job ID: {gen_job_id}")
        else:
            print("\n[Dry run] No jobs were submitted.")
    else:
        print(
            "[Local mode] Please run each command in a separate terminal or use a process manager."
        )
        print("\nExample (tmux):")
        print(f"  tmux new-session -d -s gen '{gen_cmd}'")
        for index in range(n_models):
            eval_cmd = f"{base_cmd} --client-id {index} {vllm_url_arg}"
            print(f"  tmux new-window -t gen -n eval{index} '{eval_cmd}'")
        print(f"  tmux new-window -t gen -n judge '{orch_cmd}'")
        if final_eval_enabled:
            final_eval_cmd = (
                f"python -m gecco internal test-evaluation "
                f"--config {config} --results-dir {results_dir_rel} --write-store"
            )
            print("  # Run after generator/evaluators/orchestrator finish:")
            print(f"  {final_eval_cmd}")
    return None


def main(args: argparse.Namespace) -> int | None:
    """Run the CMG launcher from parsed CLI arguments."""
    return run_cmg_distributed_launcher(
        config=args.config,
        dry_run=args.dry_run,
        local=args.local,
        vllm_url=args.vllm_url,
        conda_env=args.conda_env,
        partition=args.partition,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        run_final_eval=args.run_final_eval,
    )
