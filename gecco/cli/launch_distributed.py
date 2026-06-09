"""CLI route for distributed launch."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config.schema import load_config
from gecco.load_llms.provider_registry import get_provider_spec
from gecco.sentry_init import init_sentry


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def register_parser(subparsers) -> argparse.ArgumentParser:
    """Register the distributed launcher subcommand."""
    parser = subparsers.add_parser("distributed", help="Launch a distributed GeCCo run")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--profiles", type=str, default=None)
    parser.add_argument("--extra-clients", type=int, default=0)
    parser.add_argument("--launch-vllm", action="store_true")
    parser.add_argument("--vllm-model", type=str, default=None)
    parser.add_argument("--vllm-tp", type=int, default=1)
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-url", type=str, default=None)
    parser.add_argument("--conda-env", type=str, default=None)
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--cpus-per-task", type=int, default=None)
    parser.add_argument("--mem", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--launch-orchestrator", action="store_true")
    parser.set_defaults(handler=main)
    return parser


def get_profiles_from_config(config_path):
    """Read client profile names from a YAML config file."""
    cfg = load_config(config_path)
    clients = getattr(cfg, "clients", {}) or {}
    if not clients:
        return []
    return list(clients.keys())


def run_cmd(cmd, dry_run=False):
    """Run a shell command, or just print it if dry_run."""
    print(f"  $ {cmd}")
    if dry_run:
        return None
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        raise SystemExit(1)
    output = result.stdout.strip()
    if output.startswith("Submitted batch job "):
        return output.split()[-1]
    return output


def run_distributed_launcher(
    *,
    config: str,
    profiles: str | None = None,
    extra_clients: int = 0,
    launch_vllm: bool = False,
    vllm_model: str | None = None,
    vllm_tp: int = 1,
    vllm_port: int = 8000,
    vllm_url: str | None = None,
    conda_env: str | None = None,
    partition: str | None = None,
    cpus_per_task: int | None = None,
    mem: str | None = None,
    dry_run: bool = False,
    launch_orchestrator: bool = False,
) -> int | None:
    """Launch a distributed GeCCo search from a config file."""
    config_path = PROJECT_ROOT / "config" / config

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        raise SystemExit(1)

    cfg = load_config(config_path)
    resolved_profiles = profiles.split(",") if profiles else list((cfg.clients or {}).keys())
    n_profiled = len(resolved_profiles)
    n_total = n_profiled + extra_clients
    if n_total == 0:
        print(
            "ERROR: No profiles found in config and --extra-clients is 0. Nothing to launch."
        )
        raise SystemExit(1)

    all_profiles = resolved_profiles + [""] * extra_clients
    profiles_csv = ",".join(all_profiles)
    array_spec = f"0-{n_total - 1}"

    init_sentry(
        task_name=cfg.task.name,
        config_name=config,
    )

    cmg_cfg = getattr(cfg, "centralized_model_generation", None)
    if cmg_cfg and getattr(cmg_cfg, "enabled", False):
        print(
            "ERROR: This config has centralized_model_generation.enabled: true.\n"
            "       Use the CMG launcher instead:\n"
            "       python -m gecco run cmg-distributed "
            f"--config {config}"
        )
        raise SystemExit(1)

    provider = cfg.llm.provider
    try:
        provider_spec = get_provider_spec(provider)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc
    slurm_cfg = getattr(cfg, "slurm", {}) or {}
    resolved_launch_orchestrator = launch_orchestrator or getattr(cfg, "judge", None) is not None
    n_clients = getattr(cfg.loop, "n_clients", None)

    resolved_cpus_per_task = cpus_per_task or slurm_cfg.get("cpus_per_task", 48)
    resolved_mem = mem or slurm_cfg.get("mem_per_task")
    mem_flag = f"--mem={resolved_mem}" if resolved_mem else ""

    resolved_partition = partition or slurm_cfg.get("partition")
    partition_flag = f"--partition={resolved_partition}" if resolved_partition else ""

    print(f"Config:            {config}")
    print(f"Provider:          {provider_spec.label} ({provider_spec.key})")
    print(f"CPUs/task:         {resolved_cpus_per_task}")
    if resolved_mem:
        print(f"Memory:            {resolved_mem}")
    print(f"Profiles:          {resolved_profiles if resolved_profiles else '(none)'}")
    print(f"Extra clients:     {extra_clients}")
    print(f"Total clients:     {n_total}")
    print(f"Array spec:        --array={array_spec}")
    print(f"Profiles CSV:      {profiles_csv}")
    if resolved_partition:
        print(f"Partition:         {resolved_partition}")
    if provider_spec.key == "vllm":
        print(f"vLLM URL:          {vllm_url or '(from env / .vllm_env)'}")
    if resolved_launch_orchestrator:
        print("[Orchestrator]     ENABLED (centralized judge)")
        if n_clients:
            print(f"[Orchestrator]     n_clients: {n_clients}")
    print()

    resolved_vllm_model = vllm_model
    if launch_vllm and not resolved_vllm_model:
        resolved_vllm_model = cfg.llm.base_model or "Qwen/Qwen2.5-14B-Instruct"

    vllm_job_id = None
    if launch_vllm:
        print("Launching vLLM server...")
        cmd = (
            f"sbatch --parsable {partition_flag} bash/launch_vllm_server.sh "
            f'"{resolved_vllm_model}" {vllm_port} {vllm_tp}'
        )
        vllm_job_id = run_cmd(cmd, dry_run=dry_run)
        if vllm_job_id:
            print(f"  vLLM job ID: {vllm_job_id}")
        print()

    print("Launching client array...")
    dep_flag = f"--dependency=afterok:{vllm_job_id}" if vllm_job_id else ""
    vllm_url_arg = f'"{vllm_url}"' if vllm_url else '""'
    conda_arg = f'"{conda_env}"' if conda_env else '""'
    cmd = (
        f"sbatch --array={array_spec} --cpus-per-task={resolved_cpus_per_task} "
        f"{dep_flag} {partition_flag} {mem_flag} "
        f'bash/run_gecco_distributed.sh "{config}" "{profiles_csv}" {vllm_url_arg} {conda_arg}'
    )
    client_job_id = run_cmd(cmd, dry_run=dry_run)
    if client_job_id:
        print(f"  Client array job ID: {client_job_id}")
    print()

    orchestrator_job_id = None
    if resolved_launch_orchestrator:
        print("Launching centralized judge orchestrator...")
        orch_dep_flag = f"--dependency=afterok:{vllm_job_id}" if vllm_job_id else ""
        vllm_url_arg_orch = f'"{vllm_url}"' if vllm_url else '""'
        n_clients_arg = f'"{n_clients}"' if n_clients else '""'
        conda_arg = f'"{conda_env}"' if conda_env else '""'
        cmd = (
            f"sbatch {orch_dep_flag} --cpus-per-task=8 {partition_flag} --mem=16G "
            f'bash/run_judge_orchestrator.sh "{config}" {vllm_url_arg_orch} {n_clients_arg} {conda_arg}'
        )
        orchestrator_job_id = run_cmd(cmd, dry_run=dry_run)
        if orchestrator_job_id:
            print(f"  Orchestrator job ID: {orchestrator_job_id}")
        print()

    test_eval_job_id = None
    if client_job_id:
        print("Scheduling test evaluation (post-processing)...")
        test_dep_flag = f"--dependency=afterok:{client_job_id}"
        task_name = cfg.task.name
        fit_type = cfg.evaluation.fit_type
        results_dir = f"results/{task_name}"
        if fit_type == "individual":
            results_dir = f"results/{task_name}_individual"
        conda_arg = f'"{conda_env}"' if conda_env else '""'
        cmd = (
            f"sbatch {test_dep_flag} --cpus-per-task=8 {partition_flag} --mem=16G "
            f'bash/run_test_evaluation.sh "{config}" "{results_dir}" {conda_arg}'
        )
        test_eval_job_id = run_cmd(cmd, dry_run=dry_run)
        if test_eval_job_id:
            print(f"  Test evaluation job ID: {test_eval_job_id}")
        print()

    print("Launched successfully. Monitor with:")
    task_name = cfg.task.name
    print(f"  python -m gecco monitor --task {task_name} --watch 10")
    if test_eval_job_id:
        print(
            f"Test evaluation will run after all clients complete: job {test_eval_job_id}"
        )
    return None


def main(args: argparse.Namespace) -> int | None:
    """Run the distributed launcher from parsed CLI arguments."""
    return run_distributed_launcher(
        config=args.config,
        profiles=args.profiles,
        extra_clients=args.extra_clients,
        launch_vllm=args.launch_vllm,
        vllm_model=args.vllm_model,
        vllm_tp=args.vllm_tp,
        vllm_port=args.vllm_port,
        vllm_url=args.vllm_url,
        conda_env=args.conda_env,
        partition=args.partition,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        dry_run=args.dry_run,
        launch_orchestrator=args.launch_orchestrator,
    )
