#!/usr/bin/env python
"""
Launch a distributed GeCCo search from a config file.

Reads client profiles from the config's `clients:` section, launches the
vLLM server (if requested), and submits a SLURM job array with one task
per profile.

Usage:
    # Launch vLLM + clients (full pipeline)
    python scripts/launch_distributed.py --config two_step_factors_distributed.yaml --launch-vllm

    # Clients only (vLLM already running)
    python scripts/launch_distributed.py --config two_step_factors_distributed.yaml

    # Subset of profiles
    python scripts/launch_distributed.py --config two_step_factors_distributed.yaml --profiles exploit,explore

    # Dry run (show commands without submitting)
    python scripts/launch_distributed.py --config two_step_factors_distributed.yaml --dry-run

    # Extra clients with no profile (base config only)
    python scripts/launch_distributed.py --config two_step_factors_distributed.yaml --extra-clients 2
"""

import argparse
import os
import subprocess
import sys
import yaml
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gecco.sentry_init import init_sentry


def get_profiles_from_config(config_path):
    """Read client profile names from a YAML config file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    clients = cfg.get("clients", {})
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
        sys.exit(1)
    output = result.stdout.strip()
    # sbatch outputs "Submitted batch job JOBID" — extract just the numeric ID
    if output.startswith("Submitted batch job "):
        return output.split()[-1]
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Launch distributed GeCCo search",
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
        "--profiles",
        type=str,
        default=None,
        help="Comma-separated list of profiles to use (default: all from config)",
    )
    parser.add_argument(
        "--extra-clients",
        type=int,
        default=0,
        help="Additional clients with no profile override (base config only)",
    )
    parser.add_argument(
        "--launch-vllm", action="store_true", help="Also launch the vLLM server job"
    )
    parser.add_argument(
        "--vllm-model",
        type=str,
        default=None,
        help="Model for vLLM server (default: read from config)",
    )
    parser.add_argument(
        "--vllm-tp",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (default: 1)",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=8000,
        help="Port for vLLM server (default: 8000)",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=None,
        help="vLLM server URL (e.g. http://gpu-node:8000/v1). "
        "Passed to all clients. If omitted, clients read $VLLM_BASE_URL or $HOME/.vllm_env",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default=None,
        help="Conda environment to activate in each client job",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default=None,
        help="SLURM partition to submit jobs to (e.g. gpu, cpu, batch)",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=None,
        help="CPUs per client node (overrides config slurm.cpus_per_task, default: 48)",
    )
    parser.add_argument(
        "--mem",
        type=str,
        default=None,
        help="Memory per node, e.g. 64G (overrides config slurm.mem_per_task)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without submitting"
    )
    parser.add_argument(
        "--launch-orchestrator",
        action="store_true",
        help="Also launch the centralized judge orchestrator (auto-detected from config if true)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config" / args.config

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    # --- Resolve profiles ---
    if args.profiles:
        profiles = args.profiles.split(",")
    else:
        profiles = get_profiles_from_config(config_path)

    n_profiled = len(profiles)
    n_total = n_profiled + args.extra_clients

    if n_total == 0:
        print(
            "ERROR: No profiles found in config and --extra-clients is 0. Nothing to launch."
        )
        sys.exit(1)

    # Pad profile list with empty strings for extra clients
    all_profiles = profiles + [""] * args.extra_clients
    profiles_csv = ",".join(all_profiles)
    array_spec = f"0-{n_total - 1}"

    # Detect provider and SLURM settings from config
    with open(config_path, "r") as f:
        cfg_raw = yaml.safe_load(f)

    init_sentry(
        task_name=cfg_raw.get("task", {}).get("name"),
        config_name=args.config,
    )

    # Guard against CMG configs — they require launch_cmg_distributed.py
    cmg_cfg = cfg_raw.get("centralized_model_generation", {})
    if cmg_cfg and cmg_cfg.get("enabled", False):
        print(
            "ERROR: This config has centralized_model_generation.enabled: true.\n"
            "       Use the CMG launcher instead:\n"
            "       python scripts/launch_cmg_distributed.py "
            f"--config {args.config} --slurm"
        )
        sys.exit(1)

    provider = cfg_raw.get("llm", {}).get("provider", "vllm")
    slurm_cfg = cfg_raw.get("slurm", {})

    # Auto-detect if orchestrator is needed from config
    judge_cfg = cfg_raw.get("judge", {})
    orchestrated = judge_cfg.get("orchestrated", False)
    launch_orchestrator = args.launch_orchestrator or orchestrated

    # Get loop config for n_clients
    loop_cfg = cfg_raw.get("loop", {})
    n_clients = loop_cfg.get("n_clients")

    # Resolve cpus-per-task: CLI > config > default
    cpus_per_task = args.cpus_per_task or slurm_cfg.get("cpus_per_task", 48)

    # Resolve memory: CLI > config
    mem = args.mem or slurm_cfg.get("mem_per_task")
    mem_flag = f"--mem={mem}" if mem else ""

    # Resolve partition: CLI > config
    partition = args.partition or slurm_cfg.get("partition")
    partition_flag = f"--partition={partition}" if partition else ""

    print(f"Config:            {args.config}")
    print(f"Provider:          {provider}")
    print(f"CPUs/task:         {cpus_per_task}")
    if mem:
        print(f"Memory:            {mem}")
    print(f"Profiles:          {profiles if profiles else '(none)'}")
    print(f"Extra clients:     {args.extra_clients}")
    print(f"Total clients:     {n_total}")
    print(f"Array spec:        --array={array_spec}")
    print(f"Profiles CSV:      {profiles_csv}")
    if partition:
        print(f"Partition:         {partition}")
    if provider == "vllm":
        print(f"vLLM URL:          {args.vllm_url or '(from env / .vllm_env)'}")
    if launch_orchestrator:
        print(f"[Orchestrator]     ENABLED (centralized judge)")
        if n_clients:
            print(f"[Orchestrator]     n_clients: {n_clients}")
    print()

    # --- Resolve vLLM model from config if needed ---
    vllm_model = args.vllm_model
    if args.launch_vllm and not vllm_model:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        vllm_model = cfg.get("llm", {}).get("base_model", "Qwen/Qwen2.5-14B-Instruct")

    # --- Launch vLLM server ---
    vllm_job_id = None
    if args.launch_vllm:
        print("Launching vLLM server...")
        cmd = (
            f"sbatch --parsable {partition_flag} bash/launch_vllm_server.sh "
            f'"{vllm_model}" {args.vllm_port} {args.vllm_tp}'
        )
        vllm_job_id = run_cmd(cmd, dry_run=args.dry_run)
        if vllm_job_id:
            print(f"  vLLM job ID: {vllm_job_id}")
        print()

    # --- Launch client array ---
    print("Launching client array...")
    dep_flag = f"--dependency=afterok:{vllm_job_id}" if vllm_job_id else ""
    vllm_url_arg = f'"{args.vllm_url}"' if args.vllm_url else '""'
    conda_arg = f'"{args.conda_env}"' if args.conda_env else '""'
    cmd = (
        f"sbatch --array={array_spec} --cpus-per-task={cpus_per_task} {dep_flag} {partition_flag} {mem_flag} "
        f'bash/run_gecco_distributed.sh "{args.config}" "{profiles_csv}" {vllm_url_arg} {conda_arg}'
    )
    client_job_id = run_cmd(cmd, dry_run=args.dry_run)
    if client_job_id:
        print(f"  Client array job ID: {client_job_id}")
    print()

    # --- Launch orchestrator (if enabled) ---
    orchestrator_job_id = None
    if launch_orchestrator:
        print("Launching centralized judge orchestrator...")
        # Orchestrator should start at the same time as clients
        # If vLLM is being launched, orchestrator also depends on it
        orch_dep_flag = f"--dependency=afterok:{vllm_job_id}" if vllm_job_id else ""
        vllm_url_arg_orch = f'"{args.vllm_url}"' if args.vllm_url else '""'
        n_clients_arg = f'"{n_clients}"' if n_clients else '""'
        conda_arg = f'"{args.conda_env}"' if args.conda_env else '""'
        cmd = (
            f"sbatch {orch_dep_flag} --cpus-per-task=8 {partition_flag} --mem=16G "
            f'bash/run_judge_orchestrator.sh "{args.config}" {vllm_url_arg_orch} {n_clients_arg} {conda_arg}'
        )
        orchestrator_job_id = run_cmd(cmd, dry_run=args.dry_run)
        if orchestrator_job_id:
            print(f"  Orchestrator job ID: {orchestrator_job_id}")
        print()

    # --- Launch test evaluation (post-processing) ---
    test_eval_job_id = None
    if client_job_id:
        print("Scheduling test evaluation (post-processing)...")
        test_dep_flag = f"--dependency=afterok:{client_job_id}"
        # Resolve results directory from config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        task_name = cfg.get("task", {}).get("name", "unknown")
        fit_type = cfg.get("evaluation", {}).get("fit_type", "group")
        results_dir = f"results/{task_name}"
        if fit_type == "individual":
            results_dir = f"results/{task_name}_individual"
        conda_arg = f'"{args.conda_env}"' if args.conda_env else '""'
        cmd = (
            f"sbatch {test_dep_flag} --cpus-per-task=8 {partition_flag} --mem=16G "
            f'bash/run_test_evaluation.sh "{args.config}" "{results_dir}" {conda_arg}'
        )
        test_eval_job_id = run_cmd(cmd, dry_run=args.dry_run)
        if test_eval_job_id:
            print(f"  Test evaluation job ID: {test_eval_job_id}")
        print()

    # --- Summary ---
    print("Launched successfully. Monitor with:")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    task_name = cfg.get("task", {}).get("name", "unknown")
    print(f"  python scripts/monitor_distributed.py --task {task_name} --watch 10")
    if test_eval_job_id:
        print(
            f"Test evaluation will run after all clients complete: job {test_eval_job_id}"
        )


if __name__ == "__main__":
    main()
