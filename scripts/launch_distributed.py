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
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Launch distributed GeCCo search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Config YAML filename (relative to config/)")
    parser.add_argument("--profiles", type=str, default=None,
                        help="Comma-separated list of profiles to use (default: all from config)")
    parser.add_argument("--extra-clients", type=int, default=0,
                        help="Additional clients with no profile override (base config only)")
    parser.add_argument("--launch-vllm", action="store_true",
                        help="Also launch the vLLM server job")
    parser.add_argument("--vllm-model", type=str, default=None,
                        help="Model for vLLM server (default: read from config)")
    parser.add_argument("--vllm-tp", type=int, default=1,
                        help="Tensor parallel size for vLLM (default: 1)")
    parser.add_argument("--vllm-port", type=int, default=8000,
                        help="Port for vLLM server (default: 8000)")
    parser.add_argument("--vllm-url", type=str, default=None,
                        help="vLLM server URL (e.g. http://gpu-node:8000/v1). "
                             "Passed to all clients. If omitted, clients read $VLLM_BASE_URL or $HOME/.vllm_env")
    parser.add_argument("--conda-env", type=str, default=None,
                        help="Conda environment to activate in each client job")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without submitting")
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
        print("ERROR: No profiles found in config and --extra-clients is 0. Nothing to launch.")
        sys.exit(1)

    # Pad profile list with empty strings for extra clients
    all_profiles = profiles + [""] * args.extra_clients
    profiles_csv = ",".join(all_profiles)
    array_spec = f"0-{n_total - 1}"

    # Detect provider from config
    with open(config_path, "r") as f:
        cfg_raw = yaml.safe_load(f)
    provider = cfg_raw.get("llm", {}).get("provider", "vllm")

    print(f"Config:       {args.config}")
    print(f"Provider:     {provider}")
    print(f"Profiles:     {profiles if profiles else '(none)'}")
    print(f"Extra clients: {args.extra_clients}")
    print(f"Total clients: {n_total}")
    print(f"Array spec:   --array={array_spec}")
    print(f"Profiles CSV: {profiles_csv}")
    if provider == "vllm":
        print(f"vLLM URL:     {args.vllm_url or '(from env / .vllm_env)'}")
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
            f"sbatch --parsable bash/launch_vllm_server.sh "
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
        f"sbatch --array={array_spec} {dep_flag} "
        f'bash/run_gecco_distributed.sh "{args.config}" "{profiles_csv}" {vllm_url_arg} {conda_arg}'
    )
    client_job_id = run_cmd(cmd, dry_run=args.dry_run)
    if client_job_id:
        print(f"  Client array job ID: {client_job_id}")
    print()

    # --- Summary ---
    print("Launched successfully. Monitor with:")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    task_name = cfg.get("task", {}).get("name", "unknown")
    print(f"  python scripts/monitor_distributed.py --task {task_name} --watch 10")


if __name__ == "__main__":
    main()
