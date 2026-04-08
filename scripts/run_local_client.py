#!/usr/bin/env python
"""
Run a GeCCo client locally for testing and debugging.

This script is a lightweight wrapper around run_gecco_distributed.py that
simplifies local testing without requiring SLURM or a full distributed setup.

Usage:
    # Run with test mode (uses small local model, 1 run, 1 iteration)
    python scripts/run_local_client.py --config two_step_factors_distributed.yaml --test

    # Run a specific client profile
    python scripts/run_local_client.py --config two_step_factors_distributed.yaml --profile exploit

    # Run with a specific client ID
    python scripts/run_local_client.py --config two_step_factors_distributed.yaml --client-id 0

    # Dry run - just show what would be executed
    python scripts/run_local_client.py --config two_step_factors_distributed.yaml --dry-run

    # Run with custom vLLM URL
    python scripts/run_local_client.py --config two_step_factors_distributed.yaml --vllm-url http://localhost:8000/v1

    # Full example with all options
    python scripts/run_local_client.py \
        --config two_step_factors_distributed.yaml \
        --profile exploit \
        --client-id 0 \
        --vllm-url http://localhost:8000/v1 \
        --test
"""

import argparse
import os
import subprocess
import sys
import yaml
from pathlib import Path


def run_cmd(cmd, dry_run=False):
    """Run a shell command, or just print it if dry_run."""
    print(f"  $ {cmd}")
    if dry_run:
        return None
    result = subprocess.run(cmd, shell=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run a GeCCo client locally for testing/debugging",
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
        "--profile",
        type=str,
        default=None,
        help="Client profile to use (from config clients: section)",
    )
    parser.add_argument(
        "--client-id", type=int, default=0, help="Client ID (default: 0)"
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=None,
        help="vLLM server URL (e.g. http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: use small local model with 1 run and 1 iteration",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print command without executing"
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="gecco_mh",
        help="Conda environment to use (default: gecco_mh)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config" / args.config

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    # --- Display configuration ---
    print("=" * 60)
    print("GeCCo Local Client Runner")
    print("=" * 60)
    print(f"Config:     {args.config}")
    print(f"Profile:    {args.profile or 'default'}")
    print(f"Client ID:  {args.client_id}")
    print(f"Test mode:  {'Yes' if args.test else 'No'}")
    if args.vllm_url:
        print(f"vLLM URL:   {args.vllm_url}")
    print()

    # --- Build command ---
    cmd_parts = [
        "python",
        "scripts/run_gecco_distributed.py",
        f'--config "{args.config}"',
        f"--client-id {args.client_id}",
    ]

    if args.profile:
        cmd_parts.append(f'--client-profile "{args.profile}"')

    if args.vllm_url:
        cmd_parts.append(f'--vllm-url "{args.vllm_url}"')

    if args.test:
        cmd_parts.append("--test")

    cmd = " ".join(cmd_parts)

    # --- Execute ---
    print("Executing:")
    print(f"  $ {cmd}")
    print()

    if args.dry_run:
        print("(Dry run - command not executed)")
        return

    # Change to project root and run
    os.chdir(project_root)

    # Check if we're in the right conda environment
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if current_env != args.conda_env:
        print(
            f"WARNING: Current conda environment is '{current_env}', expected '{args.conda_env}'"
        )
        print(f"         Activate with: conda activate {args.conda_env}")
        print()

    exit_code = run_cmd(cmd, dry_run=args.dry_run)

    if exit_code != 0:
        print(f"\nERROR: Client exited with code {exit_code}")
        sys.exit(exit_code)

    print("\nClient completed successfully!")


if __name__ == "__main__":
    main()
