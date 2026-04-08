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
import sys
from pathlib import Path


def main():
    # Force unbuffered output so results appear immediately
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)

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
        "--print-cmd",
        action="store_true",
        help="Print the exact command to run and exit (for manual execution)",
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

    # --- Build command line args for distributed script ---
    # Build the argument list that would be passed to the distributed script
    distributed_argv = [
        "run_gecco_distributed.py",  # Program name
        "--config",
        args.config,
        "--client-id",
        str(args.client_id),
    ]

    if args.profile:
        distributed_argv.extend(["--client-profile", args.profile])

    if args.vllm_url:
        distributed_argv.extend(["--vllm-url", args.vllm_url])

    if args.test:
        distributed_argv.append("--test")

    # --- Build the shell command ---
    # Construct the full command that can be copy-pasted
    # Use python -u for unbuffered output so you see results immediately
    cmd_parts = ["python -u", "scripts/run_gecco_distributed.py"]
    cmd_parts.append(f"--config {args.config}")
    cmd_parts.append(f"--client-id {args.client_id}")

    if args.profile:
        cmd_parts.append(f"--client-profile {args.profile}")

    if args.vllm_url:
        cmd_parts.append(f'--vllm-url "{args.vllm_url}"')

    if args.test:
        cmd_parts.append("--test")

    full_cmd = " ".join(cmd_parts)

    # --- Print the command ---
    print("=" * 60)
    print("COMMAND TO RUN INDEPENDENTLY:")
    print("=" * 60)
    print(full_cmd)
    print("=" * 60)
    print("TIP: The -u flag forces unbuffered output so you see results immediately")
    print()
    print("ALTERNATIVE (with environment variable):")
    print(f"  PYTHONUNBUFFERED=1 {full_cmd.replace('python -u', 'python')}")
    print()

    if args.print_cmd:
        # Just print the command and exit
        return

    if args.dry_run:
        print("(Dry run - command not executed)")
        return

    # Change to project root
    os.chdir(project_root)

    # Check if we're in the right conda environment
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if current_env != args.conda_env:
        print(
            f"WARNING: Current conda environment is '{current_env}', expected '{args.conda_env}'"
        )
        print(f"         Activate with: conda activate {args.conda_env}")
        print()

    # --- Run distributed script directly in same process ---
    # This ensures all output is visible immediately
    original_argv = sys.argv
    try:
        # Import the distributed script's main function
        # Add scripts directory to path if needed
        sys.path.insert(0, str(project_root))

        # Import and run the distributed script
        from scripts.run_gecco_distributed import main as distributed_main

        # Set up argv as if we called the script directly
        sys.argv = distributed_argv

        # Run it
        distributed_main()

    except SystemExit as e:
        # Handle sys.exit() calls from the distributed script
        if e.code != 0 and e.code is not None:
            print(f"\nERROR: Client exited with code {e.code}")
            sys.exit(e.code)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore original argv
        sys.argv = original_argv

    print("\nClient completed successfully!")


if __name__ == "__main__":
    main()
