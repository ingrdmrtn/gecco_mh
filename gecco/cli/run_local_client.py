"""CLI route for local client execution."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .run_gecco_distributed import run_distributed_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def register_parser(subparsers) -> argparse.ArgumentParser:
    """Register the local-client command."""
    parser = subparsers.add_parser("local-client", help="Run a local GeCCo client")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--vllm-url", type=str, default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-cmd", action="store_true")
    parser.add_argument("--conda-env", type=str, default=None)
    parser.set_defaults(handler=main)
    return parser


def run_local_client(
    *,
    config: str,
    profile: str | None = None,
    client_id: int = 0,
    vllm_url: str | None = None,
    test: bool = False,
    dry_run: bool = False,
    print_cmd: bool = False,
    conda_env: str | None = None,
) -> int | None:
    """Run a GeCCo client locally for testing and debugging."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)

    config_path = PROJECT_ROOT / "config" / config
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        raise SystemExit(1)

    print("=" * 60)
    print("GeCCo Local Client Runner")
    print("=" * 60)
    print(f"Config:     {config}")
    print(f"Profile:    {profile or 'default'}")
    print(f"Client ID:  {client_id}")
    print(f"Test mode:  {'Yes' if test else 'No'}")
    if vllm_url:
        print(f"vLLM URL:   {vllm_url}")
    print()

    cmd_parts = ["python -u -m", "gecco", "internal", "distributed-client"]
    cmd_parts.append(f"--config {config}")
    cmd_parts.append(f"--client-id {client_id}")
    if profile:
        cmd_parts.append(f"--client-profile {profile}")
    if vllm_url:
        cmd_parts.append(f'--vllm-url "{vllm_url}"')
    if test:
        cmd_parts.append("--test")

    full_cmd = " ".join(cmd_parts)

    print("=" * 60)
    print("COMMAND TO RUN INDEPENDENTLY:")
    print("=" * 60)
    print(full_cmd)
    print("=" * 60)
    print("TIP: The -u flag forces unbuffered output so you see results immediately")
    print()
    print("ALTERNATIVE (with environment variable):")
    print(f"  PYTHONUNBUFFERED=1 {full_cmd.replace('python -u -m', 'python -m')}")
    print()

    if print_cmd:
        return None
    if dry_run:
        print("(Dry run - command not executed)")
        return None

    if conda_env is not None:
        current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
        if current_env != conda_env:
            print(
                f"WARNING: Current conda environment is '{current_env}', expected '{conda_env}'"
            )
            print(f"         Activate with: conda activate {conda_env}")
            print()

    original_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        run_distributed_client(
            config=config,
            client_id=client_id,
            client_profile=profile,
            vllm_url=vllm_url,
            test=test,
        )
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(f"\nERROR: Client exited with code {exc.code}")
            raise
    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback

        traceback.print_exc()
        raise SystemExit(1) from exc
    finally:
        os.chdir(original_cwd)

    print("\nClient completed successfully!")
    return None


def main(args: argparse.Namespace) -> int | None:
    """Run the local-client command from parsed CLI arguments."""
    return run_local_client(
        config=args.config,
        profile=args.profile,
        client_id=args.client_id,
        vllm_url=args.vllm_url,
        test=args.test,
        dry_run=args.dry_run,
        print_cmd=args.print_cmd,
        conda_env=args.conda_env,
    )
