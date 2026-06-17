"""Unified GeCCo CLI routing."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from . import launch_distributed
from . import monitor_distributed
from . import reset_distributed
from . import run_gecco_distributed
from . import run_judge_orchestrator
from . import run_local_client
from . import run_test_evaluation


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level GeCCo parser."""
    parser = argparse.ArgumentParser(prog="gecco", description="GeCCo command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Launch GeCCo workflows")
    run_subparsers = run_parser.add_subparsers(dest="run_command", required=True)
    launch_distributed.register_parser(run_subparsers)
    run_local_client.register_parser(run_subparsers)

    monitor_distributed.register_parser(subparsers)
    reset_distributed.register_parser(subparsers)

    internal_parser = subparsers.add_parser("internal", help=argparse.SUPPRESS)
    internal_subparsers = internal_parser.add_subparsers(
        dest="internal_command", required=True
    )
    run_gecco_distributed.register_parser(internal_subparsers)
    run_test_evaluation.register_parser(internal_subparsers)
    run_judge_orchestrator.register_parser(internal_subparsers)

    return parser


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int | None:
    """Run the GeCCo CLI."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)
