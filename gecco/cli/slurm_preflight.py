"""SLURM wrapper preflight for provider detection and startup reporting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from gecco.sentry_init import capture_operational_error, init_sentry

from .config_paths import PROJECT_ROOT, resolve_config_path


def detect_provider(config: str, project_root: Path = PROJECT_ROOT) -> str:
    """Read the LLM provider from a GeCCo config file."""
    config_path = resolve_config_path(config, project_root=project_root)
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    if not isinstance(cfg, dict):
        cfg = {}

    llm_cfg = cfg.get("llm") or {}
    if not isinstance(llm_cfg, dict):
        llm_cfg = {}

    return str(llm_cfg.get("provider", "vllm"))


def main(argv: list[str] | None = None) -> int:
    """Detect the provider for SLURM wrappers and report startup failures."""
    parser = argparse.ArgumentParser(prog="gecco.cli.slurm_preflight")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    init_sentry(config_name=args.config, component="slurm_wrapper")

    resolved_path = resolve_config_path(args.config, project_root=PROJECT_ROOT)
    try:
        provider = detect_provider(args.config, project_root=PROJECT_ROOT)
    except Exception as exc:
        capture_operational_error(
            exc,
            component="slurm_wrapper",
            operation="detect_provider",
            config_name=args.config,
            config_path=str(resolved_path),
        )
        print(
            f"[GeCCo] ERROR: Failed to detect provider from config via preflight: {exc}",
            file=sys.stderr,
        )
        return 1

    print(provider)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
