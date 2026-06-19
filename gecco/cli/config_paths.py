"""Helpers for resolving GeCCo configuration paths."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_config_path(config: str, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a config path using GeCCo's config-root semantics."""
    config_path = Path(config)
    if config_path.is_absolute():
        return config_path
    return project_root / "config" / config_path
