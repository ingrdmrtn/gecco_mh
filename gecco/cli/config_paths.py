"""Helpers for resolving GeCCo configuration paths."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def config_output_subpath(
    config: str | Path,
    *,
    project_root: Path = PROJECT_ROOT,
    individual: bool = False,
) -> Path:
    """Return the mirrored output subpath for a config file."""

    config_path = resolve_config_path(str(config), project_root=project_root)
    config_root = project_root / "config"
    try:
        mirrored = config_path.relative_to(config_root)
    except ValueError:
        mirrored = Path(config_path.name)

    if mirrored.suffix:
        mirrored = mirrored.with_suffix("")
    if individual:
        mirrored = mirrored.with_name(f"{mirrored.name}_individual")
    return mirrored


def results_dir_for_config(
    config: str | Path,
    *,
    project_root: Path = PROJECT_ROOT,
    fit_type: str = "group",
    run_id: str | None = None,
) -> Path:
    """Return the canonical results directory for a config file."""

    subpath = config_output_subpath(
        config,
        project_root=project_root,
        individual=fit_type == "individual",
    )
    if run_id:
        subpath = subpath / run_id
    return project_root / "results" / subpath


def logs_dir_for_config(
    config: str | Path,
    *,
    project_root: Path = PROJECT_ROOT,
    fit_type: str = "group",
    run_id: str | None = None,
) -> Path:
    """Return the canonical logs directory for a config file."""

    subpath = config_output_subpath(
        config,
        project_root=project_root,
        individual=fit_type == "individual",
    )
    if run_id:
        subpath = subpath / run_id
    return project_root / "logs" / subpath


def resolve_config_path(config: str, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a config path using GeCCo's config-root semantics."""
    config_path = Path(config)
    if config_path.is_absolute():
        return config_path

    parts = config_path.parts
    if parts and parts[0] == ".":
        parts = parts[1:]
    if parts and parts[0] == "config":
        parts = parts[1:]

    relative = Path(*parts) if parts else Path()
    return project_root / "config" / relative
