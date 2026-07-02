"""CLI route for batched distributed launches."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from collections.abc import Sequence
from pathlib import Path

from . import launch_distributed as launch_distributed_cli
from .config_paths import config_output_subpath, resolve_config_path
from .launch_distributed import (
    _build_distributed_launch_context,
    _command_printer,
    _print_config_table,
    _print_submission_result,
)
from .launcher_utils import LaunchCommand, LaunchExecutor, SubmissionResult


def _display_config_string(config_path: Path) -> str:
    try:
        return str(config_path.relative_to(launch_distributed_cli.PROJECT_ROOT))
    except ValueError:
        return str(config_path)


def _resolve_explicit_configs(configs: Sequence[str]) -> list[tuple[str, Path]]:
    if not configs:
        print("ERROR: --configs must include at least one config path")
        raise SystemExit(1)

    resolved_configs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for config in configs:
        config_path = resolve_config_path(config, project_root=launch_distributed_cli.PROJECT_ROOT)
        key = str(config_path)
        if key in seen:
            print(f"ERROR: Duplicate config path supplied: {config_path}")
            raise SystemExit(1)
        seen.add(key)
        resolved_configs.append((config, config_path))
    return resolved_configs


def _resolve_config_dir_configs(config_dir: str, config_glob: str) -> list[tuple[str, Path]]:
    config_dir_path = resolve_config_path(config_dir, project_root=launch_distributed_cli.PROJECT_ROOT)
    if not config_dir_path.exists() or not config_dir_path.is_dir():
        print(f"ERROR: Config directory not found: {config_dir_path}")
        raise SystemExit(1)

    config_paths = sorted(path for path in config_dir_path.glob(config_glob) if path.is_file())
    if not config_paths:
        print(
            f"ERROR: No configs matched {config_glob!r} in {config_dir_path}"
        )
        raise SystemExit(1)
    return [(_display_config_string(path), path) for path in config_paths]


def _config_run_id(config_path: Path, batch_id: str, replicate: int) -> str:
    subpath = config_output_subpath(config_path, project_root=launch_distributed_cli.PROJECT_ROOT)
    slug = str(subpath).replace("/", "__")
    return f"{batch_id}-{slug}-rep-{replicate:03d}"


def _dependency_fallback(dependency_policy: str, dependency_labels: tuple[str, ...]) -> str | None:
    if not dependency_labels:
        return None
    placeholder_ids = ":".join(f"<{label}_job_id>" for label in dependency_labels)
    return f"--dependency={dependency_policy}:{placeholder_ids}"


def _with_lane_dependencies(
    plan, dependency_labels: tuple[str, ...] | None, dependency_policy: str
):
    if not dependency_labels:
        return plan

    fallback = _dependency_fallback(dependency_policy, dependency_labels)
    commands: list[LaunchCommand] = []
    for command in plan.commands:
        if command.dependency_labels:
            commands.append(command)
        else:
            command_text = command.command
            if "{dependency}" not in command_text:
                command_text = command_text.replace("sbatch ", "sbatch {dependency} ", 1)
            commands.append(
                replace(
                    command,
                    command=command_text,
                    dependency_labels=dependency_labels,
                    dependency_policy=dependency_policy,
                    dependency_fallback=fallback,
                    lane_dependency=True,
                )
            )
    return type(plan)(commands=tuple(commands))


def register_parser(subparsers) -> argparse.ArgumentParser:
    """Register the distributed batch launcher subcommand."""
    parser = subparsers.add_parser(
        "distributed-batch", help="Launch multiple distributed GeCCo runs"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--configs", nargs="+", default=None)
    input_group.add_argument("--config-dir", type=str, default=None)
    parser.add_argument("--config-glob", type=str, default="*.yaml")
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--max-concurrent-configs", type=int, default=1)
    parser.add_argument(
        "--dependency-policy",
        choices=("afterany", "afterok"),
        default="afterany",
    )
    parser.add_argument("--vllm-url", type=str, default=None)
    parser.add_argument("--conda-env", type=str, default=None)
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--cpus-per-task", type=int, default=None)
    parser.add_argument("--mem", type=str, default=None)
    parser.add_argument("--launch-orchestrator", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(handler=main)
    return parser


def run_distributed_batch_launcher(
    *,
    configs: Sequence[str] | None = None,
    config_dir: str | None = None,
    config_glob: str = "*.yaml",
    replicates: int = 1,
    max_concurrent_configs: int = 1,
    dependency_policy: str = "afterany",
    vllm_url: str | None = None,
    conda_env: str | None = None,
    partition: str | None = None,
    cpus_per_task: int | None = None,
    mem: str | None = None,
    launch_orchestrator: bool = False,
    dry_run: bool = False,
) -> int | None:
    """Launch batched distributed GeCCo pipelines."""
    if replicates <= 0:
        print("ERROR: --replicates must be a positive integer")
        raise SystemExit(1)
    if max_concurrent_configs <= 0:
        print("ERROR: --max-concurrent-configs must be a positive integer")
        raise SystemExit(1)

    if configs is not None:
        resolved_configs = _resolve_explicit_configs(configs)
    elif config_dir is not None:
        resolved_configs = _resolve_config_dir_configs(config_dir, config_glob)
    else:
        print("ERROR: Provide either --configs or --config-dir")
        raise SystemExit(1)

    pipelines: list[tuple[str, str]] = []
    batch_id = datetime.now().strftime("batch-%Y%m%d-%H%M%S")
    for config_display, config_path in resolved_configs:
        for replicate in range(1, replicates + 1):
            pipelines.append(
                (
                    config_display,
                    _config_run_id(config_path, batch_id, replicate),
                )
            )

    executor = LaunchExecutor(printer=_command_printer)
    lane_results: list[dict[str, SubmissionResult]] = [
        {} for _ in range(max_concurrent_configs)
    ]
    lane_terminal_labels: list[tuple[str, ...] | None] = [None for _ in range(max_concurrent_configs)]

    for pipeline_index, (config_display, run_id) in enumerate(pipelines):
        lane = pipeline_index % max_concurrent_configs
        context = _build_distributed_launch_context(
            config=config_display,
            vllm_url=vllm_url,
            conda_env=conda_env,
            partition=partition,
            cpus_per_task=cpus_per_task,
            mem=mem,
            run_id=run_id,
            launch_orchestrator=launch_orchestrator,
        )

        _print_config_table(list(context.config_table_rows))
        print()

        pipeline_plan = _with_lane_dependencies(
            context.plan,
            lane_terminal_labels[lane],
            dependency_policy,
        )

        if not dry_run:
            context.logs_dir.mkdir(parents=True, exist_ok=True)

        pipeline_results = executor.execute(
            pipeline_plan,
            dry_run=dry_run,
            on_result=_print_submission_result,
            prior_results=lane_results[lane],
        )
        lane_results[lane] = {result.label: result for result in pipeline_results}
        lane_terminal_labels[lane] = context.terminal_labels

    return None


def main(args: argparse.Namespace) -> int | None:
    """Run the distributed batch launcher from parsed CLI arguments."""
    return run_distributed_batch_launcher(
        configs=args.configs,
        config_dir=args.config_dir,
        config_glob=args.config_glob,
        replicates=args.replicates,
        max_concurrent_configs=args.max_concurrent_configs,
        dependency_policy=args.dependency_policy,
        vllm_url=args.vllm_url,
        conda_env=args.conda_env,
        partition=args.partition,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        launch_orchestrator=args.launch_orchestrator,
        dry_run=args.dry_run,
    )
