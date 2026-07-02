"""CLI route for batched distributed launches."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from collections.abc import Sequence
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import launch_distributed as launch_distributed_cli
from .config_paths import config_output_subpath, resolve_config_path
from .launch_distributed import (
    _build_distributed_launch_context,
    _command_printer,
    _print_submission_result,
)
from .launcher_utils import (
    DEFAULT_SBATCH_RETRY_ATTEMPTS,
    DEFAULT_SBATCH_RETRY_BACKOFF_SECONDS,
    DEFAULT_SUBMIT_DELAY_SECONDS,
    LaunchCommand,
    LaunchExecutor,
    SubmissionResult,
)


console = Console()


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


def _render_batch_summary(
    *,
    batch_id: str,
    config_count: int,
    replicates: int,
    total_pipelines: int,
    max_concurrent_configs: int,
    dependency_policy: str,
    dry_run: bool,
) -> None:
    table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    table.add_column("Label", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Batch ID", batch_id)
    table.add_row("Configs", str(config_count))
    table.add_row("Replicates", str(replicates))
    table.add_row("Total pipelines", str(total_pipelines))
    table.add_row("Max concurrent configs/lanes", str(max_concurrent_configs))
    table.add_row("Dependency policy", dependency_policy)
    table.add_row("Mode", "dry-run" if dry_run else "submit")
    console.print(Panel(table, title="Distributed batch summary", border_style="blue"))


def _render_batch_plan(
    *,
    heading: str,
    planned_pipelines: list[tuple[int, int, str, int, str]],
) -> None:
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Pipeline", justify="right", no_wrap=True)
    table.add_column("Lane", justify="right", no_wrap=True)
    table.add_column("Config", overflow="fold")
    table.add_column("Replicate", justify="right", no_wrap=True)
    table.add_column("Run ID", overflow="fold")
    for pipeline_index, lane, config_display, replicate, run_id in planned_pipelines:
        table.add_row(
            f"{pipeline_index}",
            f"{lane}",
            config_display,
            f"{replicate}",
            run_id,
        )
    console.print(Panel(table, title=heading, border_style="cyan"))


def _render_pipeline_progress(
    *,
    pipeline_index: int,
    total_pipelines: int,
    lane: int,
    max_concurrent_configs: int,
    config_display: str,
    replicate: int,
    run_id: str,
    lane_dependencies_applied: bool,
) -> None:
    console.print(
        f"[bold]Pipeline {pipeline_index}/{total_pipelines}[/bold] "
        f"lane {lane}/{max_concurrent_configs} • config [cyan]{config_display}[/cyan] • "
        f"replicate {replicate} • run ID [yellow]{run_id}[/yellow] • "
        f"lane dependencies: {'yes' if lane_dependencies_applied else 'no'}"
    )


def _render_batch_completion(*, batch_id: str, total_pipelines: int, dry_run: bool) -> None:
    mode = "previewed" if dry_run else "submitted"
    console.print(
        Panel(
            f"Batch [bold]{batch_id}[/bold] {mode} [bold]{total_pipelines}[/bold] pipelines.",
            title="Distributed batch complete",
            border_style="green",
        )
    )


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
    parser.add_argument(
        "--submit-delay-seconds",
        type=float,
        default=DEFAULT_SUBMIT_DELAY_SECONDS,
        help=(
            "Seconds to wait between successful sbatch submissions "
            f"(default: {DEFAULT_SUBMIT_DELAY_SECONDS})."
        ),
    )
    parser.add_argument(
        "--sbatch-retry-attempts",
        type=int,
        default=DEFAULT_SBATCH_RETRY_ATTEMPTS,
        help=(
            "Maximum sbatch submission attempts for transient controller/socket errors "
            f"(default: {DEFAULT_SBATCH_RETRY_ATTEMPTS})."
        ),
    )
    parser.add_argument(
        "--sbatch-retry-backoff-seconds",
        type=float,
        default=DEFAULT_SBATCH_RETRY_BACKOFF_SECONDS,
        help=(
            "Base backoff in seconds for transient sbatch retries; each retry doubles "
            f"the delay starting from this value (default: {DEFAULT_SBATCH_RETRY_BACKOFF_SECONDS})."
        ),
    )
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
    submit_delay_seconds: float = DEFAULT_SUBMIT_DELAY_SECONDS,
    sbatch_retry_attempts: int = DEFAULT_SBATCH_RETRY_ATTEMPTS,
    sbatch_retry_backoff_seconds: float = DEFAULT_SBATCH_RETRY_BACKOFF_SECONDS,
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

    planned_pipelines: list[tuple[int, int, str, int, str]] = []
    for pipeline_index, (config_display, run_id) in enumerate(pipelines, start=1):
        lane = (pipeline_index - 1) % max_concurrent_configs + 1
        replicate = int(run_id.rsplit("-rep-", 1)[-1])
        planned_pipelines.append((pipeline_index, lane, config_display, replicate, run_id))

    _render_batch_summary(
        batch_id=batch_id,
        config_count=len(resolved_configs),
        replicates=replicates,
        total_pipelines=len(planned_pipelines),
        max_concurrent_configs=max_concurrent_configs,
        dependency_policy=dependency_policy,
        dry_run=dry_run,
    )
    plan_title = (
        "Configs to launch (explicit order)"
        if configs is not None
        else f"Configs found in {config_dir}"
    )
    _render_batch_plan(heading=plan_title, planned_pipelines=planned_pipelines)
    console.print()

    executor = LaunchExecutor(
        printer=_command_printer,
        submit_delay_seconds=submit_delay_seconds,
        sbatch_retry_attempts=sbatch_retry_attempts,
        sbatch_retry_backoff_seconds=sbatch_retry_backoff_seconds,
    )
    lane_results: list[dict[str, SubmissionResult]] = [
        {} for _ in range(max_concurrent_configs)
    ]
    lane_terminal_labels: list[tuple[str, ...] | None] = [None for _ in range(max_concurrent_configs)]

    for pipeline_index, (config_display, run_id) in enumerate(pipelines):
        lane = pipeline_index % max_concurrent_configs
        replicate = int(run_id.rsplit("-rep-", 1)[-1])
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

        pipeline_plan = _with_lane_dependencies(
            context.plan,
            lane_terminal_labels[lane],
            dependency_policy,
        )

        _render_pipeline_progress(
            pipeline_index=pipeline_index + 1,
            total_pipelines=len(pipelines),
            lane=lane + 1,
            max_concurrent_configs=max_concurrent_configs,
            config_display=config_display,
            replicate=replicate,
            run_id=run_id,
            lane_dependencies_applied=lane_terminal_labels[lane] is not None,
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

    _render_batch_completion(batch_id=batch_id, total_pipelines=len(pipelines), dry_run=dry_run)
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
        submit_delay_seconds=args.submit_delay_seconds,
        sbatch_retry_attempts=args.sbatch_retry_attempts,
        sbatch_retry_backoff_seconds=args.sbatch_retry_backoff_seconds,
        launch_orchestrator=args.launch_orchestrator,
        dry_run=args.dry_run,
    )
