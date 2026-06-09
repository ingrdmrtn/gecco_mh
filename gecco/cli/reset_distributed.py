"""CLI route for distributed reset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from config.schema import load_config
from gecco.utils import TimestampedConsole


console = TimestampedConsole()

# Subdirectories that accumulate during a distributed run
ARTIFACT_DIRS = ["models", "feedback", "bics", "parameters", "simulation"]

# Files in the results root to remove.
REGISTRY_FILES = []
DUCKDB_STATE_FILES = ["shared_registry.duckdb", "shared_registry.duckdb.lock"]
BASELINE_FILES = []


def register_parser(subparsers) -> argparse.ArgumentParser:
    """Register the reset command."""
    parser = subparsers.add_parser("reset", help="Reset distributed GeCCo run state")
    parser.add_argument("config")
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-y", "--yes", action="store_true")
    parser.set_defaults(handler=main)
    return parser


def get_results_dir(cfg) -> Path:
    """Derive the results directory from config."""
    task_name = cfg.task.name
    fit_type = getattr(cfg.evaluation, "fit_type", "group")
    suffix = "_individual" if fit_type == "individual" else ""
    return Path("results") / f"{task_name}{suffix}"


def scan_state(results_dir: Path, include_baseline: bool):
    """Scan for files and directories that would be removed."""
    items = []

    for name in REGISTRY_FILES:
        path = results_dir / name
        if path.exists():
            items.append(("file", path, path.stat().st_size))

    for name in DUCKDB_STATE_FILES:
        path = results_dir / name
        if path.exists():
            items.append(("file", path, path.stat().st_size))

    if include_baseline:
        for name in BASELINE_FILES:
            path = results_dir / name
            if path.exists():
                items.append(("file", path, path.stat().st_size))

    for dirname in ARTIFACT_DIRS:
        directory = results_dir / dirname
        if directory.exists() and any(directory.iterdir()):
            size = sum(file.stat().st_size for file in directory.rglob("*") if file.is_file())
            items.append(("dir", directory, size))

    for tmp_file in results_dir.glob("*.tmp"):
        items.append(("file", tmp_file, tmp_file.stat().st_size))

    return items


def format_size(nbytes: int) -> str:
    """Format a byte count for terminal output."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def show_summary(results_dir: Path, items, include_baseline: bool):
    """Display what will be removed."""
    table = Table(title="Items to remove", show_lines=False)
    table.add_column("Type", style="dim", width=6)
    table.add_column("Path", style="bold")
    table.add_column("Size", justify="right", style="cyan")

    total = 0
    for kind, path, size in items:
        rel = (
            path.relative_to(results_dir.parent.parent)
            if path.is_relative_to(results_dir.parent.parent)
            else path
        )
        table.add_row(kind, str(rel), format_size(size))
        total += size

    console.print(table)
    console.print(f"\nTotal: [cyan]{format_size(total)}[/]")

    preserved = []
    if not include_baseline:
        for name in BASELINE_FILES:
            path = results_dir / name
            if path.exists():
                preserved.append(str(path.relative_to(results_dir.parent.parent)))
    if preserved:
        console.print(
            Panel(
                "\n".join(preserved),
                title="[green]Preserved[/]",
                border_style="green",
            )
        )


def do_reset(items):
    """Delete the scanned items."""
    for kind, path, _ in items:
        if kind == "dir":
            shutil.rmtree(path)
            console.print(f"  [red]Removed directory[/] {path.name}/")
        else:
            path.unlink()
            console.print(f"  [red]Removed file[/]      {path.name}")


def run_reset(
    *,
    config: str,
    include_baseline: bool = False,
    dry_run: bool = False,
    yes: bool = False,
) -> int | None:
    """Reset distributed GeCCo search state for a task."""
    cfg = load_config(config)
    results_dir = get_results_dir(cfg)

    console.print(
        Panel(
            f"Task: [bold]{cfg.task.name}[/]\nResults dir: [bold]{results_dir}[/]",
            title="GeCCo Distributed Reset",
        )
    )

    if not results_dir.exists():
        console.print("[yellow]Results directory does not exist — nothing to reset.[/]")
        return None

    items = scan_state(results_dir, include_baseline=include_baseline)
    if not items:
        console.print("[green]No search state found — already clean.[/]")
        return None

    show_summary(results_dir, items, include_baseline=include_baseline)

    if dry_run:
        console.print("\n[dim]Dry run — no files were deleted.[/]")
        return None

    if not yes:
        confirm = console.input("\n[bold]Proceed with reset?[/] [y/N] ")
        if confirm.lower() not in ("y", "yes"):
            console.print("[dim]Aborted.[/]")
            return None

    console.print()
    do_reset(items)

    for dirname in ARTIFACT_DIRS:
        (results_dir / dirname).mkdir(parents=True, exist_ok=True)

    console.print("\n[bold green]Reset complete.[/] Ready for a fresh distributed run.")
    return None


def main(args: argparse.Namespace) -> int | None:
    """Run the reset command from parsed CLI arguments."""
    return run_reset(
        config=args.config,
        include_baseline=args.include_baseline,
        dry_run=args.dry_run,
        yes=args.yes,
    )
