#!/usr/bin/env python3
"""
Reset distributed GeCCo search state for a task.

Removes the shared registry, generated models, feedback, BICs, parameters,
and simulation artifacts — while preserving the baseline model fit.

Usage:
    python scripts/reset_distributed.py config/my_config.yaml
    python scripts/reset_distributed.py config/my_config.yaml --include-baseline
    python scripts/reset_distributed.py config/my_config.yaml --dry-run
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.schema import load_config

console = Console()

# Subdirectories that accumulate during a distributed run
ARTIFACT_DIRS = ["models", "feedback", "bics", "parameters", "simulation"]

# Files in the results root to remove (baseline files listed separately)
REGISTRY_FILES = ["shared_registry.json"]
BASELINE_FILES = ["baseline.json", "baseline.lock"]


def get_results_dir(cfg) -> Path:
    """Derive the results directory from config, matching run_gecco_distributed.py logic."""
    task_name = cfg.task.name
    fit_type = getattr(cfg.evaluation, "fit_type", "group")
    suffix = "_individual" if fit_type == "individual" else ""
    return Path("results") / f"{task_name}{suffix}"


def scan_state(results_dir: Path, include_baseline: bool):
    """Scan for files/dirs that would be removed. Returns (items, total_bytes)."""
    items = []

    for name in REGISTRY_FILES:
        p = results_dir / name
        if p.exists():
            items.append(("file", p, p.stat().st_size))

    if include_baseline:
        for name in BASELINE_FILES:
            p = results_dir / name
            if p.exists():
                items.append(("file", p, p.stat().st_size))

    for dirname in ARTIFACT_DIRS:
        d = results_dir / dirname
        if d.exists() and any(d.iterdir()):
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            items.append(("dir", d, size))

    # Stray .tmp files from failed atomic writes
    for tmp in results_dir.glob("*.tmp"):
        items.append(("file", tmp, tmp.stat().st_size))

    return items


def format_size(nbytes: int) -> str:
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
        rel = path.relative_to(results_dir.parent.parent) if path.is_relative_to(results_dir.parent.parent) else path
        table.add_row(kind, str(rel), format_size(size))
        total += size

    console.print(table)
    console.print(f"\nTotal: [cyan]{format_size(total)}[/]")

    # Show what's preserved
    preserved = []
    if not include_baseline:
        for name in BASELINE_FILES:
            p = results_dir / name
            if p.exists():
                preserved.append(str(p.relative_to(results_dir.parent.parent)))
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


def main():
    parser = argparse.ArgumentParser(
        description="Reset distributed GeCCo search state",
    )
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Also remove the cached baseline model fit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting anything",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = get_results_dir(cfg)

    console.print(
        Panel(
            f"Task: [bold]{cfg.task.name}[/]\nResults dir: [bold]{results_dir}[/]",
            title="GeCCo Distributed Reset",
        )
    )

    if not results_dir.exists():
        console.print("[yellow]Results directory does not exist — nothing to reset.[/]")
        return

    items = scan_state(results_dir, include_baseline=args.include_baseline)

    if not items:
        console.print("[green]No search state found — already clean.[/]")
        return

    show_summary(results_dir, items, include_baseline=args.include_baseline)

    if args.dry_run:
        console.print("\n[dim]Dry run — no files were deleted.[/]")
        return

    if not args.yes:
        confirm = console.input("\n[bold]Proceed with reset?[/] [y/N] ")
        if confirm.lower() not in ("y", "yes"):
            console.print("[dim]Aborted.[/]")
            return

    console.print()
    do_reset(items)

    # Recreate empty artifact directories so next run doesn't need to
    for dirname in ARTIFACT_DIRS:
        (results_dir / dirname).mkdir(parents=True, exist_ok=True)

    console.print("\n[bold green]Reset complete.[/] Ready for a fresh distributed run.")


if __name__ == "__main__":
    main()
