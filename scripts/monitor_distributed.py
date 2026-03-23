#!/usr/bin/env python
"""
Live monitoring dashboard for distributed GeCCo search.

Usage:
    python scripts/monitor_distributed.py --task two_step_factors
    python scripts/monitor_distributed.py --task two_step_factors --watch 10

Reads the shared registry and result files to display:
- Client status and progress
- Global best model and BIC trajectory
- Model landscape (top models across all clients)
- Per-client iteration history
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.columns import Columns

console = Console()


def load_registry(results_dir):
    """Load the shared registry, returning None if it doesn't exist yet."""
    path = results_dir / "shared_registry.json"
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def build_client_table(data):
    """Build a table showing each client's status."""
    table = Table(title="Client Status", show_header=True, header_style="bold")
    table.add_column("Client", justify="center")
    table.add_column("Status", justify="center")
    table.add_column("Last Iter", justify="right")
    table.add_column("Best BIC", justify="right")
    table.add_column("Last Update", justify="right")

    entries = data.get("client_entries", {})
    for cid in sorted(entries.keys(), key=int):
        entry = entries[cid]
        status = entry.get("status", "unknown")
        style = "green" if status == "complete" else "yellow" if status == "running" else "red"
        status_text = Text(status, style=style)

        best = entry.get("best_metric")
        best_str = f"{best:.2f}" if best is not None else "-"

        updated = entry.get("updated_at", "")
        if updated:
            try:
                dt = datetime.fromisoformat(updated)
                age = datetime.now() - dt
                if age.total_seconds() < 60:
                    time_str = f"{int(age.total_seconds())}s ago"
                elif age.total_seconds() < 3600:
                    time_str = f"{int(age.total_seconds() // 60)}m ago"
                else:
                    time_str = f"{int(age.total_seconds() // 3600)}h {int((age.total_seconds() % 3600) // 60)}m ago"
            except (ValueError, TypeError):
                time_str = updated
        else:
            time_str = "-"

        table.add_row(
            str(cid),
            status_text,
            str(entry.get("last_iteration", "-")),
            best_str,
            time_str,
        )

    return table


def build_global_best_panel(data):
    """Build a panel showing the global best model and baseline comparison."""
    best = data.get("global_best")
    baseline = data.get("baseline")

    if not best:
        return Panel("[dim]No models fitted yet[/]", title="Global Best", style="blue")

    lines = [
        f"[bold]BIC:[/] [cyan]{best['metric_value']:.2f}[/]",
        f"[bold]Found by:[/] Client {best['client_id']}, Iteration {best['iteration']}",
        f"[bold]Parameters:[/] {', '.join(best.get('param_names', []))}",
    ]

    if baseline and baseline.get("metric_value") is not None:
        improvement = baseline["metric_value"] - best["metric_value"]
        pct = 100 * improvement / abs(baseline["metric_value"]) if baseline["metric_value"] != 0 else 0
        style = "green" if improvement > 0 else "red"
        lines.append(
            f"[bold]vs Baseline:[/] [{style}]{improvement:+.2f} ({pct:+.1f}%)[/{style}]"
        )

    return Panel("\n".join(lines), title="Global Best Model", style="green")


def build_baseline_panel(data):
    """Build a panel showing the baseline (template) model results."""
    baseline = data.get("baseline")
    if not baseline or baseline.get("metric_value") is None:
        return Panel("[dim]Baseline not yet fitted[/]", title="Baseline Model", style="dim")

    lines = [
        f"[bold]BIC:[/] [yellow]{baseline['metric_value']:.2f}[/]",
        f"[bold]Parameters:[/] {', '.join(baseline.get('param_names', []))}",
    ]

    r2 = baseline.get("mean_r2")
    if r2 is not None:
        lines.append(f"[bold]Mean R²:[/] {r2:.3f}")
        per_param = baseline.get("per_param_r2", {})
        if per_param:
            r2_strs = [f"{k}: {v:.3f}" for k, v in per_param.items()]
            lines.append(f"[dim]  {', '.join(r2_strs)}[/]")

    return Panel("\n".join(lines), title="Baseline Model (Template)", style="yellow")


def build_landscape_table(data, top_n=15):
    """Build a ranked table of all models across all clients."""
    history = data.get("iteration_history", [])
    all_models = []
    for entry in history:
        client_id = entry.get("client_id", "?")
        iteration = entry.get("iteration", "?")
        for r in entry.get("results", []):
            metric = r.get("metric_value")
            if metric is not None and metric < float("inf"):
                all_models.append({
                    "name": r.get("function_name", "?"),
                    "bic": metric,
                    "params": r.get("param_names", []),
                    "client": client_id,
                    "iter": iteration,
                    "mean_r2": r.get("mean_r2"),
                })

    if not all_models:
        return Panel("[dim]No models evaluated yet[/]", title="Model Landscape")

    all_models.sort(key=lambda x: x["bic"])

    # Check if any model has R² data
    has_r2 = any(m.get("mean_r2") is not None for m in all_models)

    table = Table(title=f"Model Landscape (top {min(top_n, len(all_models))} of {len(all_models)})",
                  show_header=True, header_style="bold")
    table.add_column("Rank", justify="right", width=4)
    table.add_column("Model", width=20)
    table.add_column("BIC", justify="right", width=10)
    if has_r2:
        table.add_column("R²", justify="right", width=6)
    table.add_column("Params", width=40)
    table.add_column("Client", justify="center", width=6)
    table.add_column("Iter", justify="right", width=4)

    # Add baseline as reference row first (if available)
    baseline = data.get("baseline")
    if baseline and baseline.get("metric_value") is not None:
        b_r2 = baseline.get("mean_r2")
        b_r2_str = f"{b_r2:.3f}" if b_r2 is not None else "-"
        b_param_str = ", ".join(baseline.get("param_names", []))
        b_row = ["—", "BASELINE (template)", f"{baseline['metric_value']:.2f}"]
        if has_r2:
            b_row.append(b_r2_str)
        b_row.extend([b_param_str, "—", "—"])
        table.add_row(*b_row, style="dim yellow")

    for i, m in enumerate(all_models[:top_n]):
        style = "bold green" if i == 0 else ""
        param_str = ", ".join(m["params"])
        r2 = m.get("mean_r2")
        r2_str = f"{r2:.3f}" if r2 is not None else "-"
        row = [str(i + 1), m["name"], f"{m['bic']:.2f}"]
        if has_r2:
            row.append(r2_str)
        row.extend([param_str, str(m["client"]), str(m["iter"])])
        table.add_row(*row, style=style)

    return table


def build_trajectory_table(data):
    """Build a per-client BIC trajectory table."""
    history = data.get("iteration_history", [])
    if not history:
        return Panel("[dim]No iteration data yet[/]", title="BIC Trajectory")

    # Group by client, track best BIC per iteration
    client_iters = {}
    for entry in history:
        cid = entry.get("client_id", "?")
        it = entry.get("iteration", 0)
        results = entry.get("results", [])
        if results:
            best_bic = min(r.get("metric_value", float("inf")) for r in results)
            if cid not in client_iters:
                client_iters[cid] = {}
            client_iters[cid][it] = best_bic

    if not client_iters:
        return Panel("[dim]No iteration data yet[/]", title="BIC Trajectory")

    # Find all iteration numbers
    all_iters = sorted(set(it for cdata in client_iters.values() for it in cdata.keys()))

    table = Table(title="Best BIC per Iteration", show_header=True, header_style="bold")
    table.add_column("Iter", justify="right", width=4)

    client_ids = sorted(client_iters.keys(), key=lambda x: int(x) if str(x).isdigit() else 999)
    for cid in client_ids:
        table.add_column(f"Client {cid}", justify="right", width=12)

    for it in all_iters:
        row = [str(it)]
        for cid in client_ids:
            val = client_iters[cid].get(it)
            if val is not None and val < float("inf"):
                row.append(f"{val:.2f}")
            else:
                row.append("-")
        table.add_row(*row)

    return table


def build_summary_stats(data):
    """Build summary statistics."""
    history = data.get("iteration_history", [])
    entries = data.get("client_entries", {})
    tried = data.get("tried_param_sets", [])

    total_models = sum(
        len(e.get("results", []))
        for e in history
    )
    total_iters = len(history)
    n_clients = len(entries)
    running = sum(1 for e in entries.values() if e.get("status") == "running")
    complete = sum(1 for e in entries.values() if e.get("status") == "complete")
    n_param_combos = len(tried)

    lines = [
        f"[bold]Clients:[/] {n_clients} ({running} running, {complete} complete)",
        f"[bold]Total iterations:[/] {total_iters}",
        f"[bold]Total models evaluated:[/] {total_models}",
        f"[bold]Unique param combos:[/] {n_param_combos}",
    ]
    return Panel("\n".join(lines), title="Summary", style="blue")


def build_r2_table(data):
    """Build a table showing R² for the best models (by BIC)."""
    history = data.get("iteration_history", [])

    # Collect models that have per-param R²
    models_with_r2 = []
    for entry in history:
        for r in entry.get("results", []):
            per_param = r.get("per_param_r2")
            if per_param and r.get("metric_value") is not None:
                models_with_r2.append({
                    "name": r.get("function_name", "?"),
                    "bic": r["metric_value"],
                    "mean_r2": r.get("mean_r2", 0.0),
                    "per_param_r2": per_param,
                    "client": entry.get("client_id", "?"),
                })

    if not models_with_r2:
        return None  # Don't show panel if no R² data exists

    # Show top 5 by BIC, plus the top 3 by R² if different
    by_bic = sorted(models_with_r2, key=lambda x: x["bic"])[:5]
    by_r2 = sorted(models_with_r2, key=lambda x: -x["mean_r2"])[:3]

    # Merge, deduplicate, keep order
    shown = []
    seen = set()
    for m in by_bic + by_r2:
        key = (m["name"], m["client"])
        if key not in seen:
            shown.append(m)
            seen.add(key)

    # Collect all parameter names across shown models
    all_params = []
    for m in shown:
        for p in m["per_param_r2"]:
            if p not in all_params:
                all_params.append(p)

    table = Table(title="Individual Differences R²", show_header=True, header_style="bold")
    table.add_column("Model", width=20)
    table.add_column("BIC", justify="right", width=10)
    table.add_column("Mean R²", justify="right", width=8)
    for p in all_params:
        table.add_column(p, justify="right", width=8)
    table.add_column("Client", justify="center", width=6)

    for m in shown:
        row = [m["name"], f"{m['bic']:.2f}", f"{m['mean_r2']:.3f}"]
        for p in all_params:
            val = m["per_param_r2"].get(p)
            row.append(f"{val:.3f}" if val is not None else "-")
        row.append(str(m["client"]))
        table.add_row(*row)

    return table


def render_dashboard(results_dir):
    """Render the full monitoring dashboard."""
    data = load_registry(results_dir)

    if data is None:
        return Panel(
            f"[yellow]Waiting for shared registry...[/]\n"
            f"[dim]Expected at: {results_dir / 'shared_registry.json'}[/]",
            title="GeCCo Distributed Monitor",
            style="yellow",
        )

    from rich.console import Group

    elements = [
        build_summary_stats(data),
        build_baseline_panel(data),
        build_client_table(data),
        build_global_best_panel(data),
        build_trajectory_table(data),
        build_landscape_table(data),
    ]
    r2_table = build_r2_table(data)
    if r2_table:
        elements.append(r2_table)
    timestamp = datetime.now().strftime("%H:%M:%S")
    header = Text(f"GeCCo Distributed Monitor — {results_dir.name} — {timestamp}", style="bold blue")

    return Group(header, *elements)


def main():
    parser = argparse.ArgumentParser(description="Monitor distributed GeCCo search")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (matches task.name in config)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results directory path")
    parser.add_argument("--watch", type=int, default=None,
                        help="Refresh interval in seconds (omit for one-shot)")
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        project_root = Path(__file__).resolve().parents[1]
        results_dir = project_root / "results" / args.task

    if args.watch:
        try:
            with Live(render_dashboard(results_dir), console=console,
                       refresh_per_second=0.5, screen=True) as live:
                while True:
                    time.sleep(args.watch)
                    live.update(render_dashboard(results_dir))
        except KeyboardInterrupt:
            console.print("\n[dim]Monitor stopped.[/]")
    else:
        console.print(render_dashboard(results_dir))


if __name__ == "__main__":
    main()
