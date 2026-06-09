"""CLI route for distributed monitoring."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from gecco.coordination import SharedRegistry


console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def register_parser(subparsers) -> argparse.ArgumentParser:
    """Register the monitor command."""
    parser = subparsers.add_parser("monitor", help="Monitor a distributed GeCCo run")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--watch", type=int, default=None)
    parser.set_defaults(handler=main)
    return parser


def load_registry(results_dir: Path) -> dict | None:
    """Load the shared registry snapshot from DuckDB."""
    path = results_dir / "shared_registry.duckdb"
    if not path.exists():
        return None
    try:
        return SharedRegistry.open_existing(path).read()
    except (FileNotFoundError, OSError):
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
    for client_id in sorted(entries.keys(), key=int):
        entry = entries[client_id]
        status = entry.get("status", "unknown")
        style = (
            "green"
            if status in {"complete", "complete_no_success"}
            else "yellow"
            if status == "running"
            else "red"
        )
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
                    time_str = (
                        f"{int(age.total_seconds() // 3600)}h "
                        f"{int((age.total_seconds() % 3600) // 60)}m ago"
                    )
            except (ValueError, TypeError):
                time_str = updated
        else:
            time_str = "-"

        table.add_row(
            str(client_id),
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
    """Build a panel showing the baseline model results."""
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
            r2_strs = [f"{key}: {value:.3f}" for key, value in per_param.items()]
            lines.append(f"[dim]  {', '.join(r2_strs)}[/]")

    return Panel("\n".join(lines), title="Baseline Model (Template)", style="yellow")


def build_landscape_table(data, top_n=15):
    """Build a ranked table of all models across all clients."""
    history = data.get("iteration_history", [])
    all_models = []
    for entry in history:
        client_id = entry.get("client_id", "?")
        iteration = entry.get("iteration", "?")
        for result in entry.get("results", []):
            metric = result.get("metric_value")
            if metric is not None and metric < float("inf"):
                all_models.append(
                    {
                        "name": result.get("function_name", "?"),
                        "bic": metric,
                        "params": result.get("param_names", []),
                        "client": client_id,
                        "iter": iteration,
                        "mean_r2": result.get("mean_r2"),
                    }
                )

    if not all_models:
        return Panel("[dim]No models evaluated yet[/]", title="Model Landscape")

    all_models.sort(key=lambda item: item["bic"])
    has_r2 = any(model.get("mean_r2") is not None for model in all_models)

    table = Table(
        title=f"Model Landscape (top {min(top_n, len(all_models))} of {len(all_models)})",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Rank", justify="right", width=4)
    table.add_column("Model", width=20)
    table.add_column("BIC", justify="right", width=10)
    if has_r2:
        table.add_column("R²", justify="right", width=6)
    table.add_column("Params", width=40)
    table.add_column("Client", justify="center", width=6)
    table.add_column("Iter", justify="right", width=4)

    baseline = data.get("baseline")
    if baseline and baseline.get("metric_value") is not None:
        b_r2 = baseline.get("mean_r2")
        b_r2_str = f"{b_r2:.3f}" if b_r2 is not None else "-"
        b_param_str = ", ".join(baseline.get("param_names", []))
        baseline_row = ["—", "BASELINE (template)", f"{baseline['metric_value']:.2f}"]
        if has_r2:
            baseline_row.append(b_r2_str)
        baseline_row.extend([b_param_str, "—", "—"])
        table.add_row(*baseline_row, style="dim yellow")

    for index, model in enumerate(all_models[:top_n]):
        style = "bold green" if index == 0 else ""
        param_str = ", ".join(model["params"])
        r2 = model.get("mean_r2")
        r2_str = f"{r2:.3f}" if r2 is not None else "-"
        row = [str(index + 1), model["name"], f"{model['bic']:.2f}"]
        if has_r2:
            row.append(r2_str)
        row.extend([param_str, str(model["client"]), str(model["iter"])])
        table.add_row(*row, style=style)

    return table


def build_trajectory_table(data):
    """Build a per-client BIC trajectory table."""
    history = data.get("iteration_history", [])
    if not history:
        return Panel("[dim]No iteration data yet[/]", title="BIC Trajectory")

    client_iters = {}
    for entry in history:
        client_id = entry.get("client_id", "?")
        iteration = entry.get("iteration", 0)
        results = entry.get("results", [])
        if results:
            best_bic = min(result.get("metric_value", float("inf")) for result in results)
            client_iters.setdefault(client_id, {})[iteration] = best_bic

    if not client_iters:
        return Panel("[dim]No iteration data yet[/]", title="BIC Trajectory")

    all_iters = sorted({iteration for cdata in client_iters.values() for iteration in cdata.keys()})

    table = Table(title="Best BIC per Iteration", show_header=True, header_style="bold")
    table.add_column("Iter", justify="right", width=4)

    client_ids = sorted(
        client_iters.keys(), key=lambda item: int(item) if str(item).isdigit() else 999
    )
    for client_id in client_ids:
        table.add_column(f"Client {client_id}", justify="right", width=12)

    for iteration in all_iters:
        row = [str(iteration)]
        for client_id in client_ids:
            val = client_iters[client_id].get(iteration)
            row.append(f"{val:.2f}" if val is not None and val < float("inf") else "-")
        table.add_row(*row)

    return table


def build_summary_stats(data):
    """Build summary statistics."""
    history = data.get("iteration_history", [])
    entries = data.get("client_entries", {})
    tried = data.get("tried_param_sets", [])

    total_models = sum(len(entry.get("results", [])) for entry in history)
    total_iters = len(history)
    n_clients = len(entries)
    running = sum(1 for entry in entries.values() if entry.get("status") == "running")
    complete = sum(
        1
        for entry in entries.values()
        if entry.get("status") in {"complete", "complete_no_success"}
    )
    n_param_combos = len(tried)

    lines = [
        f"[bold]Clients:[/] {n_clients} ({running} running, {complete} complete)",
        f"[bold]Total iterations:[/] {total_iters}",
        f"[bold]Total models evaluated:[/] {total_models}",
        f"[bold]Unique param combos:[/] {n_param_combos}",
    ]
    return Panel("\n".join(lines), title="Summary", style="blue")


def build_r2_table(data):
    """Build a table showing R² for the best models by BIC."""
    history = data.get("iteration_history", [])
    models_with_r2 = []
    for entry in history:
        for result in entry.get("results", []):
            per_param = result.get("per_param_r2")
            if per_param and result.get("metric_value") is not None:
                models_with_r2.append(
                    {
                        "name": result.get("function_name", "?"),
                        "bic": result["metric_value"],
                        "mean_r2": result.get("mean_r2", 0.0),
                        "per_param_r2": per_param,
                        "client": entry.get("client_id", "?"),
                    }
                )

    if not models_with_r2:
        return None

    by_bic = sorted(models_with_r2, key=lambda item: item["bic"])[:5]
    by_r2 = sorted(models_with_r2, key=lambda item: -item["mean_r2"])[:3]

    shown = []
    seen = set()
    for model in by_bic + by_r2:
        key = (model["name"], model["client"])
        if key not in seen:
            shown.append(model)
            seen.add(key)

    all_params = []
    for model in shown:
        for param in model["per_param_r2"]:
            if param not in all_params:
                all_params.append(param)

    table = Table(title="Individual Differences R²", show_header=True, header_style="bold")
    table.add_column("Model", width=20)
    table.add_column("BIC", justify="right", width=10)
    table.add_column("Mean R²", justify="right", width=8)
    for param in all_params:
        table.add_column(param, justify="right", width=8)
    table.add_column("Client", justify="center", width=6)

    for model in shown:
        row = [model["name"], f"{model['bic']:.2f}", f"{model['mean_r2']:.3f}"]
        for param in all_params:
            val = model["per_param_r2"].get(param)
            row.append(f"{val:.3f}" if val is not None else "-")
        row.append(str(model["client"]))
        table.add_row(*row)

    return table


def render_dashboard(results_dir):
    """Render the full monitoring dashboard."""
    data = load_registry(results_dir)
    if data is None:
        return Panel(
            f"[yellow]Waiting for shared registry...[/]\n"
            f"[dim]Expected at: {results_dir / 'shared_registry.duckdb'}[/]",
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
    header = Text(
        f"GeCCo Distributed Monitor — {results_dir.name} — {timestamp}",
        style="bold blue",
    )
    return Group(header, *elements)


def run_monitor(*, task: str, results_dir: str | None = None, watch: int | None = None) -> int | None:
    """Monitor a distributed GeCCo run."""
    resolved_results_dir = Path(results_dir) if results_dir else PROJECT_ROOT / "results" / task

    if watch:
        try:
            with Live(
                render_dashboard(resolved_results_dir),
                console=console,
                refresh_per_second=0.5,
                screen=True,
            ) as live:
                while True:
                    time.sleep(watch)
                    live.update(render_dashboard(resolved_results_dir))
        except KeyboardInterrupt:
            console.print("\n[dim]Monitor stopped.[/]")
    else:
        console.print(render_dashboard(resolved_results_dir))
    return None


def main(args: argparse.Namespace) -> int | None:
    """Run the monitor command from parsed CLI arguments."""
    return run_monitor(task=args.task, results_dir=args.results_dir, watch=args.watch)
