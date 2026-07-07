"""CLI route for results comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gecco.results_comparison import (
    aggregate_configs,
    discover_run_dirs,
    export_figures,
    render_report_html,
    summarise_run,
    write_config_summary_csv,
    write_results_csv,
)


def register_parser(subparsers) -> argparse.ArgumentParser:
    """Register the ``compare`` subcommand under ``run``."""
    parser = subparsers.add_parser(
        "compare",
        help="Compare results from multiple GeCCo run directories",
        description=(
            "Discover run directories containing diagnostic DuckDB files "
            "under one or more results directories, export a run-level "
            "results.csv, a config-level config_summary.csv, a static "
            "report.html, and PNG/PDF figures."
        ),
    )
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        required=True,
        help="One or more result directories to search recursively for diagnostic DuckDB files",
    )
    parser.add_argument(
        "--output",
        default="comparison",
        help="Output directory for generated artifacts (default: comparison)",
    )
    parser.set_defaults(handler=main)
    return parser


def main(args: argparse.Namespace) -> int | None:
    """Run the comparison command from parsed CLI arguments."""
    results_dirs = [Path(d) for d in args.results_dirs]
    output_dir = Path(args.output)

    # Validate directories exist
    missing = [str(d) for d in results_dirs if not d.is_dir()]
    if missing:
        print(
            f"Error: Results directory/directories not found: {', '.join(missing)}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Discover
    discovered = discover_run_dirs(results_dirs)
    if not discovered:
        print(
            "Error: No diagnostic DuckDB files found under the specified directories.\n"
            "Use --results-dirs to point to directories containing diagnostics.duckdb "
            "or diagnostics_unified.duckdb files.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"Discovered {len(discovered)} run director{'ies' if len(discovered) != 1 else 'y'}")

    # Summarise
    summaries = [summarise_run(d) for d in discovered]
    config_rows = aggregate_configs(summaries)
    print(f"  Configs: {len(config_rows)}")

    # Print exclusion warnings
    for s in summaries:
        warnings = s.get("exclusion_warnings", [])
        if warnings:
            run_id = s.get("run_id", "?")
            for w in warnings:
                print(f"  [yellow]Warning ({run_id}): {w}[/]")

    # Export
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = write_results_csv(summaries, output_dir)
    print(f"  CSV (run):   {csv_path}")

    config_csv_path = write_config_summary_csv(config_rows, output_dir)
    print(f"  CSV (config): {config_csv_path}")

    html_path = render_report_html(summaries, output_dir, config_rows=config_rows)
    print(f"  HTML:  {html_path}")

    fig_dir = export_figures(summaries, output_dir, config_rows=config_rows)
    print(f"  Figs:  {fig_dir}/")

    return None
