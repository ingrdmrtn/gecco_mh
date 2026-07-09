"""CLI route for results comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gecco.results_comparison import (
    ComparisonThresholds,
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
    parser.add_argument(
        "--min-test-mean-nll",
        type=float,
        default=None,
        help=(
            "Minimum acceptable mean NLL for code=NULL test rows "
            "(default: 1.0).  Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--min-test-metric-value",
        type=float,
        default=None,
        help=(
            "Minimum acceptable metric value for code=NULL test rows "
            "(default: 20.0).  Set to 0 to disable."
        ),
    )
    parser.set_defaults(handler=main)
    return parser


def main(args: argparse.Namespace) -> int | None:
    """Run the comparison command from parsed CLI arguments."""
    results_dirs = [Path(d) for d in args.results_dirs]
    output_dir = Path(args.output)

    # Build thresholds from CLI args (or use defaults).
    # Use getattr for backward compat with callers passing SimpleNamespace
    # that may not have these attributes.
    cli_mean_nll = getattr(args, "min_test_mean_nll", None)
    cli_metric_value = getattr(args, "min_test_metric_value", None)
    thresholds = ComparisonThresholds(
        min_mean_nll=cli_mean_nll if cli_mean_nll is not None else 1.0,
        min_metric_value=cli_metric_value if cli_metric_value is not None else 20.0,
    )
    # A value of 0 means "disable" the threshold (allow all)
    if cli_mean_nll is not None and cli_mean_nll == 0.0:
        thresholds = ComparisonThresholds(
            min_mean_nll=None,
            min_metric_value=thresholds.min_metric_value,
        )
    if cli_metric_value is not None and cli_metric_value == 0.0:
        thresholds = ComparisonThresholds(
            min_mean_nll=thresholds.min_mean_nll,
            min_metric_value=None,
        )

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
    summaries = [summarise_run(d, thresholds=thresholds) for d in discovered]
    config_rows = aggregate_configs(summaries)
    print(f"  Configs: {len(config_rows)}")

    # Print exclusion warnings
    for s in summaries:
        warnings = s.get("exclusion_warnings", [])
        if warnings:
            run_id = s.get("run_id", "?")
            for w in warnings:
                print(f"  [yellow]Warning ({run_id}): {w}[/]")

    # Print baseline diagnostics
    for s in summaries:
        diag = s.get("baseline_diagnostics", [])
        if diag:
            run_id = s.get("run_id", "?")
            for msg in diag:
                print(f"  [cyan]Baseline ({run_id}): {msg}[/]")

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
