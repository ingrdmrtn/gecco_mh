"""Contract tests for the results comparison CLI and helpers."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from gecco.diagnostic_store.store import DiagnosticStore


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _create_minimal_diagnostics(db_path: Path, *, label: str = "run") -> None:
    """Create a minimal diagnostics DuckDB with one iteration and model rows
    on train, val, and optionally test splits."""
    store = DiagnosticStore(db_path)
    store.write_iteration(
        iteration=0,
        run_idx=0,
        iteration_results=[
            {
                "function_name": f"{label}_model_a",
                "metric_name": "BIC",
                "metric_value": 120.0,
                "mean_nll": 3.5,
                "param_names": ["alpha"],
                "val_metric_value": 125.0,
                "val_mean_nll": 3.8,
            },
            {
                "function_name": f"{label}_model_b",
                "metric_name": "BIC",
                "metric_value": 100.0,
                "mean_nll": 2.8,
                "param_names": ["beta"],
                "val_metric_value": 105.0,
                "val_mean_nll": 3.0,
            },
        ],
    )
    store.close()

    # Add a test-split model row using write_top_model_test
    store2 = DiagnosticStore(db_path)
    store2.write_top_model_test(
        {
            "model_name": f"{label}_model_a",
            "val_nll": 3.8,
            "test_mean_BIC": 130.0,
            "test_mean_NLL": 4.0,
            "test_individual_BIC": [130.0],
            "test_individual_NLL": [4.0],
            "test_individual_differences": {
                "mean_r2": 0.75,
                "max_r2": 0.85,
                "best_param": "alpha",
                "per_param_r2": {"alpha": 0.85},
                "per_param_detail": {
                    "alpha": {
                        "r2": 0.85,
                        "slope": 0.9,
                        "intercept": 0.1,
                    }
                },
            },
        }
    )
    store2.close()


def _create_minimal_diagnostics_train_only(db_path: Path, *, label: str = "train_only") -> None:
    """Create a minimal diagnostics DuckDB with train rows only (no test/ID data)."""
    store = DiagnosticStore(db_path)
    store.write_iteration(
        iteration=0,
        run_idx=0,
        iteration_results=[
            {
                "function_name": f"{label}_model_a",
                "metric_name": "BIC",
                "metric_value": 100.0,
                "mean_nll": 2.5,
                "param_names": ["alpha"],
                "val_metric_value": 105.0,
                "val_mean_nll": 2.7,
            },
        ],
    )
    store.close()


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def test_compare_discovers_direct_and_nested_result_dirs(tmp_path: Path):
    """Discovery finds DuckDB files at both direct and nested locations."""
    from gecco.results_comparison import discover_run_dirs

    # Direct layout: results/mymodel/diagnostics.duckdb
    direct = tmp_path / "results" / "mymodel"
    direct.mkdir(parents=True)
    _create_minimal_diagnostics(direct / "diagnostics.duckdb", label="direct")

    # Nested layout: results/group/judge_off/run-001/diagnostics.duckdb
    nested = tmp_path / "results" / "group" / "judge_off" / "run-001"
    nested.mkdir(parents=True)
    _create_minimal_diagnostics(nested / "diagnostics.duckdb", label="nested")

    # Nested layout with diagnostics_unified.duckdb
    unified = tmp_path / "results" / "other" / "run-002"
    unified.mkdir(parents=True)
    _create_minimal_diagnostics(unified / "diagnostics_unified.duckdb", label="unified")

    discovered = discover_run_dirs([tmp_path / "results"])
    # Should find 3 run dirs
    assert len(discovered) == 3, f"Expected 3, got {len(discovered)}: {discovered}"

    # Verify exact run_ids
    run_ids = {d["run_id"] for d in discovered}
    assert run_ids == {"mymodel", "run-001", "run-002"}, f"Unexpected run_ids: {run_ids}"

    # Verify exact config_labels
    config_labels = {d["config_label"] for d in discovered}
    assert config_labels == {"mymodel", "group/judge_off", "other"}, \
        f"Unexpected config_labels: {config_labels}"

    # No config or batch manifest required (Negative)
    for d in discovered:
        assert "config" not in d or d.get("config") is None
        assert "batch_id" not in d or d.get("batch_id") is None


def test_compare_writes_one_csv_row_per_run(tmp_path: Path):
    """Two discovered run directories produce exactly two CSV rows."""
    from gecco.results_comparison import (
        discover_run_dirs,
        summarise_run,
        write_results_csv,
    )

    # Create two separate run dirs
    run1 = tmp_path / "runs" / "group_a" / "run_001"
    run1.mkdir(parents=True)
    _create_minimal_diagnostics(run1 / "diagnostics.duckdb", label="run_a")

    run2 = tmp_path / "runs" / "group_b" / "run_002"
    run2.mkdir(parents=True)
    _create_minimal_diagnostics(run2 / "diagnostics.duckdb", label="run_b")

    discovered = discover_run_dirs([tmp_path / "runs"])
    assert len(discovered) == 2

    summaries = [summarise_run(d) for d in discovered]

    out_dir = tmp_path / "comparison"
    csv_path = write_results_csv(summaries, out_dir)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2, f"Expected 2 CSV rows, got {len(rows)}"


def test_compare_uses_best_split_metrics_and_id_results(tmp_path: Path):
    """Lower fit metric selects best train/val/test values; best test ID fields exported."""
    from gecco.diagnostic_store.store import DiagnosticStore
    from gecco.results_comparison import (
        discover_run_dirs,
        summarise_run,
        write_results_csv,
    )

    run = tmp_path / "single_run"
    run.mkdir(parents=True)
    _create_minimal_diagnostics(run / "diagnostics.duckdb", label="best")

    # Add a second test model row with a lower (better) metric to prove
    # that the lowest test metric is selected as best_test_metric.
    store = DiagnosticStore(run / "diagnostics.duckdb")
    store.write_top_model_test(
        {
            "model_name": "best_model_b",
            "val_nll": 3.0,
            "test_mean_BIC": 110.0,  # lower than model_a's 130.0
            "test_mean_NLL": 3.2,
            "test_individual_BIC": [110.0],
            "test_individual_NLL": [3.2],
            "test_individual_differences": {
                "mean_r2": 0.88,
                "max_r2": 0.92,
                "best_param": "beta",
                "per_param_r2": {"beta": 0.92},
                "per_param_detail": {
                    "beta": {
                        "r2": 0.92,
                        "slope": 0.8,
                        "intercept": 0.2,
                    }
                },
            },
        }
    )
    store.close()

    discovered = discover_run_dirs([tmp_path])
    assert len(discovered) >= 1

    summary = summarise_run(discovered[0])
    # model_b has lower metric_value (100 < 120), so it should be selected
    assert summary["best_train_metric"] == 100.0
    assert summary["best_val_metric"] == 105.0
    # model_b also has lower test metric (110 < 130) → selected as best
    assert summary["best_test_metric"] == 110.0
    assert summary["best_test_nll"] == 3.2
    assert summary["best_test_mean_r2"] == 0.88
    assert summary["best_test_max_r2"] == 0.92
    assert summary["best_test_param"] == "beta"
    assert summary["best_model_name"] == "best_model_b"
    assert summary["has_test_eval"] is True
    assert summary["has_individual_differences"] is True


def test_compare_cli_writes_csv_html_and_figures(tmp_path: Path):
    """CLI handler writes CSV, HTML, PNG, and PDF files."""
    from gecco.cli.compare_results import main as compare_main
    from types import SimpleNamespace

    run = tmp_path / "cli_test_runs" / "run_alpha"
    run.mkdir(parents=True)
    _create_minimal_diagnostics(run / "diagnostics.duckdb", label="cli")

    out_dir = tmp_path / "cli_output"

    args = SimpleNamespace(
        results_dirs=[tmp_path / "cli_test_runs"],
        output=str(out_dir),
    )
    compare_main(args)

    assert (out_dir / "results.csv").exists()
    assert (out_dir / "report.html").exists()
    assert (out_dir / "figures" / "model_fit_by_config.png").exists()
    assert (out_dir / "figures" / "model_fit_by_config.pdf").exists()
    assert (out_dir / "figures" / "individual_differences_by_config.png").exists()
    assert (out_dir / "figures" / "individual_differences_by_config.pdf").exists()
    assert (out_dir / "figures" / "fit_vs_prediction.png").exists()
    assert (out_dir / "figures" / "fit_vs_prediction.pdf").exists()


def test_compare_handles_missing_test_or_id_data(tmp_path: Path):
    """Train-only diagnostics DB does not crash CSV/report generation."""
    from gecco.results_comparison import (
        discover_run_dirs,
        summarise_run,
        write_results_csv,
        render_report_html,
        export_figures,
    )

    run = tmp_path / "train_only_run"
    run.mkdir(parents=True)
    _create_minimal_diagnostics_train_only(run / "diagnostics.duckdb", label="train_only")

    discovered = discover_run_dirs([tmp_path])
    assert len(discovered) >= 1

    summary = summarise_run(discovered[0])
    # Should not crash; test fields are absent/None
    assert summary["best_train_metric"] == 100.0
    assert summary["best_test_metric"] is None or summary["best_test_metric"] == ""
    assert summary["has_test_eval"] is False
    assert summary["has_individual_differences"] is False

    out_dir = tmp_path / "comparison_no_test"
    csv_path = write_results_csv([summary], out_dir)
    assert csv_path.exists()

    html_path = render_report_html([summary], out_dir)
    assert html_path.exists()

    # Export figures (should not crash with missing data)
    fig_dir = export_figures([summary], out_dir)
    assert fig_dir.exists()
    for suffix in ("png", "pdf"):
        assert (fig_dir / f"model_fit_by_config.{suffix}").exists()
