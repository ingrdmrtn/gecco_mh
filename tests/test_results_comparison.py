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
            "status": "ok",
            "code": f"def {label}_model_a(stimulus, action, reward, params):\\n    return 0.0",
            "param_names": ["alpha"],
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
            "status": "ok",
            "code": "def best_model_b(stimulus, action, reward, params):\n    return 0.0",
            "param_names": ["beta"],
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


# --------------------------------------------------------------------------- #
# Config-level aggregation
# --------------------------------------------------------------------------- #


def test_config_summary_aggregates_by_label(tmp_path: Path):
    """Two runs with the same config_label produce one config summary row."""
    from gecco.results_comparison import (
        aggregate_configs,
        discover_run_dirs,
        summarise_run,
    )

    # Two runs with identical config_label structure
    run1 = tmp_path / "runs" / "group_a" / "run_001"
    run1.mkdir(parents=True)
    _create_minimal_diagnostics(run1 / "diagnostics.duckdb", label="model")

    run2 = tmp_path / "runs" / "group_a" / "run_002"
    run2.mkdir(parents=True)
    _create_minimal_diagnostics(run2 / "diagnostics.duckdb", label="model")

    # One run with a different config_label
    run3 = tmp_path / "runs" / "group_b" / "run_003"
    run3.mkdir(parents=True)
    _create_minimal_diagnostics(run3 / "diagnostics.duckdb", label="other")

    discovered = discover_run_dirs([tmp_path / "runs"])
    assert len(discovered) == 3

    summaries = [summarise_run(d) for d in discovered]
    config_rows = aggregate_configs(summaries)

    assert len(config_rows) == 2, f"Expected 2 config rows, got {len(config_rows)}"

    labels = {r["config_label"] for r in config_rows}
    assert labels == {"group_a", "group_b"}


def test_config_summary_csv_contents(tmp_path: Path):
    """config_summary.csv has expected columns and correct aggregation values."""
    from gecco.results_comparison import (
        aggregate_configs,
        discover_run_dirs,
        summarise_run,
        write_config_summary_csv,
    )

    # Two runs with same config_label, both from _create_minimal_diagnostics
    # Each has: best_train_metric=100, best_test_metric=130
    run1 = tmp_path / "runs" / "cfg" / "run_001"
    run1.mkdir(parents=True)
    _create_minimal_diagnostics(run1 / "diagnostics.duckdb", label="a")

    run2 = tmp_path / "runs" / "cfg" / "run_002"
    run2.mkdir(parents=True)
    _create_minimal_diagnostics(run2 / "diagnostics.duckdb", label="b")

    discovered = discover_run_dirs([tmp_path / "runs"])
    summaries = [summarise_run(d) for d in discovered]
    config_rows = aggregate_configs(summaries)

    out_dir = tmp_path / "out"
    csv_path = write_config_summary_csv(config_rows, out_dir)
    assert csv_path.exists()
    assert csv_path.name == "config_summary.csv"

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert row["config_label"] == "cfg"
    assert row["n_runs"] == "2"
    assert row["n_with_test_eval"] == "2"
    assert row["n_with_individual_differences"] == "2"

    # Both runs have best_train_metric=100, best_test_metric=130
    assert float(row["best_train_metric_mean"]) == pytest.approx(100.0, abs=1e-4)
    assert float(row["best_train_metric_n"]) == 2
    # std of [100, 100] = 0
    assert float(row["best_train_metric_std"]) == pytest.approx(0.0, abs=1e-4)

    assert float(row["best_test_metric_mean"]) == pytest.approx(130.0, abs=1e-4)
    assert int(row["best_test_metric_n"]) == 2
    assert float(row["best_test_metric_std"]) == pytest.approx(0.0, abs=1e-4)


def test_config_summary_handles_missing_data(tmp_path: Path):
    """Aggregation ignores None values; std is None/0 when only 1 run."""
    from gecco.results_comparison import aggregate_configs

    # Manually create summaries with mixed missing data
    summaries = [
        {
            "config_label": "grp",
            "best_train_metric": 100.0,
            "best_val_metric": 105.0,
            "best_test_metric": None,
            "best_test_nll": None,
            "best_test_mean_r2": None,
            "best_test_max_r2": None,
            "n_models": 5,
            "n_failed_models": 1,
            "has_test_eval": False,
            "has_individual_differences": False,
        },
        {
            "config_label": "grp",
            "best_train_metric": 120.0,
            "best_val_metric": None,
            "best_test_metric": 130.0,
            "best_test_nll": 4.0,
            "best_test_mean_r2": 0.75,
            "best_test_max_r2": 0.85,
            "n_models": 8,
            "n_failed_models": 0,
            "has_test_eval": True,
            "has_individual_differences": True,
        },
    ]

    config_rows = aggregate_configs(summaries)
    assert len(config_rows) == 1
    row = config_rows[0]

    assert row["n_runs"] == 2
    assert row["n_with_test_eval"] == 1
    assert row["n_with_individual_differences"] == 1
    assert row["n_models_total"] == 13
    assert row["n_failed_models_total"] == 1

    # best_train_metric has 2 values
    assert row["best_train_metric_mean"] == 110.0
    assert row["best_train_metric_n"] == 2
    # best_val_metric has 1 value → std is 0.0
    assert row["best_val_metric_mean"] == 105.0
    assert row["best_val_metric_std"] == 0.0
    assert row["best_val_metric_n"] == 1
    # best_test_metric has 1 value
    assert row["best_test_metric_mean"] == 130.0
    assert row["best_test_metric_n"] == 1
    # best_test_nll has 1 value
    assert row["best_test_nll_n"] == 1
    # best_test_mean_r2 has 1 value
    assert row["best_test_mean_r2_n"] == 1


def test_compare_cli_writes_config_summary(tmp_path: Path):
    """CLI handler writes config_summary.csv alongside results.csv."""
    from gecco.cli.compare_results import main as compare_main
    from types import SimpleNamespace

    # Two runs sharing a config_label, one run in another config
    for i, cfg in enumerate(["cfg_a", "cfg_a", "cfg_b"]):
        run = tmp_path / "cli_runs" / cfg / f"run_{i:03d}"
        run.mkdir(parents=True)
        _create_minimal_diagnostics(run / "diagnostics.duckdb", label=f"m{i}")

    out_dir = tmp_path / "cli_output"

    args = SimpleNamespace(
        results_dirs=[tmp_path / "cli_runs"],
        output=str(out_dir),
    )
    compare_main(args)

    assert (out_dir / "results.csv").exists()
    assert (out_dir / "config_summary.csv").exists()

    # Verify config_summary.csv content
    with open(out_dir / "config_summary.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2, f"Expected 2 config rows, got {len(rows)}"
    labels = {r["config_label"] for r in rows}
    assert labels == {"cfg_a", "cfg_b"}

    for row in rows:
        if row["config_label"] == "cfg_a":
            assert row["n_runs"] == "2"
        elif row["config_label"] == "cfg_b":
            assert row["n_runs"] == "1"


def test_html_report_shows_config_and_run_tables(tmp_path: Path):
    """HTML report contains both a config-level summary table and run details."""
    from gecco.results_comparison import (
        aggregate_configs,
        discover_run_dirs,
        render_report_html,
        summarise_run,
    )

    # Two configs, one run each
    run1 = tmp_path / "runs" / "cfg_x" / "run_001"
    run1.mkdir(parents=True)
    _create_minimal_diagnostics(run1 / "diagnostics.duckdb", label="x")

    run2 = tmp_path / "runs" / "cfg_y" / "run_002"
    run2.mkdir(parents=True)
    _create_minimal_diagnostics(run2 / "diagnostics.duckdb", label="y")

    discovered = discover_run_dirs([tmp_path / "runs"])
    summaries = [summarise_run(d) for d in discovered]
    config_rows = aggregate_configs(summaries)

    out_dir = tmp_path / "html_out"
    html_path = render_report_html(summaries, out_dir, config_rows=config_rows)
    assert html_path.exists()

    html = html_path.read_text(encoding="utf-8")
    # Should contain both tables
    assert "Config Summary" in html
    assert "Run Details" in html or "Run Overview" in html
    # Should show config-level rows
    assert "cfg_x" in html
    assert "cfg_y" in html
    # Should show run-level rows
    assert "run_001" in html
    assert "run_002" in html


def test_export_figures_with_config_rows_uses_means(tmp_path: Path):
    """Figures exported with config_rows show config-level means with error bars."""
    from gecco.results_comparison import (
        aggregate_configs,
        discover_run_dirs,
        export_figures,
        summarise_run,
    )

    # Two runs in one config, one run in another
    run1 = tmp_path / "runs" / "cfg_a" / "run_001"
    run1.mkdir(parents=True)
    _create_minimal_diagnostics(run1 / "diagnostics.duckdb", label="a1")

    run2 = tmp_path / "runs" / "cfg_a" / "run_002"
    run2.mkdir(parents=True)
    _create_minimal_diagnostics(run2 / "diagnostics.duckdb", label="a2")

    run3 = tmp_path / "runs" / "cfg_b" / "run_003"
    run3.mkdir(parents=True)
    _create_minimal_diagnostics(run3 / "diagnostics.duckdb", label="b")

    discovered = discover_run_dirs([tmp_path / "runs"])
    summaries = [summarise_run(d) for d in discovered]
    config_rows = aggregate_configs(summaries)

    out_dir = tmp_path / "fig_out"
    fig_dir = export_figures(summaries, out_dir, config_rows=config_rows)
    assert fig_dir.exists()
    for fname in ("model_fit_by_config", "individual_differences_by_config", "fit_vs_prediction"):
        assert (fig_dir / f"{fname}.png").exists()
        assert (fig_dir / f"{fname}.pdf").exists()


def test_export_config_level_figures_uses_only_test_metrics_for_main_figure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The main config comparison figure is driven only by test metrics."""
    from gecco.results_comparison import _export_config_level_figures

    calls = []

    def fake_bar_chart_with_errors(
        base_path, labels, series, title, ylabel, caption
    ):
        calls.append(
            {
                "base_path": base_path,
                "labels": labels,
                "series": series,
                "title": title,
                "ylabel": ylabel,
                "caption": caption,
            }
        )

    monkeypatch.setattr(
        "gecco.results_comparison._bar_chart_with_errors",
        fake_bar_chart_with_errors,
    )

    config_rows = [
        {
            "config_label": "cfg_a",
            "best_train_metric_mean": 100.0,
            "best_train_metric_std": 5.0,
            "best_val_metric_mean": 110.0,
            "best_val_metric_std": 6.0,
            "best_test_metric_mean": 130.0,
            "best_test_metric_std": 7.0,
            "best_test_mean_r2_mean": 0.5,
            "best_test_mean_r2_std": 0.1,
            "best_test_max_r2_mean": 0.6,
            "best_test_max_r2_std": 0.2,
        },
        {
            "config_label": "cfg_b",
            "best_train_metric_mean": 200.0,
            "best_train_metric_std": 8.0,
            "best_val_metric_mean": 210.0,
            "best_val_metric_std": 9.0,
            "best_test_metric_mean": 230.0,
            "best_test_metric_std": 10.0,
            "best_test_mean_r2_mean": 0.7,
            "best_test_mean_r2_std": 0.3,
            "best_test_max_r2_mean": 0.8,
            "best_test_max_r2_std": 0.4,
        },
    ]

    _export_config_level_figures(tmp_path, config_rows)

    assert len(calls) == 2

    main_call = next(c for c in calls if c["base_path"].name == "model_fit_by_config")
    assert main_call["labels"] == ["cfg_a", "cfg_b"]
    assert [name for name, _, _ in main_call["series"]] == ["Test"]
    assert main_call["series"][0][1] == [130.0, 230.0]
    assert main_call["series"][0][2] == [7.0, 10.0]
    assert main_call["title"] == "Test Evaluation by Config (mean ± SD)"
    assert main_call["ylabel"] == "Metric Value"
    assert main_call["caption"] == "Best Test Metric (lower is better)"


# --------------------------------------------------------------------------- #
# Config-level figure data helpers
# --------------------------------------------------------------------------- #


def test_prepare_fit_vs_prediction_with_config_rows_uses_means():
    """_prepare_fit_vs_prediction_data returns config-level means when
    config_rows is provided, not run-level points."""
    from gecco.results_comparison import _prepare_fit_vs_prediction_data

    summaries = [
        {"config_label": "cfg_a", "run_id": "run_001",
         "best_train_metric": 100.0, "best_test_metric": 130.0},
        {"config_label": "cfg_a", "run_id": "run_002",
         "best_train_metric": 110.0, "best_test_metric": 140.0},
        {"config_label": "cfg_b", "run_id": "run_003",
         "best_train_metric": 200.0, "best_test_metric": 250.0},
    ]
    config_rows = [
        {"config_label": "cfg_a",
         "best_train_metric_mean": 105.0, "best_test_metric_mean": 135.0},
        {"config_label": "cfg_b",
         "best_train_metric_mean": 200.0, "best_test_metric_mean": 250.0},
    ]

    labels, x_vals, y_vals = _prepare_fit_vs_prediction_data(
        summaries, config_rows=config_rows
    )

    # Uses config-level means, not run-level points
    assert labels == ["cfg_a", "cfg_b"]
    assert x_vals == [105.0, 200.0]   # config-level means
    assert y_vals == [135.0, 250.0]   # config-level means


def test_prepare_fit_vs_prediction_without_config_rows_uses_run_level():
    """_prepare_fit_vs_prediction_data returns run-level values when
    config_rows is None."""
    from gecco.results_comparison import _prepare_fit_vs_prediction_data

    summaries = [
        {"config_label": "cfg_a", "run_id": "run_001",
         "best_train_metric": 100.0, "best_test_metric": 130.0},
        {"config_label": "cfg_b", "run_id": "run_002",
         "best_train_metric": 200.0, "best_test_metric": 250.0},
    ]

    labels, x_vals, y_vals = _prepare_fit_vs_prediction_data(summaries)

    assert labels == ["cfg_a", "cfg_b"]
    assert x_vals == [100.0, 200.0]   # run-level values
    assert y_vals == [130.0, 250.0]   # run-level values


def test_config_level_series_uses_nan_for_missing_values():
    """_config_level_series returns NaN, not zero, for missing metric values."""
    import math
    from gecco.results_comparison import _config_level_series

    config_rows = [
        {"config_label": "cfg_a",
         "best_train_metric_mean": 100.0, "best_train_metric_std": 5.0,
         "best_test_metric_mean": None, "best_test_metric_std": None},
        {"config_label": "cfg_b",
         "best_train_metric_mean": 200.0, "best_train_metric_std": 10.0,
         "best_test_metric_mean": 300.0, "best_test_metric_std": 15.0},
    ]

    # For best_train_metric: both present
    means, errors = _config_level_series(config_rows, "best_train_metric")
    assert means == [100.0, 200.0]
    assert errors == [5.0, 10.0]

    # For best_test_metric: cfg_a is missing → NaN, not 0.0
    means, errors = _config_level_series(config_rows, "best_test_metric")
    assert math.isnan(means[0]), f"Expected NaN for missing, got {means[0]}"
    assert means[1] == 300.0
    assert math.isnan(errors[0]), f"Expected NaN for missing std, got {errors[0]}"


# --------------------------------------------------------------------------- #
# Test-row exclusion logic
# --------------------------------------------------------------------------- #


def _create_invalid_test_row(db_path: Path, *, model_name: str = "invalid_model",
                             low_bic: float = 80.0) -> None:
    """Write an invalid test model row (status != 'ok') to the diagnostics DB."""
    store = DiagnosticStore(db_path)
    store.write_top_model_test({
        "model_name": model_name,
        "val_nll": 2.0,
        "test_mean_BIC": low_bic,
        "test_mean_NLL": 2.0,
        "test_individual_BIC": [low_bic],
        "test_individual_NLL": [2.0],
        "test_individual_differences": None,
        "code": "def invalid_model(s, a, r, p):\n    return 0.0",
        "param_names": ["alpha"],
        "status": "choice_leakage",
        "error_type": "InvalidLikelihoodError",
        "error_message": "Choice leakage detected",
        "error_details": {"reason": "choice_leakage"},
    })
    store.close()


def _create_legacy_test_row(db_path: Path, *, model_name: str = "legacy_model",
                            low_bic: float = 90.0) -> None:
    """Write a legacy summary-only test row (status='ok' but code IS NULL)."""
    store = DiagnosticStore(db_path)
    store.write_top_model_test({
        "model_name": model_name,
        "val_nll": 2.5,
        "test_mean_BIC": low_bic,
        "test_mean_NLL": 2.5,
        "test_individual_BIC": [low_bic],
        "test_individual_NLL": [2.5],
        "test_individual_differences": None,
        "code": None,  # legacy – no model code
        "param_names": [],
        "status": "ok",
    })
    store.close()


def test_invalid_test_row_excluded_in_favor_of_valid(tmp_path: Path):
    """An invalid test row (status != 'ok') with a low BIC is excluded;
    the valid row with a higher BIC is selected as best_test_metric."""
    from gecco.results_comparison import discover_run_dirs, summarise_run

    run = tmp_path / "invalid_vs_valid"
    run.mkdir(parents=True)
    # Create valid test row with higher BIC
    _create_minimal_diagnostics(run / "diagnostics.duckdb", label="valid")
    # Add invalid row with a lower (better-looking) BIC
    _create_invalid_test_row(run / "diagnostics.duckdb",
                             model_name="invalid_better", low_bic=80.0)

    discovered = discover_run_dirs([tmp_path])
    summary = summarise_run(discovered[0])

    # Invalid row is excluded; valid row's metrics are used
    assert summary["best_test_metric"] == 130.0  # valid_model_a's test BIC
    # best_model_name comes from train split (lowest train BIC = model_b)
    assert summary["best_model_name"] == "valid_model_b"
    assert summary["n_excluded_rows"] >= 1


def test_code_null_test_row_is_selected_when_lower_bic(tmp_path: Path):
    """A successful test row with code=NULL and a lower BIC is selected,
    not excluded.  Only status != 'ok' rows are excluded."""
    from gecco.results_comparison import discover_run_dirs, summarise_run

    run = tmp_path / "code_null_selected"
    run.mkdir(parents=True)
    # Create valid test row with higher BIC
    _create_minimal_diagnostics(run / "diagnostics.duckdb", label="modern")
    # Add code-NULL row with lower (better-looking) BIC
    _create_legacy_test_row(run / "diagnostics.duckdb",
                            model_name="code_null_better", low_bic=90.0)

    discovered = discover_run_dirs([tmp_path])
    summary = summarise_run(discovered[0])

    # code-NULL row has lower BIC (90 < 130) so it is selected
    assert summary["best_test_metric"] == 90.0
    # best_model_name still comes from train split (lowest train BIC = model_b)
    assert summary["best_model_name"] == "modern_model_b"
    # No rows are excluded (both have status='ok')
    assert summary["n_excluded_rows"] == 0


def test_exclusion_warnings_available(tmp_path: Path):
    """summarise_run returns exclusion_warnings and n_excluded_rows."""
    from gecco.results_comparison import discover_run_dirs, summarise_run

    run = tmp_path / "excl_warnings"
    run.mkdir(parents=True)
    _create_minimal_diagnostics(run / "diagnostics.duckdb", label="m")
    _create_invalid_test_row(run / "diagnostics.duckdb",
                             model_name="bad", low_bic=80.0)

    discovered = discover_run_dirs([tmp_path])
    summary = summarise_run(discovered[0])

    assert summary["n_excluded_rows"] >= 1
    assert isinstance(summary["exclusion_warnings"], list)
    assert len(summary["exclusion_warnings"]) >= 1
    # Should mention the invalid row
    combined = " ".join(summary["exclusion_warnings"]).lower()
    assert "invalid" in combined or "excluded" in combined


def _create_test_only_diagnostics(db_path: Path, *,
                                  label: str = "test_only",
                                  test_bic: float = 100.0,
                                  with_id: bool = True) -> None:
    """Create a diagnostics DuckDB with only test rows (no train/val)."""
    store = DiagnosticStore(db_path)
    id_data = None
    if with_id:
        id_data = {
            "mean_r2": 0.5,
            "max_r2": 0.6,
            "best_param": "lambda",
            "per_param_r2": {"lambda": 0.6},
            "per_param_detail": {
                "lambda": {"r2": 0.6, "slope": 0.7, "intercept": 0.3},
            },
        }
    store.write_top_model_test({
        "model_name": f"{label}_test_model",
        "val_nll": 2.0,
        "test_mean_BIC": test_bic,
        "test_mean_NLL": 2.5,
        "test_individual_BIC": [test_bic],
        "test_individual_NLL": [2.5],
        "test_individual_differences": id_data,
        "code": None,
        "param_names": ["lambda"],
        "status": "ok",
    })
    store.close()


def test_code_null_test_row_with_id_joins_correctly(tmp_path: Path):
    """A code-NULL successful test row with individual_differences
    correctly populates ID summary fields."""
    from gecco.results_comparison import discover_run_dirs, summarise_run

    run = tmp_path / "code_null_id"
    run.mkdir(parents=True)
    # Create diagnostics with a code-NULL test row that HAS individual differences
    _create_test_only_diagnostics(run / "diagnostics.duckdb",
                                  label="code_null_id", test_bic=95.0, with_id=True)

    discovered = discover_run_dirs([tmp_path])
    summary = summarise_run(discovered[0])

    assert summary["best_test_metric"] == 95.0
    assert summary["best_test_nll"] == 2.5
    assert summary["best_test_mean_r2"] == 0.5
    assert summary["best_test_max_r2"] == 0.6
    assert summary["best_test_param"] == "lambda"
    assert summary["has_test_eval"] is True
    assert summary["has_individual_differences"] is True


# --------------------------------------------------------------------------- #
# Split-store fallback (unified primary + sibling diagnostics)
# --------------------------------------------------------------------------- #


def test_split_store_fallback_uses_sibling_test_rows(tmp_path: Path):
    """When diagnostics_unified.duckdb has train/val only and a sibling
    diagnostics.duckdb has test/ID rows, the summary combines train/val
    from unified and test/ID from sibling."""
    from gecco.results_comparison import discover_run_dirs, summarise_run

    run = tmp_path / "split_fallback"
    run.mkdir(parents=True)

    # Create primary unified DB with train/val only
    _create_minimal_diagnostics_train_only(
        run / "diagnostics_unified.duckdb", label="unified"
    )

    # Create sibling diagnostics.duckdb with test/ID rows
    _create_test_only_diagnostics(
        run / "diagnostics.duckdb", label="sibling", test_bic=88.0, with_id=True
    )

    discovered = discover_run_dirs([tmp_path])
    # Discovery should pick up diagnostics_unified.duckdb (it's listed first)
    entry = discovered[0]
    assert "diagnostics_unified" in entry["db_path"]

    summary = summarise_run(entry)

    # Train/val come from the unified primary
    assert summary["best_train_metric"] == 100.0
    # Test/ID come from the sibling fallback
    assert summary["best_test_metric"] == 88.0
    assert summary["best_test_nll"] == 2.5
    assert summary["best_test_mean_r2"] == 0.5
    assert summary["best_test_max_r2"] == 0.6
    assert summary["best_test_param"] == "lambda"
    assert summary["has_test_eval"] is True
    assert summary["has_individual_differences"] is True


def test_split_store_primary_wins_when_it_has_test_rows(tmp_path: Path):
    """When diagnostics_unified.duckdb already has valid test rows,
    sibling diagnostics.duckdb test rows are NOT used."""
    from gecco.results_comparison import discover_run_dirs, summarise_run

    run = tmp_path / "split_primary_wins"
    run.mkdir(parents=True)

    # Create primary unified DB with train/val/test rows (test BIC = 130)
    _create_minimal_diagnostics(run / "diagnostics_unified.duckdb", label="unified")

    # Create sibling diagnostics.duckdb with LOWER test BIC (should be ignored)
    _create_test_only_diagnostics(
        run / "diagnostics.duckdb", label="sibling", test_bic=20.0, with_id=True
    )

    discovered = discover_run_dirs([tmp_path])
    entry = discovered[0]

    summary = summarise_run(entry)

    # Primary's test row (BIC=130) wins over sibling (BIC=20)
    assert summary["best_test_metric"] == 130.0
    # ID data comes from primary
    assert summary["best_test_mean_r2"] == 0.75
    assert summary["has_test_eval"] is True


def test_excluded_rows_do_not_feed_figure_data(tmp_path: Path):
    """Excluded (status != 'ok') rows do not affect summary metrics
    used by figure/report generation."""
    from gecco.results_comparison import discover_run_dirs, summarise_run

    run = tmp_path / "excl_figures"
    run.mkdir(parents=True)
    # Create a valid test row (BIC=130)
    _create_minimal_diagnostics(run / "diagnostics.duckdb", label="good")

    discovered = discover_run_dirs([tmp_path])
    summary_before = summarise_run(discovered[0])

    # Now add an invalid row with a very low BIC (should be excluded)
    _create_invalid_test_row(run / "diagnostics.duckdb",
                             model_name="bad_invalid", low_bic=10.0)

    # Re-read; summarise_run must recompute
    discovered_after = discover_run_dirs([tmp_path])
    summary_after = summarise_run(discovered_after[0])

    # best_test_metric must be unchanged (invalid rows excluded)
    assert summary_after["best_test_metric"] == summary_before["best_test_metric"]
    assert summary_after["best_test_metric"] == 130.0
    assert summary_after["best_model_name"] == summary_before["best_model_name"]
    # n_models may differ because rows were added; but n_excluded_rows > 0
    assert summary_after["n_excluded_rows"] == 1
