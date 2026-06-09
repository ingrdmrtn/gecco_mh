"""Phase 8 contract tests for DuckDB-backed reporting."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from gecco.diagnostic_store.store import DiagnosticStore
from gecco.reporting import load_report_summary, render_report_html, render_report_text


def _build_results_store(results_dir: Path) -> Path:
    """Create a minimal DuckDB-backed results store for reporting tests."""
    results_dir.mkdir(parents=True, exist_ok=True)
    db_path = results_dir / "diagnostics.duckdb"
    store = DiagnosticStore(db_path)
    store.write_iteration(
        iteration=0,
        run_idx=0,
        iteration_results=[
            {
                "function_name": "model_a",
                "metric_name": "BIC",
                "metric_value": 110.0,
                "param_names": ["alpha"],
            }
        ],
    )
    store.write_iteration(
        iteration=1,
        run_idx=0,
        iteration_results=[
            {
                "function_name": "model_b",
                "metric_name": "BIC",
                "metric_value": 98.5,
                "param_names": ["beta"],
            }
        ],
    )
    store.close()
    return db_path


def test_report_summary_reads_duckdb_from_custom_results_dir(tmp_path: Path):
    """The summary reader should load an explicit non-default results directory."""
    results_dir = tmp_path / "custom_results"
    _build_results_store(results_dir)

    summary = load_report_summary(results_dir=results_dir)

    assert summary["result_name"] == "custom_results"
    assert summary["iteration_count"] == 2
    assert summary["metric_trajectory"] == [
        {"iteration": 0, "best_metric": 110.0},
        {"iteration": 1, "best_metric": 98.5},
    ]
    assert summary["best_overall"]["function_name"] == "model_b"


def test_report_summary_fails_when_only_json_artifacts_exist(tmp_path: Path):
    """JSON artefacts alone must not satisfy the report reader."""
    results_dir = tmp_path / "json_only"
    (results_dir / "bics").mkdir(parents=True)
    (results_dir / "bics" / "iter0_run0.json").write_text(
        '{"models": []}',
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="diagnostics.duckdb|DuckDB"):
        load_report_summary(results_dir=results_dir)


def test_report_renderer_uses_supplied_summary_without_rescanning(tmp_path: Path):
    """Renderers should consume the already-loaded summary only."""
    summary = {
        "result_name": "custom_results",
        "iteration_count": 2,
        "best_overall": {"function_name": "model_b", "metric_value": 98.5},
        "metric_trajectory": [
            {"iteration": 0, "best_metric": 110.0},
            {"iteration": 1, "best_metric": 98.5},
        ],
        "best_models": [],
        "results_dir": str(tmp_path / "custom_results"),
    }

    with patch("gecco.reporting.duckdb.connect", side_effect=AssertionError("renderer should not read DuckDB")):
        text = render_report_text(summary)
        html = render_report_html(summary)

    assert "custom_results" in text
    assert "Iteration count: 2" in text
    assert "model_b" in text
    assert "custom_results" in html
    assert "model_b" in html


def test_report_renderer_does_not_open_default_results_dir(tmp_path: Path):
    """The renderer should not open files or scan a default results root."""
    summary = {
        "result_name": "results",
        "iteration_count": 0,
        "best_overall": None,
        "metric_trajectory": [],
        "best_models": [],
        "results_dir": str(tmp_path / "results"),
    }

    with patch("gecco.reporting.Path.open", side_effect=AssertionError("renderer should not open files")):
        render_report_text(summary)
        render_report_html(summary)
