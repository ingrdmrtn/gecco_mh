"""Dashboard compatibility tests for DuckDB canonical state contracts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd
import pytest

from gecco.coordination import SharedRegistry
from gecco.diagnostic_store import DiagnosticStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_ROOT = PROJECT_ROOT / "gecco-mh-dashboard"

if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))


def _load_dashboard_module(module_name: str, relative_path: str):
    module_path = DASHBOARD_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dashboard_data_adapter = _load_dashboard_module(
    "compat_data_adapter", "dashboard/data_adapter.py"
)
dashboard_config = _load_dashboard_module(
    "compat_dashboard_config", "dashboard/config.py"
)


# --------------------------------------------------------------------------- #
# Contract: Registry state remains DuckDB-canonical
# --------------------------------------------------------------------------- #


def test_load_registry_snapshot_uses_shared_registry_duckdb_and_no_json_fallback(tmp_path: Path):
    """load_registry_snapshot must read the DuckDB registry snapshot directly."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "shared_registry.json").write_text("{\"legacy\": true}", encoding="utf-8")

    snapshot = {"registry": "duckdb"}
    with (
        patch.object(SharedRegistry, "open_existing") as open_existing,
        patch.object(dashboard_data_adapter.json, "load", side_effect=AssertionError("JSON fallback must not be used")),
        patch.object(dashboard_data_adapter.json, "loads", side_effect=AssertionError("JSON fallback must not be used")),
    ):
        open_existing.return_value.read.return_value = snapshot
        result = dashboard_data_adapter.load_registry_snapshot(results_dir)

    open_existing.assert_called_once_with(results_dir / "shared_registry.duckdb")
    open_existing.return_value.read.assert_called_once_with()
    assert result == snapshot


# --------------------------------------------------------------------------- #
# Contract: Terminal statuses match CLI semantics
# --------------------------------------------------------------------------- #


def test_summary_counts_complete_no_success_as_terminal():
    """complete_no_success must be counted as terminal complete,
    same as 'complete', and must not be counted as error."""
    data = {
        "client_entries": {
            "0": {"status": "running"},
            "1": {"status": "complete"},
            "2": {"status": "complete_no_success"},
        },
        "iteration_history": [],
        "tried_param_sets": [],
    }
    stats = dashboard_data_adapter.summary_stats(data)

    assert stats["complete"] == 2, "complete_no_success must count as complete"
    assert stats["running"] == 1
    assert stats["errors"] == 0
    assert stats["recovery_failed"] == 0


# --------------------------------------------------------------------------- #
# Contract: Task discovery finds current result layouts
# --------------------------------------------------------------------------- #


def test_available_tasks_detects_registry_and_diagnostics_duckdb(tmp_path: Path):
    """available_tasks must detect dirs containing shared_registry.duckdb,
    diagnostics.duckdb, or diagnostics_<client>.duckdb."""
    results_root = tmp_path / "results"
    (results_root / "task_a").mkdir(parents=True)
    (results_root / "task_a" / "shared_registry.duckdb").touch()

    (results_root / "task_b").mkdir(parents=True)
    (results_root / "task_b" / "diagnostics.duckdb").touch()

    (results_root / "task_c").mkdir(parents=True)
    (results_root / "task_c" / "diagnostics_client0.duckdb").touch()

    (results_root / "task_d").mkdir(parents=True)
    (results_root / "task_d" / "old_registry.json").touch()

    with patch.object(dashboard_config, "project_root", return_value=tmp_path):
        tasks = dashboard_config.available_tasks()

    assert "task_a" in tasks
    assert "task_b" in tasks
    assert "task_c" in tasks
    assert "task_d" not in tasks


def test_available_tasks_discovers_nested_result_dirs(tmp_path: Path):
    """available_tasks must return nested task names without flattening them."""
    results_root = tmp_path / "results"
    nested = results_root / "two_step_factors" / "deepseekv4flash" / "judge_off"
    nested.mkdir(parents=True)
    (nested / "shared_registry.duckdb").touch()

    with patch.object(dashboard_config, "project_root", return_value=tmp_path):
        tasks = dashboard_config.available_tasks()

    assert tasks == ["two_step_factors/deepseekv4flash/judge_off"]


# --------------------------------------------------------------------------- #
# Contract: Diagnostics are read from diagnostics*.duckdb
# --------------------------------------------------------------------------- #


def _create_diagnostics_db(db_path: Path, models: list[dict]) -> DiagnosticStore:
    """Populate a diagnostics.duckdb with minimal iteration results."""
    store = DiagnosticStore(db_path)
    store.write_iteration(
        iteration=0,
        run_idx=0,
        iteration_results=models,
    )
    store.close()
    return store


def test_diagnostics_adapter_reads_split_aware_model_rows(tmp_path: Path):
    """Diagnostics adapter must return split-aware rows (train/val/test)."""
    db_path = tmp_path / "diagnostics.duckdb"
    _create_diagnostics_db(
        db_path,
        [
            {
                "function_name": "model_a",
                "metric_name": "BIC",
                "metric_value": 10.0,
                "param_names": ["alpha"],
                "code": "def model_a(): pass",
                "val_metric_value": 12.0,
            },
            {
                "function_name": "model_b",
                "metric_name": "BIC",
                "metric_value": 15.0,
                "param_names": ["beta"],
                "code": "def model_b(): pass",
            },
        ],
    )

    result = dashboard_data_adapter.load_diagnostics_summary(tmp_path)

    assert result is not None
    assert not result.empty
    assert set(result["db_path"].unique()) == {str(db_path)}
    assert set(result["source_db"].unique()) == {db_path.name}

    model_a_rows = result[result["name"] == "model_a"]
    assert len(model_a_rows) == 2
    assert set(model_a_rows["split"].values) == {"train", "val"}

    train_row = model_a_rows[model_a_rows["split"] == "train"].iloc[0]
    assert train_row["model_id"] is not None
    assert train_row["dashboard_model_key"].startswith(f"{db_path.name}::")


def test_diagnostics_adapter_preserves_split_identity_for_individual_differences(tmp_path: Path):
    """individual_differences rows must be matched by model_id and split."""
    db_path = tmp_path / "diagnostics.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            "CREATE TABLE iterations ("
            "iteration_id INTEGER, run_idx INTEGER, client_id INTEGER, tag VARCHAR, "
            "timestamp VARCHAR, n_models_proposed INTEGER)"
        )
        conn.execute(
            "CREATE TABLE models ("
            "model_id INTEGER, iteration_id INTEGER, iteration INTEGER, name VARCHAR, code VARCHAR, "
            "metric_name VARCHAR, metric_value DOUBLE, mean_nll DOUBLE, split VARCHAR, param_names VARCHAR, status VARCHAR)"
        )
        conn.execute(
            "CREATE TABLE individual_differences ("
            "model_id INTEGER, mean_r2 DOUBLE, max_r2 DOUBLE, best_param VARCHAR, "
            "per_param_r2 VARCHAR, per_param_detail VARCHAR, split VARCHAR)"
        )
        conn.execute("INSERT INTO iterations VALUES (1, 0, 0, 'tag', '2025-01-01T00:00:00', 1)")
        conn.execute(
            "INSERT INTO models VALUES (11, 1, 0, 'shared_model', 'def shared_model(): return 1', 'BIC', 10.0, 1.0, 'train', '[\"alpha\"]', 'complete')"
        )
        conn.execute(
            "INSERT INTO individual_differences VALUES (11, 0.91, 0.97, 'wrong', '{\"alpha\": 0.97}', '{\"alpha\": \"wrong\"}', 'test')"
        )
        conn.execute(
            "INSERT INTO individual_differences VALUES (11, 0.82, 0.88, 'correct', '{\"alpha\": 0.88}', '{\"alpha\": \"correct\"}', 'train')"
        )

    summary = dashboard_data_adapter.load_diagnostics_summary(tmp_path)
    assert summary is not None
    assert len(summary) == 1

    row = summary.iloc[0]
    assert row["split"] == "train"
    assert row["detail"]["individual_differences"]["split"] == "train"
    assert row["detail"]["individual_differences"]["best_param"] == "correct"
    assert row["individual_differences_split"] == "train"

    detail = dashboard_data_adapter.load_model_detail(tmp_path, 11, split="train")
    assert detail is not None
    assert detail["split"] == "train"
    assert detail["individual_differences"]["split"] == "train"
    assert detail["individual_differences"]["best_param"] == "correct"


def test_diagnostics_adapter_ignores_json_only_results(tmp_path: Path):
    """Directories with only JSON artefacts must report diagnostics as unavailable."""
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "iter0_run0.json").write_text("[]")

    result = dashboard_data_adapter.load_diagnostics_summary(tmp_path)

    assert result is None or result.empty


def test_diagnostics_adapter_marks_empty_selection_as_available(tmp_path: Path):
    """An existing diagnostics.duckdb with no matching iteration rows is available."""
    db_path = tmp_path / "diagnostics.duckdb"
    _create_diagnostics_db(
        db_path,
        [
            {
                "function_name": "model_a",
                "metric_name": "BIC",
                "metric_value": 10.0,
                "param_names": ["alpha"],
                "code": "def model_a(): pass",
            }
        ],
    )

    state = dashboard_data_adapter.load_diagnostics_model_rows_state(tmp_path, iteration=1)

    assert state.available is True
    assert state.rows == []


# --------------------------------------------------------------------------- #
# Contract: Model identity is stable
# --------------------------------------------------------------------------- #


def test_model_detail_uses_model_id_with_duplicate_names(tmp_path: Path):
    """Model detail must be selectable by model_id, not just by function_name."""
    db_path = tmp_path / "diagnostics.duckdb"
    _create_diagnostics_db(
        db_path,
        [
            {
                "function_name": "shared_model",
                "metric_name": "BIC",
                "metric_value": 10.0,
                "param_names": ["alpha"],
                "code": "def shared_model(): return 1",
            },
            {
                "function_name": "shared_model",
                "metric_name": "BIC",
                "metric_value": 20.0,
                "param_names": ["beta"],
                "code": "def shared_model(): return 2",
            },
        ],
    )

    summary = dashboard_data_adapter.load_diagnostics_summary(tmp_path)
    assert summary is not None and not summary.empty

    model_ids = summary["model_id"].unique()
    assert len(model_ids) == 2

    detail_0 = dashboard_data_adapter.load_model_detail(tmp_path, int(model_ids[0]))
    detail_1 = dashboard_data_adapter.load_model_detail(tmp_path, int(model_ids[1]))

    assert detail_0 is not None
    assert detail_1 is not None
    assert detail_0["model_id"] != detail_1["model_id"]
    assert detail_0["name"] == "shared_model"
    assert detail_1["name"] == "shared_model"


def test_diagnostics_adapter_uses_source_db_for_colliding_model_ids(tmp_path: Path):
    """Shard identity must be carried through summary rows and detail lookup."""
    shard0 = tmp_path / "diagnostics_client0.duckdb"
    shard1 = tmp_path / "diagnostics_client1.duckdb"
    _create_diagnostics_db(
        shard0,
        [
            {
                "function_name": "shared_model",
                "metric_name": "BIC",
                "metric_value": 10.0,
                "param_names": ["alpha"],
                "code": "def shared_model(): return 'client0'",
            }
        ],
    )
    _create_diagnostics_db(
        shard1,
        [
            {
                "function_name": "shared_model",
                "metric_name": "BIC",
                "metric_value": 20.0,
                "param_names": ["beta"],
                "code": "def shared_model(): return 'client1'",
            }
        ],
    )

    summary = dashboard_data_adapter.load_diagnostics_summary(tmp_path)
    assert summary is not None and len(summary) == 2
    assert set(summary["db_path"].unique()) == {str(shard0), str(shard1)}
    assert set(summary["source_db"].unique()) == {shard0.name, shard1.name}

    rows = dashboard_data_adapter.load_diagnostics_model_rows(tmp_path, iteration=0)
    assert len(rows) == 2
    assert {row["source_db"] for row in rows} == {shard0.name, shard1.name}
    assert {row["detail"]["code"] for row in rows} == {
        "def shared_model(): return 'client0'",
        "def shared_model(): return 'client1'",
    }

    shard0_row = summary[summary["db_path"] == str(shard0)].iloc[0]
    shard1_row = summary[summary["db_path"] == str(shard1)].iloc[0]

    detail_0 = dashboard_data_adapter.load_model_detail(
        tmp_path,
        int(shard0_row["model_id"]),
        db_path=shard0_row["db_path"],
        split=shard0_row["split"],
    )
    detail_1 = dashboard_data_adapter.load_model_detail(
        tmp_path,
        int(shard1_row["model_id"]),
        db_path=shard1_row["db_path"],
        split=shard1_row["split"],
    )

    assert detail_0 is not None and detail_1 is not None
    assert detail_0["source_db"] == shard0.name
    assert detail_1["source_db"] == shard1.name
    assert detail_0["code"] != detail_1["code"]


def test_feedback_and_raw_llm_artifact_reader_loads_real_payloads(tmp_path: Path):
    """Feedback/model/review artifact readers must return real file content."""
    (tmp_path / "feedback").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "reviews").mkdir()

    (tmp_path / "feedback" / "iter0_run0.txt").write_text("raw feedback text", encoding="utf-8")
    (tmp_path / "models" / "iter0_run0.txt").write_text("def model_a(): return 1", encoding="utf-8")
    (tmp_path / "reviews" / "iter0.json").write_text(
        '{"summary": "ok", "synthesized_feedback": "Use a simpler mechanism."}',
        encoding="utf-8",
    )

    rows = dashboard_data_adapter.load_feedback_artifacts(tmp_path)

    assert len(rows) == 3
    assert {row["kind"] for row in rows} == {"feedback", "model_code", "review"}
    feedback_row = next(row for row in rows if row["kind"] == "feedback")
    model_row = next(row for row in rows if row["kind"] == "model_code")
    review_row = next(row for row in rows if row["kind"] == "review")

    assert feedback_row["content"] == "raw feedback text"
    assert model_row["content"] == "def model_a(): return 1"
    assert review_row["payload"]["synthesized_feedback"] == "Use a simpler mechanism."
    assert feedback_row["path"] == "feedback/iter0_run0.txt"
    assert review_row["path"] == "reviews/iter0.json"

    empty_dir = tmp_path / "missing"
    assert dashboard_data_adapter.load_feedback_artifacts(empty_dir) == []


def test_judge_trace_artifact_reader_loads_real_payloads(tmp_path: Path):
    """Judge trace artifact reader must return parsed trace payloads."""
    (tmp_path / "judge").mkdir()
    (tmp_path / "judge" / "iter0_run0.json").write_text(
        '{"synthesized_feedback": {"default": "Improve recovery."}, "trace": [{"tool": "call"}], "full_trace": [{"stage": "analysis"}], "verdict": {"accepted": true}}',
        encoding="utf-8",
    )

    rows = dashboard_data_adapter.load_judge_trace_artifacts(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "judge_trace"
    assert row["trace"] == [{"tool": "call"}]
    assert row["full_trace"] == [{"stage": "analysis"}]
    assert row["synthesized_feedback"]["default"] == "Improve recovery."
    assert row["verdict"]["accepted"] is True
    assert row["path"] == "judge/iter0_run0.json"

    empty_dir = tmp_path / "missing-trace"
    assert dashboard_data_adapter.load_judge_trace_artifacts(empty_dir) == []


# --------------------------------------------------------------------------- #
# Contract: Judge registry state is first-class
# --------------------------------------------------------------------------- #


def test_judge_state_comes_from_registry_without_trace_json(tmp_path: Path):
    """Judge iterations from the registry must be visible without any judge/*.json."""
    data = {
        "judge_iterations": {
            "0": {
                "synthesized_feedback": {"default": "Improve recovery."},
                "verdict": {"accepted": True},
                "failed": False,
                "timestamp": "2025-01-01T00:00:00",
                "error": None,
            },
            "1": {
                "synthesized_feedback": None,
                "verdict": None,
                "failed": True,
                "timestamp": "2025-01-02T00:00:00",
                "error": "Judge process crashed",
            },
        },
        "iteration_history": [
            {"client_id": 0, "iteration": 0, "results": [{"function_name": "m0", "metric_value": 1.0}]},
            {"client_id": 0, "iteration": 1, "results": [{"function_name": "m1", "metric_value": 2.0}]},
        ],
        "client_entries": {},
        "tried_param_sets": [],
    }

    judge_rows = dashboard_data_adapter.normalize_judge_iterations(data)

    assert len(judge_rows) == 2

    row_0 = next(r for r in judge_rows if r["iteration"] == 0)
    assert row_0["failed"] is False
    assert row_0["has_feedback"] is True
    assert row_0["verdict"] is not None

    row_1 = next(r for r in judge_rows if r["iteration"] == 1)
    assert row_1["failed"] is True
    assert row_1["has_feedback"] is False
    assert row_1["error"] == "Judge process crashed"
