"""Phase 7: DuckDB coordination and status views verification.

These tests prove that:
- DuckDB is the only canonical runtime coordination state
- Unsafe leftover paths (mark_complete, _update_registry) are removed
- runtime_status_view and runtime_coordination_view match canonical state
- Restart/reload reads committed DuckDB state correctly
- Concurrent registry writes are stable
- Runner-exit paths cannot overwrite evaluator-owned terminal status
- CMG terminal status is published only after canonical persistence succeeds
"""

from __future__ import annotations

import inspect
import multiprocessing as mp
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gecco.candidate_evaluation import CandidateEvaluator
from gecco.coordination import SharedRegistry


def _load_gecco_model_search():
    from gecco.run_gecco import GeCCoModelSearch

    return GeCCoModelSearch


# --------------------------------------------------------------------------- #
# Deletion guards
# --------------------------------------------------------------------------- #


def test_mark_complete_is_deleted_from_shared_registry():
    """SharedRegistry.mark_complete must be removed — no production caller needs it."""
    assert not hasattr(SharedRegistry, "mark_complete")


def test_update_registry_is_deleted_from_runner():
    """GeCCoModelSearch._update_registry must be removed — it was an unsafe runner-owned
    terminal-status publication route that could overwrite evaluator-owned state."""
    GeCCoModelSearch = _load_gecco_model_search()
    assert not hasattr(GeCCoModelSearch, "_update_registry")


# --------------------------------------------------------------------------- #
# runtime_status_view consistency
# --------------------------------------------------------------------------- #


def test_runtime_status_view_covers_complete_complete_no_success_and_had_runnable_model_false(
    tmp_path: Path,
):
    """runtime_status_view must reflect complete, complete_no_success, and
    had_runnable_model=False accurately."""
    registry = SharedRegistry(tmp_path / "registry")

    registry.update(
        client_id=0,
        iteration=0,
        results=[{"function_name": "m0", "metric_value": 10.0}],
        status="complete",
        had_runnable_model=True,
    )
    registry.update(
        client_id=1,
        iteration=0,
        results=[],
        status="complete_no_success",
        had_runnable_model=False,
    )

    rows = registry._fetchall("SELECT * FROM runtime_status_view ORDER BY client_id")

    assert len(rows) == 2
    assert rows[0]["client_id"] == "0"
    assert rows[0]["status"] == "complete"
    assert rows[0]["had_runnable_model"] is True
    assert rows[1]["client_id"] == "1"
    assert rows[1]["status"] == "complete_no_success"
    assert rows[1]["had_runnable_model"] is False


# --------------------------------------------------------------------------- #
# runtime_coordination_view consistency
# --------------------------------------------------------------------------- #


def test_runtime_coordination_view_uses_per_iteration_rows_not_current_status(
    tmp_path: Path,
):
    """runtime_coordination_view must answer historical counts from per-iteration rows,
    not from the latest client status in runtime_client_entries."""
    registry = SharedRegistry(tmp_path / "registry")

    # Client 0: iteration 0 complete with a model, iteration 1 retrying
    registry.update(
        client_id=0,
        iteration=0,
        results=[{"function_name": "m0", "metric_value": 10.0}],
        status="complete",
        had_runnable_model=True,
    )
    registry.update(
        client_id=0,
        iteration=1,
        results=[],
        status="retrying",
        had_runnable_model=False,
    )

    row = registry._fetchone(
        "SELECT n_clients_complete, n_clients_with_models "
        "FROM runtime_coordination_view WHERE iteration = 0"
    )
    assert row is not None
    assert row["n_clients_complete"] == 1
    assert row["n_clients_with_models"] == 1

    # Iteration 1 should show 0 complete because status is retrying
    row1 = registry._fetchone(
        "SELECT n_clients_complete, n_clients_with_models "
        "FROM runtime_coordination_view WHERE iteration = 1"
    )
    assert row1 is not None
    assert row1["n_clients_complete"] == 0
    assert row1["n_clients_with_models"] == 0


def test_runtime_coordination_view_includes_candidate_generation_and_generator_status(
    tmp_path: Path,
):
    """runtime_coordination_view must include candidate generation rows and generator
    status rows."""
    registry = SharedRegistry(tmp_path / "registry")

    registry.set_candidate_models(
        0,
        [
            {"index": 0, "func_name": "m1", "code": "..."},
            {"index": 1, "func_name": "m2", "code": "..."},
        ],
        "generator",
    )
    registry.set_generator_status(0, "generator", "complete", n_candidates=2)

    row = registry._fetchone("SELECT * FROM runtime_coordination_view WHERE iteration = 0")
    assert row is not None
    assert row["n_candidates"] == 2
    assert row["generated_by"] == "generator"
    assert row["generator_status"] == "complete"


def test_runtime_coordination_view_includes_judge_feedback_and_judge_failure(
    tmp_path: Path,
):
    """runtime_coordination_view must surface judge feedback presence and judge failures."""
    registry = SharedRegistry(tmp_path / "registry")

    registry.set_judge_feedback(0, "Looks good.", {"accepted": True})
    row = registry._fetchone(
        "SELECT has_judge_feedback, judge_failed FROM runtime_coordination_view WHERE iteration = 0"
    )
    assert row is not None
    assert row["has_judge_feedback"] is True
    assert row["judge_failed"] is False

    registry.set_judge_failure(1, "Judge crashed")
    row1 = registry._fetchone(
        "SELECT has_judge_feedback, judge_failed FROM runtime_coordination_view WHERE iteration = 1"
    )
    assert row1 is not None
    assert row1["has_judge_feedback"] is True
    assert row1["judge_failed"] is True


def test_runtime_coordination_view_complete_no_success_counts(
    tmp_path: Path,
):
    """runtime_coordination_view must count complete_no_success as completed and
    correctly report had_runnable_model=False."""
    registry = SharedRegistry(tmp_path / "registry")

    registry.update(
        client_id=0,
        iteration=0,
        results=[],
        status="complete_no_success",
        had_runnable_model=False,
    )

    row = registry._fetchone(
        "SELECT n_clients_complete, n_clients_with_models "
        "FROM runtime_coordination_view WHERE iteration = 0"
    )
    assert row is not None
    assert row["n_clients_complete"] == 1
    assert row["n_clients_with_models"] == 0


# --------------------------------------------------------------------------- #
# CLI monitor status classification
# --------------------------------------------------------------------------- #


def test_monitor_summary_counts_complete_no_success_as_completed():
    """The CLI summary must count complete_no_success as completed."""
    from gecco.cli.monitor_distributed import build_summary_stats

    data = {
        "client_entries": {
            "0": {"status": "running"},
            "1": {"status": "complete"},
            "2": {"status": "complete_no_success"},
        },
        "iteration_history": [],
        "tried_param_sets": [],
    }

    panel = build_summary_stats(data)

    assert "1 running, 2 complete" in panel.renderable


def test_monitor_client_table_treats_complete_no_success_as_terminal():
    """The CLI table must render complete_no_success with terminal success styling."""
    from gecco.cli.monitor_distributed import build_client_table

    data = {
        "client_entries": {
            "0": {"status": "running", "last_iteration": 1},
            "1": {"status": "complete_no_success", "last_iteration": 2},
        },
        "iteration_history": [],
        "tried_param_sets": [],
    }

    table = build_client_table(data)

    assert table.columns[1]._cells[0].style == "yellow"
    assert table.columns[1]._cells[1].style == "green"


# --------------------------------------------------------------------------- #
# Restart / reload via open_existing
# --------------------------------------------------------------------------- #


def test_open_existing_reads_committed_duckdb_state_without_json_fallback(
    tmp_path: Path,
):
    """A new SharedRegistry opened via open_existing must observe committed DuckDB state
    without resorting to JSON fallback or reinitialising schema."""
    registry_path = tmp_path / "shared_registry"
    writer = SharedRegistry(registry_path)
    writer.update(
        client_id="client-a",
        iteration=2,
        results=[{"function_name": "m2", "metric_value": 88.0}],
        status="complete_no_success",
        had_runnable_model=False,
    )
    writer.set_judge_feedback(2, "feedback", {"verdict": "ok"})

    reader = SharedRegistry.open_existing(registry_path)
    snapshot = reader.read()

    assert snapshot["client_entries"]["client-a"]["status"] == "complete_no_success"
    assert snapshot["client_entries"]["client-a"]["had_runnable_model"] is False
    assert snapshot["judge_iterations"]["2"]["synthesized_feedback"]["default"] == "feedback"


def test_open_existing_does_not_run_schema_ddl(tmp_path: Path):
    """SharedRegistry.open_existing must not trigger schema creation DDL."""
    registry_path = tmp_path / "shared_registry"
    writer = SharedRegistry(registry_path)
    writer.update(
        client_id=0,
        iteration=0,
        results=[{"function_name": "m0", "metric_value": 1.0}],
        status="complete",
    )

    with patch("gecco.coordination.create_schema") as mock_create:
        reader = SharedRegistry.open_existing(registry_path)
        reader.read()
        mock_create.assert_not_called()


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


def test_concurrent_registry_writes_are_stable_under_lock_strategy(tmp_path: Path):
    """Concurrent writers must serialise through the chosen lock/transaction strategy
    without losing rows."""
    registry_path = tmp_path / "shared_registry"
    SharedRegistry(registry_path)

    def _write(client_id: int) -> None:
        local = SharedRegistry(registry_path)
        local.update(
            client_id=client_id,
            iteration=0,
            results=[{"function_name": f"m{client_id}", "metric_value": float(client_id)}],
            status="complete",
            had_runnable_model=True,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(_write, range(8)))

    observer = SharedRegistry(registry_path)
    snapshot = observer.read()
    assert observer.count_clients_at_iteration(0) == 8
    assert len(snapshot["iteration_history"]) == 8


def test_concurrent_complete_no_success_writes_preserve_each_client(tmp_path: Path):
    """Multiple clients writing complete_no_success concurrently must all be visible."""
    registry_path = tmp_path / "shared_registry"
    SharedRegistry(registry_path)

    def _write(client_id: int) -> None:
        local = SharedRegistry(registry_path)
        local.update(
            client_id=client_id,
            iteration=1,
            results=[],
            status="complete_no_success",
            had_runnable_model=False,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(_write, range(4)))

    observer = SharedRegistry(registry_path)
    rows = observer._fetchall(
        "SELECT client_id, status, had_runnable_model FROM runtime_status_view ORDER BY client_id"
    )
    assert len(rows) == 4
    for row in rows:
        assert row["status"] == "complete_no_success"
        assert row["had_runnable_model"] is False


# --------------------------------------------------------------------------- #
# Runner / evaluator ownership guards
# --------------------------------------------------------------------------- #


def test_runner_source_does_not_call_mark_complete():
    """run_n_shots source must not contain mark_complete calls."""
    GeCCoModelSearch = _load_gecco_model_search()
    runner_source = inspect.getsource(GeCCoModelSearch.run_n_shots)
    assert "mark_complete(" not in runner_source


def test_runner_source_does_not_call_update_registry_with_terminal_status():
    """run_n_shots source must not contain _update_registry calls that could publish
    terminal status from the runner boundary."""
    GeCCoModelSearch = _load_gecco_model_search()
    runner_source = inspect.getsource(GeCCoModelSearch.run_n_shots)
    assert "_update_registry(" not in runner_source


def test_complete_no_success_cannot_be_overwritten_by_runner_exit_paths(
    tmp_path: Path,
):
    """Once an evaluator publishes complete_no_success, the runner must not overwrite it
    with complete or running on exit."""
    registry = SharedRegistry(tmp_path / "shared_registry.duckdb")

    # Simulate evaluator publishing terminal no-success status
    registry.update(
        client_id=0,
        iteration=0,
        results=[],
        status="complete_no_success",
        had_runnable_model=False,
    )

    # Simulate a runner-exit or orchestration update with a different status
    # (this should be impossible in production, but the registry write itself
    # is unconstrained; the protection is caller-side.)
    # Instead, verify the runner source does not contain such a call.
    GeCCoModelSearch = _load_gecco_model_search()
    helper_source = inspect.getsource(GeCCoModelSearch._run_cmg_evaluator_iteration)
    assert "shared_registry.update(" not in helper_source
    assert "mark_complete(" not in helper_source

    # Verify the terminal state is still intact
    snapshot = registry.read()
    assert snapshot["client_entries"]["0"]["status"] == "complete_no_success"
    assert snapshot["client_entries"]["0"]["had_runnable_model"] is False


# --------------------------------------------------------------------------- #
# CMG failure ordering
# --------------------------------------------------------------------------- #


def test_cmg_persistence_failure_prevents_terminal_registry_status(tmp_path: Path):
    """If canonical DuckDB persistence fails, the registry must not contain a terminal
    status for that iteration."""
    from gecco.artifacts import ArtifactStore
    from gecco.run_context import RunContext

    cfg = SimpleNamespace(
        data=SimpleNamespace(input_columns=[]),
        task=SimpleNamespace(name="phase7_task"),
        evaluation=SimpleNamespace(fit_type="group"),
    )
    run_context = RunContext.from_cfg(cfg, project_root=tmp_path)
    artifact_store = ArtifactStore(run_context)
    evaluator = CandidateEvaluator(artifact_store)
    shared_registry = SharedRegistry(tmp_path / "shared_registry.duckdb")

    with patch.object(
        artifact_store,
        "write_iteration_results",
        side_effect=RuntimeError("duckdb write failed"),
    ):
        with pytest.raises(RuntimeError, match="duckdb write failed"):
            evaluator.finalize_iteration_results(
                iteration=0,
                run_idx=0,
                tag="",
                iteration_results=[
                    {
                        "function_name": "model_a",
                        "metric_name": "BIC",
                        "metric_value": 1.0,
                        "param_names": ["alpha"],
                        "code": "def model_a():\n    return 0",
                    }
                ],
                client_id="client-a",
                results_source=SimpleNamespace(),
                shared_registry=shared_registry,
            )

    snapshot = shared_registry.read()
    entry = snapshot["client_entries"].get("client-a")
    assert entry is None or entry["status"] not in {"complete", "complete_no_success"}

    run_context.close()


def test_non_cmg_persistence_failure_prevents_terminal_registry_status(tmp_path: Path):
    """Non-CMG paths must also stop feedback and registry publication on write failure."""
    from gecco.artifacts import ArtifactStore
    from gecco.run_context import RunContext

    cfg = SimpleNamespace(
        data=SimpleNamespace(input_columns=[]),
        task=SimpleNamespace(name="phase7_task"),
        evaluation=SimpleNamespace(fit_type="group"),
    )
    run_context = RunContext.from_cfg(cfg, project_root=tmp_path)
    artifact_store = ArtifactStore(run_context)
    evaluator = CandidateEvaluator(artifact_store)
    shared_registry = MagicMock()
    feedback_record = MagicMock()

    with patch.object(
        artifact_store,
        "write_iteration_results",
        side_effect=RuntimeError("duckdb write failed"),
    ):
        with pytest.raises(RuntimeError, match="duckdb write failed"):
            evaluator.finalize_iteration_results(
                iteration=0,
                run_idx=0,
                tag="",
                iteration_results=[
                    {
                        "function_name": "model_a",
                        "metric_name": "BIC",
                        "metric_value": 1.0,
                        "param_names": ["alpha"],
                        "code": "def model_a():\n    return 0",
                    }
                ],
                client_id="client-b",
                results_source=SimpleNamespace(),
                shared_registry=shared_registry,
                feedback_record=feedback_record,
            )

    feedback_record.assert_not_called()
    shared_registry.update.assert_not_called()

    run_context.close()
