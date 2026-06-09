"""Tests for DuckDB-backed centralized model generation registry methods."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gecco.diagnostic_store import DiagnosticStore
from gecco.coordination import SharedRegistry


@pytest.fixture
def registry(tmp_path):
    """Create a temporary registry for testing."""
    registry_path = tmp_path / "shared_registry.duckdb"
    reg = SharedRegistry(registry_path)
    yield reg


def test_empty_registry_has_cmg_keys():
    data = SharedRegistry._empty_registry()
    assert "candidate_generations" in data
    assert "generator_status" in data


def test_set_and_get_candidate_models(registry):
    candidates = [
        {"index": 0, "func_name": "cognitive_model1", "name": "model_a", "code": "..."},
        {"index": 1, "func_name": "cognitive_model2", "name": "model_b", "code": "..."},
    ]
    registry.set_candidate_models(0, candidates, "generator")
    result = registry.get_candidate_models(0)
    assert result is not None
    assert result["generated_by"] == "generator"
    assert len(result["candidates"]) == 2
    assert result["candidates"][0]["index"] == 0
    assert result["candidates"][1]["index"] == 1


def test_set_candidate_models_idempotent(registry):
    candidates = [{"index": 0, "func_name": "cognitive_model1", "name": "model_a", "code": "..."}]
    registry.set_candidate_models(0, candidates, "generator")
    # Second call should not overwrite
    registry.set_candidate_models(0, [{"index": 99}], "hacker")
    result = registry.get_candidate_models(0)
    assert result["candidates"][0]["index"] == 0
    assert result["generated_by"] == "generator"


def test_update_candidate_model(registry):
    candidates = [
        {"index": 0, "func_name": "cognitive_model1", "name": "original", "code": "original_code"},
        {"index": 1, "func_name": "cognitive_model2", "name": "unchanged", "code": "unchanged_code"},
    ]
    registry.set_candidate_models(0, candidates, "generator")

    # Update candidate 0 only
    registry.update_candidate_model(0, 0, {
        "index": 0, "func_name": "cognitive_model1", "name": "repaired", "code": "repaired_code"
    })

    result = registry.get_candidate_models(0)
    assert result["candidates"][0]["code"] == "repaired_code"
    assert result["candidates"][0]["name"] == "repaired"
    assert result["candidates"][1]["code"] == "unchanged_code"  # Unchanged


def test_update_candidate_model_missing_index(registry):
    candidates = [{"index": 0, "func_name": "cognitive_model1", "code": "ok"}]
    registry.set_candidate_models(0, candidates, "generator")
    with pytest.raises(ValueError, match="No candidate with index 5"):
        registry.update_candidate_model(0, 5, {})


def test_get_candidate_models_missing(registry):
    result = registry.get_candidate_models(99)
    assert result is None


def test_set_generator_status(registry):
    registry.set_generator_status(0, "generator", "complete", n_candidates=4)
    data = registry.read()
    status = data["generator_status"]["0"]
    assert status["client_id"] == "generator"
    assert status["status"] == "complete"
    assert status["n_candidates"] == 4


def test_wait_for_candidate_models_timeout(registry):
    result = registry.wait_for_candidate_models(42, timeout_seconds=0.5, poll_seconds=0.1)
    assert result is None


def test_registry_initialises_duckdb_state_without_legacy_payload(tmp_path):
    """Initialisation should create a DuckDB-backed runtime store."""
    registry_path = tmp_path / "shared_registry.duckdb"
    reg = SharedRegistry(registry_path)

    assert reg.db_path.exists()
    assert reg.read()["candidate_generations"] == {}


def test_set_generator_status_with_error(registry):
    """Generator failure should store error message."""
    registry.set_generator_status(0, "gen", "failed", n_candidates=0, error="Generation failed: no models")
    data = registry.read()
    status = data["generator_status"]["0"]
    assert status["status"] == "failed"
    assert status["error"] == "Generation failed: no models"
    assert status["n_candidates"] == 0


def test_set_generator_status_without_error(registry):
    """Generator success should not have an error key."""
    registry.set_generator_status(0, "gen", "complete", n_candidates=3)
    data = registry.read()
    status = data["generator_status"]["0"]
    assert "error" not in status


# --- get_max_iteration_for_client (Chunk 1) ---

def test_get_max_iteration_for_client_empty(registry):
    """Empty registry should return -1 for any client."""
    assert registry.get_max_iteration_for_client(0) == -1
    assert registry.get_max_iteration_for_client("generator") == -1


def test_get_max_iteration_for_client_specific(registry):
    """Should only count iterations belonging to the requested client."""
    registry.update(client_id=0, iteration=0, results=[{"function_name": "m1", "metric_value": 100.0}], status="complete")
    registry.update(client_id=1, iteration=0, results=[{"function_name": "m2", "metric_value": 200.0}], status="complete")
    assert registry.get_max_iteration_for_client(0) == 0
    assert registry.get_max_iteration_for_client(1) == 0


def test_get_max_iteration_for_client_mixed(registry):
    """Client 0 has completed iteration 0; client 1 has not."""
    registry.update(client_id=0, iteration=0, results=[{"function_name": "m1", "metric_value": 100.0}], status="complete")
    assert registry.get_max_iteration_for_client(0) == 0
    assert registry.get_max_iteration_for_client(1) == -1


def test_get_max_iteration_for_client_multiple_iterations(registry):
    """Multiple iterations for same client should return the highest."""
    registry.update(client_id=0, iteration=0, results=[{"function_name": "m1", "metric_value": 100.0}], status="complete")
    registry.update(client_id=0, iteration=1, results=[{"function_name": "m2", "metric_value": 200.0}], status="complete")
    registry.update(client_id=1, iteration=0, results=[{"function_name": "m3", "metric_value": 300.0}], status="complete")
    assert registry.get_max_iteration_for_client(0) == 1
    assert registry.get_max_iteration_for_client(1) == 0


def test_get_max_iteration_for_client_ignores_retrying(registry):
    """Retrying status should not count as completed for resume."""
    registry.update(client_id=0, iteration=0, results=[], status="retrying")
    assert registry.get_max_iteration_for_client(0) == -1


def test_get_max_iteration_for_client_counts_complete(registry):
    """Complete status should count as completed for resume."""
    registry.update(client_id=0, iteration=0, results=[{"function_name": "m1", "metric_value": 100.0}], status="complete")
    assert registry.get_max_iteration_for_client(0) == 0


def test_get_max_iteration_for_client_complete_vs_retrying(registry):
    """Client 0 complete, client 1 retrying — only complete counts."""
    registry.update(client_id=0, iteration=0, results=[{"function_name": "m1", "metric_value": 100.0}], status="complete")
    registry.update(client_id=1, iteration=0, results=[], status="retrying")
    assert registry.get_max_iteration_for_client(0) == 0
    assert registry.get_max_iteration_for_client(1) == -1


def test_get_max_iteration_for_client_retrying_with_prior_complete(registry):
    """Retrying a later iteration should not erase completed earlier iterations."""
    registry.update(client_id=0, iteration=0, results=[{"function_name": "m1", "metric_value": 100.0}], status="complete")
    registry.update(client_id=0, iteration=1, results=[], status="retrying")
    assert registry.get_max_iteration_for_client(0) == 0


def test_get_max_iteration_for_client_retrying_after_multiple_complete(registry):
    """Retrying iteration 2 after completing 0 and 1 should return 1."""
    registry.update(client_id=0, iteration=0, results=[{"function_name": "m0", "metric_value": 100.0}], status="complete")
    registry.update(client_id=0, iteration=1, results=[{"function_name": "m1", "metric_value": 200.0}], status="complete")
    registry.update(client_id=0, iteration=2, results=[], status="retrying")
    assert registry.get_max_iteration_for_client(0) == 1


# --- get_max_generator_iteration (Chunk 2) ---

def test_get_max_generator_iteration_complete_with_candidates(registry):
    """Generator status complete with matching candidates should return iteration."""
    registry.set_candidate_models(0, [{"index": 0, "func_name": "cognitive_model1", "code": "..."}], "generator")
    registry.set_generator_status(0, "generator", "complete", n_candidates=1)
    assert registry.get_max_generator_iteration("generator") == 0


def test_get_max_generator_iteration_failed_status(registry):
    """Generator status failed should not count for resume."""
    registry.set_candidate_models(0, [{"index": 0, "func_name": "cognitive_model1", "code": "..."}], "generator")
    registry.set_generator_status(0, "generator", "failed", n_candidates=0, error="oops")
    assert registry.get_max_generator_iteration("generator") == -1


def test_get_max_generator_iteration_missing_candidates(registry):
    """Generator status complete but missing candidate_generations should not count."""
    registry.set_generator_status(0, "generator", "complete", n_candidates=1)
    assert registry.get_max_generator_iteration("generator") == -1


def test_get_max_generator_iteration_multiple(registry):
    """Multiple generator iterations should return the highest completed one."""
    registry.set_candidate_models(0, [{"index": 0, "func_name": "m1", "code": "..."}], "generator")
    registry.set_generator_status(0, "generator", "complete", n_candidates=1)
    registry.set_candidate_models(1, [{"index": 0, "func_name": "m2", "code": "..."}], "generator")
    registry.set_generator_status(1, "generator", "complete", n_candidates=1)
    assert registry.get_max_generator_iteration("generator") == 1


def test_registry_round_trip_preserves_runtime_snapshot(registry):
    """Writers and readers should round-trip runtime state via DuckDB."""
    registry.update(
        client_id=0,
        iteration=0,
        results=[
            {
                "function_name": "m1",
                "metric_name": "BIC",
                "metric_value": 101.5,
                "param_names": ["alpha"],
                "code": "def m1(): pass",
                "individual_differences": {
                    "mean_r2": 0.2,
                    "max_r2": 0.3,
                    "best_param": "alpha",
                    "per_param_r2": {"alpha": 0.3},
                },
            }
        ],
        best_model="def m1(): pass",
        best_metric=101.5,
        param_names=["alpha"],
        tried_param_sets=[["alpha"]],
        status="complete",
        had_runnable_model=True,
    )
    registry.set_baseline(
        {
            "function_name": "baseline_model",
            "metric_name": "BIC",
            "metric_value": 150.0,
            "param_names": ["beta"],
            "code": "def baseline_model(): pass",
        }
    )

    snapshot = registry.read()

    assert snapshot["global_best"]["client_id"] == 0
    assert snapshot["global_best"]["metric_value"] == pytest.approx(101.5)
    assert snapshot["baseline"]["code"] == "def baseline_model(): pass"
    assert snapshot["tried_param_sets"] == [["alpha"]]
    assert snapshot["client_entries"]["0"]["status"] == "complete"
    assert snapshot["iteration_history"][0]["results"][0]["function_name"] == "m1"


def test_registry_supports_restart_and_reload(tmp_path):
    """A new client instance should observe previously committed DuckDB state."""
    registry_path = tmp_path / "shared_registry.duckdb"
    reg1 = SharedRegistry(registry_path)
    reg1.update(
        client_id="generator",
        iteration=2,
        results=[{"function_name": "m2", "metric_value": 88.0}],
        status="complete",
    )

    reg2 = SharedRegistry(registry_path)
    snapshot = reg2.read()

    assert reg2.get_max_iteration() == 2
    assert snapshot["iteration_history"][0]["client_id"] == "generator"
    assert snapshot["client_entries"]["generator"]["last_iteration"] == 2


def test_registry_concurrent_clients_share_single_canonical_store(tmp_path):
    """Concurrent clients should serialize writes through DuckDB-backed locking."""
    registry_path = tmp_path / "shared_registry.duckdb"

    def _write(client_id: int) -> None:
        local_registry = SharedRegistry(registry_path)
        local_registry.update(
            client_id=client_id,
            iteration=0,
            results=[{"function_name": f"m{client_id}", "metric_value": 100.0 + client_id}],
            status="complete",
            had_runnable_model=True,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(_write, range(4)))

    registry = SharedRegistry(registry_path)
    snapshot = registry.read()

    assert registry.count_clients_at_iteration(0) == 4
    assert len(snapshot["iteration_history"]) == 4
    assert sorted(snapshot["client_entries"].keys()) == ["0", "1", "2", "3"]


def test_runtime_views_match_registry_state(registry):
    """DuckDB status and coordination views should reflect runtime updates."""
    registry.update(
        client_id=0,
        iteration=1,
        results=[{"function_name": "m1", "metric_value": 10.0}],
        status="complete",
        had_runnable_model=True,
    )
    registry.set_candidate_models(1, [{"index": 0, "func_name": "m1", "code": "..."}], "generator")
    registry.set_generator_status(1, "generator", "complete", n_candidates=1)
    registry.set_judge_feedback(1, "Looks good.", {"accepted": True})

    store = DiagnosticStore(registry.db_path)
    status_rows = store.fetchall("SELECT * FROM runtime_status_view")
    coordination_rows = store.fetchall(
        "SELECT * FROM runtime_coordination_view WHERE iteration = 1"
    )
    store.close()

    assert status_rows[0]["client_id"] == "0"
    assert status_rows[0]["status"] == "complete"
    assert coordination_rows[0]["n_client_results"] == 1
    assert coordination_rows[0]["n_candidates"] == 1
    assert coordination_rows[0]["generator_status"] == "complete"
    assert coordination_rows[0]["has_judge_feedback"] is True
