"""Tests for centralized model generation registry methods."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from gecco.coordination import SharedRegistry


@pytest.fixture
def registry():
    """Create a temporary registry for testing."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        registry_path = f.name
    reg = SharedRegistry(registry_path)
    yield reg
    os.unlink(registry_path)


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


def test_existing_registry_without_cmg_keys():
    """Verify that old registry files without CMG keys don't crash."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"global_best": None, "baseline": None, "tried_param_sets": [], "client_entries": {}, "iteration_history": []}, f)
        registry_path = f.name
    try:
        reg = SharedRegistry(registry_path)
        # These should not crash
        assert reg.get_candidate_models(0) is None
        reg.set_generator_status(0, "gen", "complete", n_candidates=1)
        reg.set_candidate_models(0, [{"index": 0, "func_name": "m1", "code": ""}], "gen")
        assert reg.get_candidate_models(0) is not None
    finally:
        os.unlink(registry_path)


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
