"""Contract tests for Phase 6 service extractions."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gecco.artifacts import ArtifactStore
from gecco.candidate_evaluation import CandidateEvaluator
from gecco.candidate_generation import CandidateGenerator
from gecco.diagnostic_store.store import DiagnosticStore
from gecco.distributed_coordinator import DistributedCoordinator
from gecco.feedback_coordinator import FeedbackCoordinator
from gecco.run_context import RunContext


def test_run_context_resolves_paths_and_cleans_tempdir(tmp_path: Path):
    """RunContext should own run paths and temporary-directory lifecycle."""

    cfg = SimpleNamespace(
        task=SimpleNamespace(name="phase6_task"),
        evaluation=SimpleNamespace(fit_type="individual"),
    )

    context = RunContext.from_cfg(cfg, project_root=tmp_path, client_id=3)
    try:
        assert context.project_root == tmp_path
        assert context.results_dir == tmp_path / "results" / "phase6_task_individual"
        assert (context.results_dir / "models").exists()
        assert (context.results_dir / "bics").exists()
        assert (context.results_dir / "feedback").exists()
        assert context.diagnostics_path == context.results_dir / "diagnostics_3.duckdb"
        assert context.tempdir_path.exists()
        assert context.tempdir_path.parent == tmp_path / "tmp"
    finally:
        tempdir_path = context.tempdir_path
        context.close()

    assert not tempdir_path.exists()


def test_run_context_rejects_missing_required_config(tmp_path: Path):
    """Invalid config should fail before any filesystem ownership is established."""

    cfg = SimpleNamespace(task=SimpleNamespace(name=""), evaluation=SimpleNamespace(fit_type="group"))

    with pytest.raises(ValueError, match="cfg.task.name"):
        RunContext.from_cfg(cfg, project_root=tmp_path)


def test_artifact_store_iteration_write_round_trip(tmp_path: Path):
    """Artefact writes should round-trip through DuckDB as the canonical store."""

    store = DiagnosticStore(tmp_path / "diagnostics.duckdb")
    artifact_store = ArtifactStore(tmp_path, store)

    iteration_results = [
        {
            "function_name": "model_a",
            "metric_name": "BIC",
            "metric_value": 12.5,
            "param_names": ["alpha"],
            "code": "def model_a():\n    return 0",
        }
    ]

    had_runnable_model = artifact_store.write_iteration_results(
        iteration=0,
        run_idx=1,
        tag="",
        iteration_results=iteration_results,
        client_id="client-a",
    )

    assert had_runnable_model is True
    assert (tmp_path / "bics" / "iter0_run1.json").exists()
    assert store.fetchone("SELECT COUNT(*) AS n FROM models") == {"n": 1}

    # A repeated write should remain idempotent at the DuckDB layer.
    artifact_store.write_iteration_results(
        iteration=0,
        run_idx=1,
        tag="",
        iteration_results=iteration_results,
        client_id="client-a",
    )
    assert store.fetchone("SELECT COUNT(*) AS n FROM models") == {"n": 1}
    store.close()


def test_candidate_generator_contract_uses_generation_backend_and_registry(tmp_path: Path):
    """The generator service should persist artefacts and publish candidates."""

    search = MagicMock()
    search.cfg = SimpleNamespace(
        clients={
            "client-a": SimpleNamespace(
                naive_ideation=SimpleNamespace(enabled=False)
            )
        }
    )
    search.client_id = "client-a"
    search.df = SimpleNamespace(participant=["p1"])
    search.results_dir = tmp_path
    search._file_tag.return_value = ""
    search._set_activity = MagicMock()
    search.prompt_builder.build_input_prompt.return_value = "prompt text"
    search.generate_models.return_value = (
        "def model_a():\n    return 0",
        [
            {"name": "model_a", "code": "code_a", "parameters": ["alpha"]},
            {"name": "model_b", "code": "code_b", "parameters": ["beta"]},
        ],
    )
    search.generate_models_naive = MagicMock()
    search.shared_registry = MagicMock()

    generator = CandidateGenerator(ArtifactStore(tmp_path))
    result = generator.generate_iteration(
        search=search,
        iteration=0,
        run_idx=2,
        feedback="use simpler models",
        cmg_cfg=SimpleNamespace(n_models=2),
    )

    search.generate_models.assert_called_once_with("prompt text", n_models=2)
    search.shared_registry.set_candidate_models.assert_called_once()
    search.shared_registry.set_generator_status.assert_called_once_with(
        iteration=0,
        client_id="client-a",
        status="complete",
        n_candidates=2,
    )
    assert result.candidates[0]["func_name"] == "cognitive_model1"
    assert result.model_file.exists()


def test_candidate_evaluator_contract_finalises_iteration_results(tmp_path: Path):
    """The evaluator service should fit the assigned candidate and finalise it."""

    search = MagicMock()
    search.client_id = 0
    search.df = SimpleNamespace(participant=["p1"])
    search.results_dir = tmp_path
    search.cfg = SimpleNamespace(
        judge=SimpleNamespace(barrier=SimpleNamespace(client_wait_seconds=1)),
        validation=SimpleNamespace(max_syntax_retries=0),
    )
    search._file_tag.return_value = ""
    search._cmg_evaluator_index.return_value = 0
    search._is_cmg_repairable_error.return_value = False
    search._fit_candidate_model.return_value = (
        {"function_name": "model_a", "metric_name": "BIC", "metric_value": 10.0},
        False,
    )
    search._finalize_iteration_results = MagicMock()
    search._update_registry = MagicMock()
    search.shared_registry = MagicMock()
    search.shared_registry.wait_for_candidate_models.return_value = {
        "candidates": [
            {
                "index": 0,
                "func_name": "cognitive_model1",
                "name": "model_a",
                "code": "def model_a():\n    return 0",
                "parameters": [],
            }
        ]
    }

    evaluator = CandidateEvaluator(ArtifactStore(tmp_path))
    result = evaluator.evaluate_iteration(
        search=search,
        iteration=0,
        run_idx=3,
        feedback="feedback",
        cmg_cfg=SimpleNamespace(n_models=1),
        baseline_bic=99.0,
    )

    search._fit_candidate_model.assert_called_once()
    search._finalize_iteration_results.assert_called_once()
    assert result.iteration_results[0]["metric_value"] == 10.0
    assert result.model_file.exists()


def test_feedback_coordinator_uses_orchestrated_pipeline_for_persona_feedback(tmp_path: Path):
    """The feedback coordinator should adapt orchestrated feedback for the active persona."""

    search = SimpleNamespace(
        tool_judge=MagicMock(),
        cfg=SimpleNamespace(),
        results_dir=tmp_path,
        client_id="client-a",
        best_model=None,
        best_metric=None,
        shared_registry=None,
        _set_activity=MagicMock(),
    )
    artifact = MagicMock()
    artifact.feedback_for_persona.return_value = "use fewer free parameters"

    with patch("gecco.feedback_coordinator.run_orchestrated_judge_pipeline", return_value=artifact) as runner:
        feedback, verdict = FeedbackCoordinator().resolve_feedback(
            search=search,
            iteration=4,
            run_idx=1,
            tag="",
            best_model=None,
            best_metric=None,
            recovery_failures=[{"name": "model_a"}],
            prev_had_success=True,
        )

    assert feedback == "use fewer free parameters"
    assert verdict.synthesized_feedback == "use fewer free parameters"
    runner.assert_called_once()
    search._set_activity.assert_called_once()


def test_distributed_coordinator_sync_and_update_delegate_registry():
    """Distributed coordination should be a small registry-facing service."""

    search = SimpleNamespace(
        shared_registry=MagicMock(),
        best_metric=float("inf"),
        best_model=None,
        best_params=[],
        tried_param_sets=[],
        feedback=SimpleNamespace(history=[]),
        _merged_history_count=0,
        client_id="client-a",
    )
    search.shared_registry.read.return_value = {
        "global_best": {
            "metric_value": 9.0,
            "model_code": "def best():\n    return 0",
            "param_names": ["alpha"],
            "client_id": "client-b",
        },
        "tried_param_sets": [["alpha"]],
        "iteration_history": [
            {"iteration": 0, "results": [{"function_name": "model_a"}], "client_id": "client-b"}
        ],
    }

    coordinator = DistributedCoordinator()
    coordinator.sync_from_registry(search=search)

    assert search.best_metric == 9.0
    assert search.best_model == "def best():\n    return 0"
    assert search.feedback.history[0]["client_id"] == "client-b"

    coordinator.update_registry(
        search=search,
        iteration=1,
        results=[{"function_name": "model_a"}],
        status="complete",
        had_runnable_model=True,
    )
    search.shared_registry.update.assert_called_once()
