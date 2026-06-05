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
from gecco.run_gecco import GeCCoModelSearch
from gecco.run_context import RunContext


def _group_cfg() -> SimpleNamespace:
    return SimpleNamespace(task=SimpleNamespace(name="phase6_task"), evaluation=SimpleNamespace(fit_type="group"))


def _individual_cfg() -> SimpleNamespace:
    return SimpleNamespace(task=SimpleNamespace(name="phase6_task"), evaluation=SimpleNamespace(fit_type="individual"))


def test_run_context_resolves_paths_and_cleans_tempdir(tmp_path: Path):
    """RunContext should own run paths and temporary-directory lifecycle."""

    context = RunContext.from_cfg(_individual_cfg(), project_root=tmp_path, client_id=3)
    try:
        assert context.project_root == tmp_path
        assert context.results_dir == tmp_path / "results" / "phase6_task_individual"
        assert context.is_individual is True
        assert context.candidate_model_path(iteration=0, run_idx=2, tag="", participant="p1") == (
            tmp_path / "results" / "phase6_task_individual" / "models" / "iter0_run2_participantp1.txt"
        )
        assert context.feedback_path(iteration=1, run_idx=2, tag="", participant="p1").name == "iter1_run2_participantp1.txt"
        assert context.iteration_results_path(iteration=0, run_idx=2, tag="").name == "iter0_run2.json"
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


def test_artifact_store_contract_handles_group_and_individual_paths(tmp_path: Path):
    """Artefact writes should round-trip and use the correct path layout."""

    group_context = RunContext.from_cfg(_group_cfg(), project_root=tmp_path)
    individual_context = RunContext.from_cfg(_individual_cfg(), project_root=tmp_path)
    store = DiagnosticStore(tmp_path / "diagnostics.duckdb")
    try:
        group_store = ArtifactStore(group_context, store)
        individual_store = ArtifactStore(individual_context, store)

        assert group_store.candidate_model_path(iteration=0, run_idx=1, tag="") == (
            tmp_path / "results" / "phase6_task" / "models" / "iter0_run1.txt"
        )
        assert individual_store.candidate_model_path(iteration=0, run_idx=1, tag="", participant="p1") == (
            tmp_path / "results" / "phase6_task_individual" / "models" / "iter0_run1_participantp1.txt"
        )

        iteration_results = [
            {
                "function_name": "model_a",
                "metric_name": "BIC",
                "metric_value": 12.5,
                "param_names": ["alpha"],
                "code": "def model_a():\n    return 0",
            }
        ]

        had_runnable_model = group_store.write_iteration_results(
            iteration=0,
            run_idx=1,
            tag="",
            iteration_results=iteration_results,
            client_id="client-a",
        )

        assert had_runnable_model is True
        assert (tmp_path / "results" / "phase6_task" / "bics" / "iter0_run1.json").exists()
        assert store.fetchone("SELECT COUNT(*) AS n FROM models") == {"n": 1}
    finally:
        store.close()
        group_context.close()
        individual_context.close()


def test_candidate_generator_contract_uses_explicit_inputs_and_registry(tmp_path: Path):
    """The generator should operate from explicit inputs rather than search internals."""

    run_context = RunContext.from_cfg(_group_cfg(), project_root=tmp_path)
    artifact_store = ArtifactStore(run_context)
    generator = CandidateGenerator(artifact_store)
    shared_registry = MagicMock()

    result = generator.generate_iteration(
        iteration=0,
        run_idx=2,
        feedback="use simpler models",
        cmg_cfg=SimpleNamespace(n_models=2),
        tag="",
        client_id="client-a",
        naive_enabled=False,
        build_prompt=MagicMock(return_value="prompt text"),
        generate_models=MagicMock(
            return_value=(
                "def model_a():\n    return 0",
                [
                    {"name": "model_a", "code": "code_a", "parameters": ["alpha"]},
                    {"name": "model_b", "code": "code_b", "parameters": ["beta"]},
                ],
            )
        ),
        generate_models_naive=MagicMock(),
        shared_registry=shared_registry,
        participant=None,
        set_activity=MagicMock(),
    )

    assert result.candidates[0]["func_name"] == "cognitive_model1"
    assert result.model_file.exists()
    shared_registry.set_candidate_models.assert_called_once()
    shared_registry.set_generator_status.assert_called_once_with(
        iteration=0,
        client_id="client-a",
        status="complete",
        n_candidates=2,
    )
    run_context.close()


def test_candidate_evaluator_contract_finalises_iteration_results(tmp_path: Path):
    """The evaluator should fit the assigned candidate and finalise it."""

    run_context = RunContext.from_cfg(_group_cfg(), project_root=tmp_path)
    artifact_store = ArtifactStore(run_context)
    shared_registry = MagicMock()
    shared_registry.wait_for_candidate_models.return_value = {
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

    fit_candidate_model = MagicMock(
        side_effect=[
            (
                {
                    "function_name": "model_a",
                    "metric_name": "VALIDATION_ERROR",
                    "metric_value": float("inf"),
                    "error_type": "syntax",
                    "error_message": "bad syntax",
                    "error_details": {},
                },
                False,
            ),
            (
                {
                    "function_name": "model_a",
                    "metric_name": "BIC",
                    "metric_value": 10.0,
                    "param_names": [],
                },
                False,
            ),
        ]
    )
    repair_candidate = MagicMock(
        return_value={
            "func_name": "cognitive_model1",
            "name": "model_a",
            "code": "def model_a():\n    return 1",
            "parameters": [],
        }
    )
    update_registry = MagicMock()
    finalize_iteration_results = MagicMock(return_value=True)

    evaluator = CandidateEvaluator(artifact_store)
    result = evaluator.evaluate_iteration(
        iteration=0,
        run_idx=3,
        cmg_cfg=SimpleNamespace(n_models=1),
        tag="",
        client_id=0,
        evaluator_index=0,
        baseline_bic=99.0,
        shared_registry=shared_registry,
        fit_candidate_model=fit_candidate_model,
        is_repairable_error=lambda row: row is not None and row.get("metric_name") == "VALIDATION_ERROR",
        repair_candidate=repair_candidate,
        update_registry=update_registry,
        finalize_iteration_results=finalize_iteration_results,
        max_syntax_retries=1,
        barrier_timeout_seconds=1,
    )

    assert result.iteration_results[0]["metric_value"] == 10.0
    assert result.model_file.exists()
    update_registry.assert_called_once_with(0, [], status="retrying")
    finalize_iteration_results.assert_called_once()
    run_context.close()


def test_feedback_coordinator_uses_orchestrated_pipeline_for_persona_feedback(tmp_path: Path):
    """The feedback coordinator should adapt orchestrated feedback for the active persona."""

    judge = MagicMock()
    artifact = MagicMock()
    artifact.feedback_for_persona.return_value = "use fewer free parameters"
    original_import = __import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "gecco.run_gecco":
            raise AssertionError("FeedbackCoordinator should not import gecco.run_gecco")
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=guarded_import):
        feedback, verdict = FeedbackCoordinator(
            run_orchestrated_judge_pipeline=MagicMock(return_value=artifact)
        ).resolve_feedback(
            judge=judge,
            cfg=SimpleNamespace(),
            results_dir=tmp_path,
            iteration=4,
            run_idx=1,
            tag="",
            best_model=None,
            best_metric=None,
            recovery_failures=[{"name": "model_a"}],
            prev_had_success=True,
            persona_name="client-a",
            set_activity=MagicMock(),
        )

    assert feedback == "use fewer free parameters"
    assert verdict.synthesized_feedback == "use fewer free parameters"
    artifact.feedback_for_persona.assert_called_once_with("client-a")


def test_distributed_coordinator_sync_and_update_delegate_registry():
    """Distributed coordination should be a small registry-facing service."""

    shared_registry = MagicMock()
    shared_registry.read.return_value = {
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
    sync_result = coordinator.sync_from_registry(
        shared_registry=shared_registry,
        best_metric=float("inf"),
        best_model=None,
        best_params=[],
        tried_param_sets=[],
        feedback_history=[],
        merged_history_count=0,
        client_id="client-a",
    )

    assert sync_result.best_metric == 9.0
    assert sync_result.best_model == "def best():\n    return 0"
    assert sync_result.feedback_history[0]["client_id"] == "client-b"

    coordinator.update_registry(
        shared_registry=shared_registry,
        client_id="client-a",
        iteration=1,
        results=[{"function_name": "model_a"}],
        best_model="def best():\n    return 0",
        best_metric=9.0,
        best_params=["alpha"],
        tried_param_sets=[["alpha"]],
        status="complete",
        had_runnable_model=True,
    )
    shared_registry.update.assert_called_once()


def test_run_n_shots_non_cmg_routes_through_extracted_services(tmp_path: Path):
    """Non-CMG iterations should use the extracted generator/evaluator services."""

    search = GeCCoModelSearch.__new__(GeCCoModelSearch)
    search.cfg = SimpleNamespace(
        loop=SimpleNamespace(max_iterations=1),
        evaluation=SimpleNamespace(fit_type="group", metric="bic"),
        llm=SimpleNamespace(models_per_iteration=1),
        judge=None,
        validation=SimpleNamespace(max_syntax_retries=0),
        clients={},
    )
    search.df = SimpleNamespace()
    search.df_val = None
    search.model = object()
    search.tokenizer = object()
    search.prompt_builder = SimpleNamespace(
        build_input_prompt=MagicMock(return_value="prompt text")
    )
    search.client_id = None
    search.shared_registry = None
    search.best_model = None
    search.best_metric = float("inf")
    search.best_params = []
    search.best_param_names = []
    search.best_param_values = None
    search.best_iter = -1
    search.best_id_results = None
    search.tried_param_sets = []
    search.feedback = SimpleNamespace(history=[], record_iteration=MagicMock())
    search.tool_judge = None
    search._merged_history_count = 0
    search.recovery_checker = None
    search.id_eval_data = None
    search.ppc_enabled = False
    search._ppc_simulator = None
    search.ppc_n_sims = 100
    search.block_residuals_enabled = False
    search.block_residuals_n_blocks = 10
    search.run_context = None
    search.results_dir = tmp_path / "results"
    search.artifact_store = MagicMock()
    search.artifact_store.results_dir = search.results_dir
    search.artifact_store.write_candidate_artifacts = MagicMock(
        return_value=search.results_dir / "models" / "iter0_run0.txt"
    )
    search.candidate_generator = MagicMock()
    search.candidate_generator.generate_models = MagicMock(
        return_value=(
            "code text",
            [
                {
                    "name": "model_a",
                    "code": "def model_a():\n    return 0",
                    "parameters": [],
                }
            ],
        )
    )
    search.candidate_generator.generate_models_naive = MagicMock(
        side_effect=AssertionError("naive generation should not be used")
    )
    search.candidate_evaluator = MagicMock()
    search.candidate_evaluator.fit_candidate_model = MagicMock(
        return_value=(
            {
                "function_name": "model_a",
                "metric_name": "BIC",
                "metric_value": 1.0,
                "param_names": [],
                "code": "def model_a():\n    return 0",
            },
            False,
        )
    )
    search.candidate_evaluator.finalize_iteration_results = MagicMock(return_value=True)
    search.generate_models = MagicMock(side_effect=AssertionError("old path used"))
    search.generate_models_naive = MagicMock(side_effect=AssertionError("old path used"))
    search._fit_candidate_model = MagicMock(side_effect=AssertionError("old path used"))
    search._finalize_iteration_results = MagicMock(side_effect=AssertionError("old path used"))
    search._cmg_config = MagicMock(return_value=None)
    search._sync_from_registry = MagicMock()
    search._file_tag = MagicMock(return_value="")
    search._set_activity = MagicMock()
    search._update_registry = MagicMock()
    search.generate = MagicMock(return_value="text")
    search.distributed_coordinator = MagicMock(start_iteration=MagicMock(return_value=0))
    search.run_n_shots = GeCCoModelSearch.run_n_shots.__get__(search, GeCCoModelSearch)

    search.run_n_shots(0, None)

    search.candidate_generator.generate_models.assert_called_once()
    search.candidate_evaluator.fit_candidate_model.assert_called_once()
    search.candidate_evaluator.finalize_iteration_results.assert_called_once()
