"""Direct contract tests for the candidate evaluation service."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import pytest

from gecco.artifacts import ArtifactStore
from gecco.candidate_evaluation import BestModelState, CandidateEvaluator
from gecco.coordination import SharedRegistry
from gecco.diagnostic_store.store import DiagnosticStore
from gecco.run_context import RunContext


def test_candidate_evaluator_fits_and_finalises_without_monolith(tmp_path: Path):
    """The evaluator should fit a candidate and finalise iteration results directly."""

    cfg = SimpleNamespace(
        data=SimpleNamespace(input_columns=["trial"]),
        task=SimpleNamespace(name="phase6_task"),
        evaluation=SimpleNamespace(fit_type="group"),
    )
    run_context = RunContext.from_cfg(cfg, project_root=tmp_path)
    diagnostic_store = DiagnosticStore(tmp_path / "diagnostics.duckdb")
    artifact_store = ArtifactStore(run_context, diagnostic_store)
    evaluator = CandidateEvaluator(artifact_store)

    with patch("gecco.offline_evaluation.fit_generated_models.run_fit") as run_fit:
        run_fit.return_value = {
            "function_name": "model_a",
            "metric_name": "BIC",
            "metric_value": 12.3,
            "param_names": ["alpha"],
            "code": "@njit\ndef cognitive_model1(x, model_parameters):\n    alpha, = model_parameters\n    return alpha",
            "eval_metrics": [12.3],
            "participant_n_trials": [3],
            "parameter_values": [[0.5]],
            "mean_nll": 4.2,
            "per_participant_nll": [4.2],
        }

        result, should_stop = evaluator.fit_candidate_model(
            model_dict={
                "func_name": "cognitive_model1",
                "name": "model_a",
                "code": "@njit\ndef cognitive_model1(x, model_parameters):\n    alpha, = model_parameters\n    return alpha",
                "parameters": [{"name": "alpha", "lower_bound": 0, "upper_bound": 1}],
            },
            model_idx=0,
            n_models=1,
            it=0,
            run_idx=1,
            tag="",
            model_file=artifact_store.candidate_model_path(iteration=0, run_idx=1, tag=""),
            baseline_bic=None,
            df=SimpleNamespace(),
            cfg=cfg,
        )

    assert should_stop is False
    assert result["metric_name"] == "BIC"

    had_runnable_model = evaluator.finalize_iteration_results(
        iteration=0,
        run_idx=1,
        tag="",
        iteration_results=[result],
        client_id=None,
        results_source=SimpleNamespace(),
        shared_registry=None,
        feedback_record=MagicMock(),
    )

    assert had_runnable_model is True
    assert not (run_context.results_dir / "bics" / "iter0_run1.json").exists()
    assert diagnostic_store.fetchone("SELECT COUNT(*) AS n FROM iterations") == {"n": 1}
    assert diagnostic_store.fetchone("SELECT COUNT(*) AS n FROM models") == {"n": 1}

    diagnostic_store.close()
    run_context.close()


def test_candidate_evaluator_finalisation_orders_feedback_before_registry_publish(
    tmp_path: Path,
):
    """Finalisation should persist, then record feedback, then publish status."""

    cfg = SimpleNamespace(
        data=SimpleNamespace(input_columns=[]),
        task=SimpleNamespace(name="phase6_eval_task"),
        evaluation=SimpleNamespace(fit_type="group"),
    )
    run_context = RunContext.from_cfg(cfg, project_root=tmp_path)
    artifact_store = ArtifactStore(run_context)
    evaluator = CandidateEvaluator(artifact_store)
    events: list[str] = []
    iteration_results = [
        {
            "function_name": "model_a",
            "metric_name": "BIC",
            "metric_value": 12.3,
            "param_names": ["alpha"],
            "code": "def model_a():\n    return 0",
        }
    ]
    shared_registry = MagicMock()

    with patch.object(
        artifact_store,
        "write_iteration_results",
        side_effect=lambda **kwargs: events.append("write") or True,
    ):
        feedback_record = MagicMock(side_effect=lambda *args, **kwargs: events.append("feedback"))
        shared_registry.update.side_effect = lambda **kwargs: events.append("publish")

        had_runnable_model = evaluator.finalize_iteration_results(
            iteration=0,
            run_idx=1,
            tag="",
            iteration_results=iteration_results,
            client_id="client-a",
            results_source=SimpleNamespace(),
            shared_registry=shared_registry,
            feedback_record=feedback_record,
        )

    assert had_runnable_model is True
    assert events == ["write", "feedback", "publish"]
    feedback_record.assert_called_once_with(0, iteration_results)
    shared_registry.update.assert_called_once()

    run_context.close()


def test_candidate_evaluator_cmg_success_persists_before_registry_completion(
    tmp_path: Path,
):
    """CMG success should publish complete status only after canonical persistence."""

    cfg = SimpleNamespace(
        data=SimpleNamespace(input_columns=[]),
        task=SimpleNamespace(name="phase6_eval_task"),
        evaluation=SimpleNamespace(fit_type="group"),
    )
    run_context = RunContext.from_cfg(cfg, project_root=tmp_path)
    artifact_store = ArtifactStore(run_context)
    evaluator = CandidateEvaluator(artifact_store)
    shared_registry = SharedRegistry(tmp_path / "shared_registry.duckdb")

    with patch.object(evaluator, "fit_candidate_model") as fit_candidate_model:
        fit_candidate_model.return_value = (
            {
                "function_name": "model_a",
                "metric_name": "BIC",
                "metric_value": 2.0,
                "param_names": ["alpha"],
                "parameter_values": [[0.5]],
                "code": "def cognitive_model1(x, model_parameters):\n    return 0.0",
            },
            False,
        )
        shared_registry.set_candidate_models(
            0,
            [
                {
                    "index": 0,
                    "func_name": "cognitive_model1",
                    "name": "model_a",
                    "code": "def cognitive_model1(x, model_parameters):\n    return 0.0",
                    "parameters": [{"name": "alpha"}],
                }
            ],
            "generator",
        )

        result = evaluator.evaluate_iteration(
            iteration=0,
            run_idx=1,
            cmg_cfg=SimpleNamespace(n_models=1),
            tag="",
            client_id=0,
            evaluator_index=0,
            baseline_bic=None,
            shared_registry=shared_registry,
            df=SimpleNamespace(),
            cfg=cfg,
            max_syntax_retries=0,
            barrier_timeout_seconds=1,
            best_state=BestModelState(),
            tried_param_sets=[],
        )

    snapshot = shared_registry.read()

    assert result.iteration_results[0]["metric_name"] == "BIC"
    assert snapshot["client_entries"]["0"]["status"] == "complete"
    assert snapshot["client_entries"]["0"]["had_runnable_model"] is True
    assert snapshot["iteration_history"][0]["results"][0]["function_name"] == "model_a"
    run_context.close()


def test_candidate_evaluator_cmg_no_success_marks_complete_no_success(tmp_path: Path):
    """CMG failure without a runnable candidate should publish complete_no_success."""

    cfg = SimpleNamespace(
        data=SimpleNamespace(input_columns=[]),
        task=SimpleNamespace(name="phase6_eval_task"),
        evaluation=SimpleNamespace(fit_type="group"),
    )
    run_context = RunContext.from_cfg(cfg, project_root=tmp_path)
    artifact_store = ArtifactStore(run_context)
    evaluator = CandidateEvaluator(artifact_store)
    shared_registry = SharedRegistry(tmp_path / "shared_registry.duckdb")

    with patch.object(evaluator, "fit_candidate_model") as fit_candidate_model:
        fit_candidate_model.return_value = (
            {
                "function_name": "model_a",
                "metric_name": "VALIDATION_ERROR",
                "metric_value": float("inf"),
                "param_names": [],
                "code": "def cognitive_model1(x, model_parameters):\n    return 0.0",
                "error_type": "syntax",
                "error_message": "bad syntax",
                "error_details": {},
            },
            False,
        )
        shared_registry.set_candidate_models(
            0,
            [
                {
                    "index": 0,
                    "func_name": "cognitive_model1",
                    "name": "model_a",
                    "code": "def cognitive_model1(x, model_parameters):\n    return 0.0",
                    "parameters": [],
                }
            ],
            "generator",
        )

        evaluator.evaluate_iteration(
            iteration=0,
            run_idx=1,
            cmg_cfg=SimpleNamespace(n_models=1),
            tag="",
            client_id=0,
            evaluator_index=0,
            baseline_bic=None,
            shared_registry=shared_registry,
            df=SimpleNamespace(),
            cfg=cfg,
            max_syntax_retries=0,
            barrier_timeout_seconds=1,
            best_state=BestModelState(),
            tried_param_sets=[],
        )

    snapshot = shared_registry.read()

    assert snapshot["client_entries"]["0"]["status"] == "complete_no_success"
    assert snapshot["client_entries"]["0"]["had_runnable_model"] is False
    run_context.close()


def test_candidate_evaluator_cmg_write_failure_does_not_publish_completion(
    tmp_path: Path,
):
    """CMG persistence failure should raise and leave the registry non-terminal."""

    cfg = SimpleNamespace(
        data=SimpleNamespace(input_columns=[]),
        task=SimpleNamespace(name="phase6_eval_task"),
        evaluation=SimpleNamespace(fit_type="group"),
    )
    run_context = RunContext.from_cfg(cfg, project_root=tmp_path)
    artifact_store = ArtifactStore(run_context)
    evaluator = CandidateEvaluator(artifact_store)
    shared_registry = SharedRegistry(tmp_path / "shared_registry.duckdb")

    with patch.object(evaluator, "fit_candidate_model") as fit_candidate_model:
        fit_candidate_model.return_value = (
            {
                "function_name": "model_a",
                "metric_name": "BIC",
                "metric_value": 2.0,
                "param_names": ["alpha"],
                "parameter_values": [[0.5]],
                "code": "def cognitive_model1(x, model_parameters):\n    return 0.0",
            },
            False,
        )
        shared_registry.set_candidate_models(
            0,
            [
                {
                    "index": 0,
                    "func_name": "cognitive_model1",
                    "name": "model_a",
                    "code": "def cognitive_model1(x, model_parameters):\n    return 0.0",
                    "parameters": [{"name": "alpha"}],
                }
            ],
            "generator",
        )

        with patch.object(
            artifact_store,
            "write_iteration_results",
            side_effect=RuntimeError("duckdb unavailable"),
        ):
            with pytest.raises(RuntimeError, match="duckdb unavailable"):
                evaluator.evaluate_iteration(
                    iteration=0,
                    run_idx=1,
                    cmg_cfg=SimpleNamespace(n_models=1),
                    tag="",
                    client_id=0,
                    evaluator_index=0,
                    baseline_bic=None,
                    shared_registry=shared_registry,
                    df=SimpleNamespace(),
                    cfg=cfg,
                    max_syntax_retries=0,
                    barrier_timeout_seconds=1,
                    best_state=BestModelState(),
                    tried_param_sets=[],
                )

    snapshot = shared_registry.read()
    entry = snapshot["client_entries"].get("0")

    assert entry is None or entry["status"] not in {"complete", "complete_no_success"}
    run_context.close()


def test_candidate_evaluator_finalisation_write_failure_skips_feedback_and_publish(
    tmp_path: Path,
):
    """Persistence failures must stop feedback recording and registry publication."""

    cfg = SimpleNamespace(
        data=SimpleNamespace(input_columns=[]),
        task=SimpleNamespace(name="phase6_eval_task"),
        evaluation=SimpleNamespace(fit_type="group"),
    )
    run_context = RunContext.from_cfg(cfg, project_root=tmp_path)
    artifact_store = ArtifactStore(run_context)
    evaluator = CandidateEvaluator(artifact_store)
    feedback_record = MagicMock()
    shared_registry = MagicMock()

    with patch.object(
        artifact_store,
        "write_iteration_results",
        side_effect=RuntimeError("duckdb unavailable"),
    ):
        with pytest.raises(RuntimeError, match="duckdb unavailable"):
            evaluator.finalize_iteration_results(
                iteration=0,
                run_idx=1,
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
                feedback_record=feedback_record,
            )

    feedback_record.assert_not_called()
    shared_registry.update.assert_not_called()
    run_context.close()


def test_candidate_evaluator_repair_uses_direct_prompt_and_preserves_tried_params(
    tmp_path: Path,
):
    """CMG repair should use a direct repair prompt and preserve tried param sets."""

    cfg = SimpleNamespace(
        task=SimpleNamespace(name="phase6_eval_task"),
        evaluation=SimpleNamespace(fit_type="group"),
        data=SimpleNamespace(input_columns=[]),
        llm=SimpleNamespace(structured_output=True),
        clients=SimpleNamespace(
            evaluator=SimpleNamespace(
                naive_ideation=SimpleNamespace(enabled=True)
            )
        ),
    )
    run_context = RunContext.from_cfg(cfg, project_root=tmp_path)
    artifact_store = ArtifactStore(run_context)
    evaluator = CandidateEvaluator(artifact_store)
    shared_registry = MagicMock()
    shared_registry.wait_for_candidate_models.return_value = {
        "candidates": [
            {
                "index": 0,
                "func_name": "cognitive_model1",
                "name": "model_a",
                "code": "def cognitive_model1(x, model_parameters):\n    return 0.0",
                "parameters": [],
                "rationale": "compact model",
            }
        ]
    }

    fit_call_count = {"count": 0}

    def fit_candidate_model_side_effect(*, tried_param_sets=None, **kwargs):
        fit_call_count["count"] += 1
        if fit_call_count["count"] == 1:
            return (
                {
                    "function_name": "model_a",
                    "metric_name": "VALIDATION_ERROR",
                    "metric_value": float("inf"),
                    "error_type": "syntax",
                    "error_message": "bad syntax",
                    "error_details": {},
                },
                False,
            )
        if tried_param_sets is not None:
            tried_param_sets.append(["alpha"])
        return (
            {
                "function_name": "model_a",
                "metric_name": "BIC",
                "metric_value": 2.5,
                "param_names": ["alpha"],
                "parameter_values": [[0.5]],
                "code": "def cognitive_model1(x, model_parameters):\n    return 1.0",
            },
            False,
        )

    evaluator.fit_candidate_model = MagicMock(side_effect=fit_candidate_model_side_effect)
    prompt_builder = MagicMock()
    prompt_builder.build_input_prompt.return_value = "repair prompt"
    tried_param_sets: list[list[str]] = []

    with patch(
        "gecco.structured_output.parse_model_response",
        return_value=(
            [
                {
                    "name": "model_a",
                    "code": "def cognitive_model1(x, model_parameters):\n    return 1.0",
                    "parameters": [{"name": "alpha"}],
                }
            ],
            True,
        ),
    ), patch(
        "gecco.structured_output.get_model_schema",
        return_value={"type": "object"},
    ), patch.object(
        evaluator,
        "_validate_repaired_func_name",
        return_value=True,
    ):
        result = evaluator.evaluate_iteration(
            iteration=0,
            run_idx=1,
            cmg_cfg=SimpleNamespace(n_models=1),
            tag="",
            client_id="evaluator",
            evaluator_index=0,
            baseline_bic=None,
            shared_registry=shared_registry,
            df=SimpleNamespace(),
            cfg=cfg,
            model=object(),
            tokenizer=object(),
            generate_text=MagicMock(return_value='{"models": []}'),
            prompt_builder=prompt_builder,
            max_syntax_retries=1,
            barrier_timeout_seconds=1,
            best_state=BestModelState(),
            tried_param_sets=tried_param_sets,
        )

    assert result.iteration_results[0]["metric_name"] == "BIC"
    assert tried_param_sets == [["alpha"]]
    prompt_builder.build_input_prompt.assert_called_once_with(
        feedback_text=ANY,
        n_models=1,
        force_include_feedback=True,
    )
    shared_registry.update_candidate_model.assert_called_once()
    run_context.close()


def test_candidate_evaluator_finalisation_write_failure_is_surfaced(tmp_path: Path):
    """Canonical persistence failures should stop publication immediately."""

    cfg = SimpleNamespace(
        data=SimpleNamespace(input_columns=[]),
        task=SimpleNamespace(name="phase6_eval_task"),
        evaluation=SimpleNamespace(fit_type="group"),
    )
    run_context = RunContext.from_cfg(cfg, project_root=tmp_path)
    artifact_store = ArtifactStore(run_context)
    evaluator = CandidateEvaluator(artifact_store)
    feedback_record = MagicMock()

    with patch.object(
        artifact_store,
        "write_iteration_results",
        side_effect=RuntimeError("duckdb unavailable"),
    ):
        with patch.object(evaluator, "fit_candidate_model") as fit_candidate_model:
            fit_candidate_model.return_value = (
                {
                    "function_name": "model_a",
                    "metric_name": "BIC",
                    "metric_value": 2.0,
                    "param_names": ["alpha"],
                    "code": "def cognitive_model1(x, model_parameters):\n    return 0.0",
                },
                False,
            )
            with pytest.raises(RuntimeError, match="duckdb unavailable"):
                evaluator.run_non_cmg_iteration(
                    iteration=0,
                    run_idx=1,
                    tag="",
                    generation_result=SimpleNamespace(
                        parsed_models=[
                            {
                                "func_name": "cognitive_model1",
                                "name": "model_a",
                                "code": "def cognitive_model1(x, model_parameters):\n    return 0.0",
                                "parameters": [{"name": "alpha"}],
                            }
                        ],
                        model_file=artifact_store.candidate_model_path(
                            iteration=0,
                            run_idx=1,
                            tag="",
                        ),
                    ),
                    baseline_bic=None,
                    df=SimpleNamespace(),
                    cfg=cfg,
                    shared_registry=MagicMock(),
                    client_id="client-a",
                    results_source=SimpleNamespace(),
                    feedback_record=feedback_record,
                )

    feedback_record.assert_not_called()
    run_context.close()
