"""Direct contract tests for the candidate evaluation service."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gecco.artifacts import ArtifactStore
from gecco.candidate_evaluation import CandidateEvaluator
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

    update_registry = MagicMock()
    had_runnable_model = evaluator.finalize_iteration_results(
        iteration=0,
        run_idx=1,
        tag="",
        iteration_results=[result],
        client_id=None,
        results_source=SimpleNamespace(),
        shared_registry=None,
        update_registry=update_registry,
        feedback_record=MagicMock(),
    )

    assert had_runnable_model is True
    assert (run_context.results_dir / "bics" / "iter0_run1.json").exists()
    update_registry.assert_called_once()

    diagnostic_store.close()
    run_context.close()
