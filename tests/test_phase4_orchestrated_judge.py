"""Phase 4 contract tests for the hard orchestrated-only judge pipeline."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gecco.cli.run_judge_orchestrator import run_orchestrator
from gecco.construct_feedback.orchestrated import (
    FeedbackArtifact,
    build_feedback_artifact,
    run_orchestrated_judge_pipeline,
)
from gecco.run_gecco import GeCCoModelSearch


def _make_single_worker_search(tmp_path: Path, judge_mock: MagicMock) -> GeCCoModelSearch:
    """Build a minimal search stub that exercises local judge dispatch only."""
    (tmp_path / "feedback").mkdir(parents=True, exist_ok=True)

    search = MagicMock(spec=GeCCoModelSearch)
    search.cfg = SimpleNamespace(
        task=SimpleNamespace(name="phase4_task"),
        loop=SimpleNamespace(max_iterations=1),
        judge=SimpleNamespace(
            orchestrated=True,
            barrier=SimpleNamespace(client_wait_seconds=1),
            capabilities=["performance_summary", "best_model_code"],
        ),
        evaluation=SimpleNamespace(fit_type="group", metric="bic"),
    )
    search.client_id = None
    search.shared_registry = None
    search.best_model = "def candidate_model(*args, **kwargs):\n    return 0.0"
    search.best_metric = 101.5
    search.best_iter = 0
    search.best_params = []
    search.best_param_names = []
    search.best_param_values = None
    search.best_id_results = None
    search.tried_param_sets = []
    search.feedback = SimpleNamespace(history=[], record_iteration=MagicMock())
    search.df = SimpleNamespace(participant=["p1"])
    search.results_dir = tmp_path
    search.tool_judge = judge_mock
    search._sync_from_registry = MagicMock()
    search._set_activity = MagicMock()
    search._update_registry = MagicMock()
    search._file_tag = MagicMock(return_value="")
    search._cmg_config = MagicMock(return_value=SimpleNamespace(enabled=True))
    search._validate_cmg_runtime = MagicMock()
    search._cmg_is_generator = MagicMock(return_value=False)
    search._run_cmg_generator_iteration = MagicMock()
    search._run_cmg_evaluator_iteration = MagicMock()
    search.run_n_shots = GeCCoModelSearch.run_n_shots.__get__(search, GeCCoModelSearch)
    return search


def test_build_feedback_artifact_returns_canonical_json_shape():
    """The canonical helper should define the shared feedback-artifact schema."""
    artifact = build_feedback_artifact(
        iteration=2,
        run_idx=1,
        tag="_demo",
        analysis_data={
            "trace": [{"tool": "trajectory", "result": "ok"}],
            "full_trace": [{"stage": "analysis"}],
            "best_bic": 98.2,
            "wall_time": 1.5,
            "is_stuck": False,
        },
        synthesized_feedback={"default": "Use a simpler mechanism."},
        verdict_payloads=[
            {
                "per_angle": [{"angle": "trajectory", "summary": "stable"}],
                "key_recommendations": ["Use a simpler mechanism."],
            }
        ],
        best_model=None,
        best_metric=None,
        include_best_model_code=False,
    )

    payload = artifact.model_dump()

    assert payload == {
        "iteration": 2,
        "run_idx": 1,
        "tag": "_demo",
        "timestamp": payload["timestamp"],
        "tool_call_count": 1,
        "wall_time_seconds": 1.5,
        "best_bic": 98.2,
        "tool_call_trace": [{"tool": "trajectory", "result": "ok"}],
        "full_trace": [{"stage": "analysis"}],
        "per_angle": [{"angle": "trajectory", "summary": "stable"}],
        "key_recommendations": ["Use a simpler mechanism."],
        "synthesized_feedback": {"default": "Use a simpler mechanism."},
        "personas": ["default"],
        "stuck_search": False,
        "short_circuit": False,
        "no_substantive_feedback": False,
        "random_feedback_only": False,
        "best_model_code_included": False,
    }


def test_run_orchestrated_judge_pipeline_persists_best_model_code_inside_artifact(tmp_path):
    """best_model_code should be added inside the orchestrated artifact, not later."""
    judge = MagicMock()
    judge.get_feedback_analysis.return_value = {
        "iteration": 1,
        "analysis_text": "analysis text",
        "trace": [{"tool": "get_bic_trajectory", "result": "ok"}],
        "full_trace": [{"stage": "analysis"}],
        "best_bic": 77.7,
        "is_stuck": False,
        "wall_time": 2.0,
        "short_circuit": False,
    }
    judge.synthesize_for_persona.return_value = (
        "Try a more mechanistically distinct update.",
        {
            "per_angle": [{"angle": "mechanism", "summary": "too similar"}],
            "key_recommendations": ["Try a more mechanistically distinct update."],
        },
    )

    cfg = SimpleNamespace(
        judge=SimpleNamespace(capabilities=["performance_summary", "best_model_code"]),
        clients={},
        centralized_model_generation=SimpleNamespace(enabled=False),
    )

    artifact = run_orchestrated_judge_pipeline(
        judge=judge,
        cfg=cfg,
        results_dir=tmp_path,
        iteration=1,
        run_idx=0,
        tag="",
        best_model="def best_model(x):\n    return x",
        best_metric=77.7,
        recovery_failures=None,
        prev_had_success=True,
    )

    persisted = json.loads((tmp_path / "judge" / "iter1_run0.json").read_text(encoding="utf-8"))
    feedback_text = artifact.synthesized_feedback["default"]

    assert artifact.best_model_code_included is True
    assert "Best model code so far (BIC=77.70):" in feedback_text
    assert feedback_text.count("def best_model(x):") == 1
    assert persisted["synthesized_feedback"]["default"] == feedback_text
    assert persisted["best_model_code_included"] is True


def test_single_worker_run_uses_local_orchestrated_runner_when_registry_missing(tmp_path):
    """Single-worker runs should still route through the orchestrated helper stack."""
    judge_mock = MagicMock()
    captured_feedback = []
    artifact = FeedbackArtifact(
        iteration=0,
        run_idx=0,
        tag="",
        timestamp="2026-06-04T12:00:00+00:00",
        tool_call_count=1,
        wall_time_seconds=1.0,
        best_bic=101.5,
        tool_call_trace=[],
        full_trace=[],
        per_angle=[],
        key_recommendations=["Use a simpler mechanism next."],
        synthesized_feedback={"default": "Use a simpler mechanism next."},
        personas=["default"],
        stuck_search=False,
        short_circuit=False,
        no_substantive_feedback=False,
        random_feedback_only=False,
        best_model_code_included=False,
    )

    search = _make_single_worker_search(tmp_path, judge_mock)
    search._run_cmg_evaluator_iteration.side_effect = (
        lambda it, run_idx, feedback, cmg_cfg, baseline_bic: captured_feedback.append(feedback)
    )

    with patch("gecco.run_gecco.run_orchestrated_judge_pipeline", return_value=artifact) as runner:
        search.run_n_shots(0, None)

    runner.assert_called_once()
    assert runner.call_args.kwargs["judge"] is judge_mock
    assert captured_feedback == ["Use a simpler mechanism next."]


def test_single_worker_run_surfaces_local_orchestrated_runner_failures(tmp_path):
    """Missing shared registry should not revive the direct get_feedback() bypass."""
    search = _make_single_worker_search(tmp_path, MagicMock())

    with patch(
        "gecco.run_gecco.run_orchestrated_judge_pipeline",
        side_effect=RuntimeError("judge synthesis failed"),
    ):
        with pytest.raises(RuntimeError, match="judge synthesis failed"):
            search.run_n_shots(0, None)


def test_run_gecco_does_not_mutate_feedback_after_judge_returns(tmp_path):
    """run_gecco should forward artifact feedback verbatim without appending extra code."""
    judge_mock = MagicMock()
    captured_feedback = []
    feedback_text = (
        "Use a simpler mechanism next.\n\n---\n"
        "Best model code so far (BIC=101.50):\n"
        "```python\ndef candidate_model(*args, **kwargs):\n    return 0.0\n```"
    )
    artifact = FeedbackArtifact(
        iteration=0,
        run_idx=0,
        tag="",
        timestamp="2026-06-04T12:00:00+00:00",
        tool_call_count=1,
        wall_time_seconds=1.0,
        best_bic=101.5,
        tool_call_trace=[],
        full_trace=[],
        per_angle=[],
        key_recommendations=["Use a simpler mechanism next."],
        synthesized_feedback={"default": feedback_text},
        personas=["default"],
        stuck_search=False,
        short_circuit=False,
        no_substantive_feedback=False,
        random_feedback_only=False,
        best_model_code_included=True,
    )

    search = _make_single_worker_search(tmp_path, judge_mock)
    search._run_cmg_evaluator_iteration.side_effect = (
        lambda it, run_idx, feedback, cmg_cfg, baseline_bic: captured_feedback.append(feedback)
    )

    with patch("gecco.run_gecco.run_orchestrated_judge_pipeline", return_value=artifact):
        search.run_n_shots(0, None)

    assert captured_feedback == [feedback_text]
    assert captured_feedback[0].count("Best model code so far") == 1


def test_orchestrator_uses_shared_orchestrated_runner(tmp_path):
    """Distributed orchestration should use the same helper as the local path."""
    cfg = SimpleNamespace(
        task=SimpleNamespace(name="phase4_trace"),
        loop=SimpleNamespace(max_iterations=1),
        llm=SimpleNamespace(provider="openai", base_model="gpt-test"),
        judge=SimpleNamespace(
            orchestrated=True,
            barrier=SimpleNamespace(
                orchestrator_wait_seconds=1,
                retry_wait_seconds=1,
            ),
        ),
        data=SimpleNamespace(
            path="dummy.csv",
            input_columns=["choice_1"],
            id_column="participant",
            splits={"prompt": "[1:2]"},
            data2text_function="narrative",
            narrative_template="trial {choice_1}",
        ),
        clients={
            "explore": SimpleNamespace(
                llm=SimpleNamespace(feedback_guidance="Prioritise mechanism diversity.")
            )
        },
        centralized_model_generation=SimpleNamespace(enabled=False),
    )

    mock_registry = MagicMock()
    mock_registry.wait_for_clients_complete.return_value = 1
    mock_registry.count_clients_with_models.return_value = 1
    mock_registry.read.return_value = {
        "global_best": {
            "model_code": "def distributed_best_model(x):\n    return x",
            "metric_value": 98.0,
        }
    }
    artifact = FeedbackArtifact(
        iteration=0,
        run_idx=0,
        tag="_orchestrator",
        timestamp="2026-06-04T12:00:00+00:00",
        tool_call_count=1,
        wall_time_seconds=1.0,
        best_bic=98.0,
        tool_call_trace=[{"tool": "get_bic_trajectory", "result": "ok"}],
        full_trace=[{"step": "analysis", "detail": "ok"}],
        per_angle=[{"angle": "trajectory", "summary": "stalled"}],
        key_recommendations=["Focus on a different mechanism family."],
        synthesized_feedback={"explore": "Focus on a different mechanism family."},
        personas=["explore"],
        stuck_search=False,
        short_circuit=False,
        no_substantive_feedback=False,
        random_feedback_only=False,
        best_model_code_included=False,
    )

    with patch("gecco.cli.run_judge_orchestrator.load_config", return_value=cfg):
        with patch(
            "gecco.cli.run_judge_orchestrator.SharedRegistry",
            return_value=mock_registry,
        ):
            with patch(
                "gecco.cli.run_judge_orchestrator.load_llm",
                return_value=(None, None),
            ):
                with patch(
                    "gecco.cli.run_judge_orchestrator.load_data",
                    return_value=MagicMock(),
                ):
                    with patch(
                        "gecco.cli.run_judge_orchestrator.split_by_participant",
                        return_value={"prompt": MagicMock()},
                    ):
                        with patch(
                            "gecco.cli.run_judge_orchestrator.get_data2text_function",
                            return_value=lambda *args, **kwargs: "data text",
                        ):
                            with patch(
                                "gecco.cli.run_judge_orchestrator.rebuild_from_artifacts",
                                return_value=MagicMock(),
                            ):
                                with patch(
                                    "gecco.cli.run_judge_orchestrator.ToolUsingJudge",
                                    return_value=MagicMock(),
                                ):
                                    with patch(
                                        "gecco.cli.run_judge_orchestrator.run_orchestrated_judge_pipeline",
                                        return_value=artifact,
                                    ) as runner:
                                        with patch(
                                            "gecco.cli.run_judge_orchestrator.init_sentry"
                                        ):
                                            run_orchestrator(
                                                config="test.yaml",
                                                results_dir=str(tmp_path),
                                                n_clients=1,
                                            )

    runner.assert_called_once()
    assert runner.call_args.kwargs["best_model"] == "def distributed_best_model(x):\n    return x"
    assert runner.call_args.kwargs["best_metric"] == 98.0
    assert mock_registry.set_judge_feedback.call_args.kwargs["synthesized_feedback"] == {
        "explore": "Focus on a different mechanism family."
    }
