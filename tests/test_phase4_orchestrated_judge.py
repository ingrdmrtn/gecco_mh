"""Phase 4 contract tests for the hard orchestrated-only judge pipeline."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from gecco.artifacts import ArtifactStore
from config.schema import load_config
import gecco.run_gecco as run_gecco_module
import gecco.cli.run_judge_orchestrator as run_judge_orchestrator_module
from gecco.cli.run_judge_orchestrator import run_orchestrator
from gecco.construct_feedback.orchestrated import (
    FeedbackArtifact,
    build_feedback_artifact,
    persist_feedback_artifact,
    run_orchestrated_judge_pipeline,
)
from gecco.candidate_evaluation import CandidateEvaluator
from gecco.construct_feedback.tool_judge import ToolUsingJudge
from gecco.feedback_coordinator import FeedbackCoordinator
from gecco.diagnostic_store.store import DiagnosticStore
from gecco.run_gecco import GeCCoModelSearch


def test_judge_mode_off_does_not_wait_for_centralized_feedback(tmp_path):
    """Distributed clients should not wait on judge feedback when mode is off."""
    search = GeCCoModelSearch.__new__(GeCCoModelSearch)
    search.cfg = SimpleNamespace(
        loop=SimpleNamespace(max_iterations=2),
        judge=SimpleNamespace(mode="off", barrier=SimpleNamespace(client_wait_seconds=1)),
        evaluation=SimpleNamespace(fit_type="group", metric="bic"),
        llm=SimpleNamespace(models_per_iteration=1),
        validation=SimpleNamespace(max_syntax_retries=0),
        centralized_model_generation=SimpleNamespace(enabled=False),
    )
    search.shared_registry = MagicMock()
    search.shared_registry.wait_for_judge_feedback.side_effect = AssertionError(
        "judge feedback should not be awaited"
    )
    search.client_id = "alpha"
    search.judge_enabled = False
    search.tool_judge = None
    search.best_model = None
    search.best_metric = 0.0
    search.best_params = None
    search.best_iter = 0
    search.best_state = SimpleNamespace()
    search.recovery_checker = None
    search.id_eval_data = None
    search.ppc_enabled = False
    search._ppc_simulator = None
    search.ppc_n_sims = 0
    search.block_residuals_enabled = False
    search.block_residuals_n_blocks = 0
    search.prompt_builder = None
    search.generate = None
    search.model = None
    search.tokenizer = None
    search.tried_param_sets = set()
    search.feedback = SimpleNamespace(history=[], record_iteration=MagicMock())
    search.df = SimpleNamespace()
    search._sync_from_registry = MagicMock()
    search._sync_best_attrs_from_state = MagicMock()
    search._set_activity = MagicMock()
    search._file_tag = MagicMock(return_value="")
    search._cmg_config = MagicMock(return_value=None)
    search._require_distributed_coordinator = MagicMock(
        return_value=SimpleNamespace(start_iteration=MagicMock(return_value=1))
    )
    search._require_artifact_store = MagicMock()
    search._require_candidate_generator = MagicMock(
        return_value=SimpleNamespace(generate_non_cmg_iteration=MagicMock(return_value=SimpleNamespace()))
    )
    search._require_candidate_evaluator = MagicMock(
        return_value=SimpleNamespace(
            run_non_cmg_iteration=MagicMock(
                return_value=SimpleNamespace(should_retry=False, had_runnable_model=False)
            )
        )
    )

    search.run_n_shots(run_idx=0, baseline_bic=0.0)

    search.shared_registry.wait_for_judge_feedback.assert_not_called()


def test_distributed_run_raises_immediately_when_shared_abort_is_present(tmp_path):
    """Abort state should stop a distributed client before any judge wait or fitting."""
    search = GeCCoModelSearch.__new__(GeCCoModelSearch)
    search.cfg = SimpleNamespace(
        loop=SimpleNamespace(max_iterations=2),
        judge=SimpleNamespace(mode="static", barrier=SimpleNamespace(client_wait_seconds=1)),
        evaluation=SimpleNamespace(fit_type="group", metric="bic"),
        llm=SimpleNamespace(models_per_iteration=1),
        validation=SimpleNamespace(max_syntax_retries=0),
        centralized_model_generation=SimpleNamespace(enabled=False),
    )
    search.shared_registry = MagicMock()
    search.shared_registry.raise_if_aborted.side_effect = RuntimeError("shared abort")
    search.shared_registry.wait_for_judge_feedback.side_effect = AssertionError(
        "judge feedback should not be awaited after abort"
    )
    search.client_id = "alpha"
    search.judge_enabled = True
    search.tool_judge = None
    search.best_model = "def candidate_model():\n    return 0.0"
    search.best_metric = 0.0
    search.best_params = None
    search.best_iter = 0
    search.best_state = SimpleNamespace()
    search.recovery_checker = None
    search.id_eval_data = None
    search.ppc_enabled = False
    search._ppc_simulator = None
    search.ppc_n_sims = 0
    search.block_residuals_enabled = False
    search.block_residuals_n_blocks = 0
    search.prompt_builder = None
    search.generate = MagicMock(side_effect=AssertionError("generate should not run"))
    search.model = None
    search.tokenizer = None
    search.tried_param_sets = set()
    search.feedback = SimpleNamespace(history=[], record_iteration=MagicMock())
    search.df = SimpleNamespace(participant=["p1"])
    search._sync_from_registry = MagicMock()
    search._sync_best_attrs_from_state = MagicMock()
    search._set_activity = MagicMock()
    search._file_tag = MagicMock(return_value="")
    search._cmg_config = MagicMock(return_value=None)
    search._require_distributed_coordinator = MagicMock(
        return_value=SimpleNamespace(start_iteration=MagicMock(return_value=1))
    )
    search._require_artifact_store = MagicMock()
    search._require_candidate_generator = MagicMock(
        return_value=SimpleNamespace(generate_non_cmg_iteration=MagicMock())
    )
    search._require_candidate_evaluator = MagicMock(
        return_value=SimpleNamespace(run_non_cmg_iteration=MagicMock())
    )

    with pytest.raises(RuntimeError, match="shared abort"):
        search.run_n_shots(run_idx=0, baseline_bic=0.0)

    search.shared_registry.raise_if_aborted.assert_called_once()
    search.shared_registry.wait_for_judge_feedback.assert_not_called()
    search._require_candidate_generator.assert_not_called()
    search._require_candidate_evaluator.assert_not_called()


def test_run_n_shots_reopens_diagnostic_store_only_after_judge_wait(tmp_path):
    """Judge waits should happen before the next diagnostic-store reopen."""
    search = GeCCoModelSearch.__new__(GeCCoModelSearch)
    search.cfg = SimpleNamespace(
        loop=SimpleNamespace(max_iterations=2),
        judge=SimpleNamespace(
            mode="static",
            barrier=SimpleNamespace(client_wait_seconds=1),
            diagnostic_store=SimpleNamespace(enabled=True),
        ),
        evaluation=SimpleNamespace(fit_type="group", metric="bic"),
        llm=SimpleNamespace(models_per_iteration=1),
        validation=SimpleNamespace(max_syntax_retries=0),
        centralized_model_generation=SimpleNamespace(enabled=False),
    )
    search.shared_registry = MagicMock()
    search.shared_registry.raise_if_aborted = MagicMock()
    search.client_id = "alpha"
    search.judge_enabled = True
    search.tool_judge = None
    search.best_model = "def candidate_model():\n    return 0.0"
    search.best_metric = 0.0
    search.best_params = []
    search.best_iter = 0
    search.best_state = SimpleNamespace()
    search.recovery_checker = None
    search.id_eval_data = None
    search.ppc_enabled = False
    search._ppc_simulator = None
    search.ppc_n_sims = 0
    search.block_residuals_enabled = False
    search.block_residuals_n_blocks = 0
    search.prompt_builder = None
    search.generate = MagicMock(return_value="candidate code")
    search.model = None
    search.tokenizer = None
    search.tried_param_sets = set()
    search.feedback = SimpleNamespace(history=[], record_iteration=MagicMock())
    search.df = SimpleNamespace(participant=["p1"])
    search.results_dir = tmp_path
    search.artifact_store = SimpleNamespace(diagnostic_store=MagicMock(name="diagnostic_store"))
    search.diagnostic_store = search.artifact_store.diagnostic_store
    search._sync_from_registry = MagicMock()
    search._sync_best_attrs_from_state = MagicMock()
    search._set_activity = MagicMock()
    search._file_tag = MagicMock(return_value="")
    search._cmg_config = MagicMock(return_value=None)
    search._require_artifact_store = MagicMock(return_value=search.artifact_store)
    search._require_distributed_coordinator = MagicMock(
        return_value=SimpleNamespace(start_iteration=MagicMock(return_value=0))
    )
    search._require_candidate_generator = MagicMock(
        return_value=SimpleNamespace(
            generate_non_cmg_iteration=MagicMock(return_value=SimpleNamespace())
        )
    )
    events = []

    def wait_for_judge_feedback(*, iteration, timeout_seconds, poll_seconds):
        events.append("wait")
        assert search._ensure_diagnostic_store_for_iteration.call_count == 1
        assert search.diagnostic_store is None
        return {"synthesized_feedback": "", "key_recommendations": []}

    search.shared_registry.wait_for_judge_feedback.side_effect = wait_for_judge_feedback

    def run_non_cmg_iteration(*, iteration, release_diagnostics_for_judge=None, **kwargs):
        events.append(f"write_{iteration}")
        if iteration == 0:
            assert release_diagnostics_for_judge is not None
            release_diagnostics_for_judge()
        else:
            assert search._ensure_diagnostic_store_for_iteration.call_count == 2
        return SimpleNamespace(should_retry=False, had_runnable_model=False, retry_feedback=None)

    search._require_candidate_evaluator = MagicMock(
        return_value=SimpleNamespace(run_non_cmg_iteration=MagicMock(side_effect=run_non_cmg_iteration))
    )
    search._ensure_diagnostic_store_for_iteration = MagicMock(
        side_effect=lambda: events.append("ensure")
    )
    search.run_n_shots = GeCCoModelSearch.run_n_shots.__get__(search, GeCCoModelSearch)

    search.run_n_shots(run_idx=0, baseline_bic=0.0)

    assert events == ["ensure", "write_0", "wait", "ensure", "write_1"]
    search.shared_registry.wait_for_judge_feedback.assert_called_once()
    search._ensure_diagnostic_store_for_iteration.assert_has_calls([call(), call()])


def test_run_n_shots_skips_diagnostic_store_reopen_for_cmg_generator(tmp_path):
    """CMG generator runs should not reopen diagnostics unless a write is imminent."""
    search = GeCCoModelSearch.__new__(GeCCoModelSearch)
    search.cfg = SimpleNamespace(
        loop=SimpleNamespace(max_iterations=1),
        judge=SimpleNamespace(
            mode="static",
            barrier=SimpleNamespace(client_wait_seconds=1),
            diagnostic_store=SimpleNamespace(enabled=True),
        ),
        evaluation=SimpleNamespace(fit_type="group", metric="bic"),
        llm=SimpleNamespace(models_per_iteration=1),
        validation=SimpleNamespace(max_syntax_retries=0),
        centralized_model_generation=SimpleNamespace(enabled=True),
    )
    search.shared_registry = MagicMock()
    search.shared_registry.raise_if_aborted = MagicMock()
    search.client_id = "alpha"
    search.judge_enabled = True
    search.tool_judge = None
    search.best_model = None
    search.best_metric = 0.0
    search.best_params = []
    search.best_iter = 0
    search.best_state = SimpleNamespace()
    search.recovery_checker = None
    search.id_eval_data = None
    search.ppc_enabled = False
    search._ppc_simulator = None
    search.ppc_n_sims = 0
    search.block_residuals_enabled = False
    search.block_residuals_n_blocks = 0
    search.prompt_builder = None
    search.generate = MagicMock(return_value="candidate code")
    search.model = None
    search.tokenizer = None
    search.tried_param_sets = set()
    search.feedback = SimpleNamespace(history=[], record_iteration=MagicMock())
    search.df = SimpleNamespace(participant=["p1"])
    search.results_dir = tmp_path
    search.artifact_store = SimpleNamespace(diagnostic_store=None)
    search.diagnostic_store = None
    search._sync_from_registry = MagicMock()
    search._sync_best_attrs_from_state = MagicMock()
    search._set_activity = MagicMock()
    search._file_tag = MagicMock(return_value="")
    search._cmg_config = MagicMock(return_value=SimpleNamespace(enabled=True))
    search._validate_cmg_runtime = MagicMock()
    search._cmg_is_generator = MagicMock(return_value=True)
    search._run_cmg_generator_iteration = MagicMock()
    search._run_cmg_evaluator_iteration = MagicMock()
    search._ensure_diagnostic_store_for_iteration = MagicMock()
    search._require_distributed_coordinator = MagicMock(
        return_value=SimpleNamespace(start_iteration=MagicMock(return_value=0))
    )
    search.run_n_shots = GeCCoModelSearch.run_n_shots.__get__(search, GeCCoModelSearch)

    search.run_n_shots(run_idx=0, baseline_bic=0.0)

    search._run_cmg_generator_iteration.assert_called_once()
    search._run_cmg_evaluator_iteration.assert_not_called()
    search._ensure_diagnostic_store_for_iteration.assert_not_called()


class _SummaryOnlyStore:
    """Store stub that exposes duplicate attempted models and a short trajectory."""

    def fetchone(self, query, params=None):
        if "MIN(CASE WHEN status='ok' THEN metric_value END) AS best_iter" in query:
            return {"best_iter": 98.5}
        return None

    def fetchall(self, query, params=None):
        if "GROUP BY m.iteration" in query:
            return [
                {"iteration": 0, "best_metric": 110.0, "n_models_total": 2, "n_ok": 2},
                {"iteration": 1, "best_metric": 98.5, "n_models_total": 4, "n_ok": 3},
            ]
        if "iteration =" in query:
            return [
                {"name": "alpha_model"},
                {"name": "beta_model"},
                {"name": "alpha_model"},
            ]
        return []


def _make_single_worker_search(tmp_path: Path, judge_mock: MagicMock) -> GeCCoModelSearch:
    """Build a minimal search stub that exercises local judge dispatch only."""
    (tmp_path / "feedback").mkdir(parents=True, exist_ok=True)

    search = MagicMock(spec=GeCCoModelSearch)
    search.cfg = SimpleNamespace(
        task=SimpleNamespace(name="phase4_task"),
        loop=SimpleNamespace(max_iterations=1),
        judge=SimpleNamespace(
            barrier=SimpleNamespace(client_wait_seconds=1),
            mode="static",
            context=SimpleNamespace(
                attempted_models=False,
                performance=True,
                best_model_code=True,
                diagnostic=False,
            ),
            output=SimpleNamespace(persona_synthesis=False),
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
    search.distributed_coordinator = MagicMock(start_iteration=MagicMock(return_value=0))
    search.feedback_coordinator = FeedbackCoordinator(
        lambda **kwargs: run_gecco_module.run_orchestrated_judge_pipeline(**kwargs)
    )
    search._require_distributed_coordinator = GeCCoModelSearch._require_distributed_coordinator.__get__(
        search, GeCCoModelSearch
    )
    search._require_feedback_coordinator = GeCCoModelSearch._require_feedback_coordinator.__get__(
        search, GeCCoModelSearch
    )
    search.run_n_shots = GeCCoModelSearch.run_n_shots.__get__(search, GeCCoModelSearch)
    return search


@pytest.mark.parametrize(
    "had_runnable_model, expected_status",
    [(True, "complete"), (False, "complete_no_success")],
)
def test_finalize_iteration_results_releases_diagnostics_before_registry_update(
    had_runnable_model, expected_status
):
    """Judge completion should publish only after diagnostics are released."""
    evaluator = CandidateEvaluator.__new__(CandidateEvaluator)
    events = []
    evaluator.artifact_store = SimpleNamespace(
        write_iteration_results=MagicMock(
            side_effect=lambda **kwargs: events.append("write_iteration_results") or had_runnable_model
        )
    )

    def publish_registry_status(**kwargs):
        events.append("registry_update")
        assert kwargs["status"] == expected_status

    evaluator._publish_registry_status = MagicMock(side_effect=publish_registry_status)

    release_diagnostics_for_judge = MagicMock(
        side_effect=lambda: events.append("release_diagnostics")
    )

    result = evaluator.finalize_iteration_results(
        iteration=1,
        run_idx=0,
        tag="",
        iteration_results=[{"function_name": "candidate_model", "metric_name": "BIC", "metric_value": 10.0}],
        client_id="client-a",
        results_source=SimpleNamespace(participant=["p1"]),
        shared_registry=SimpleNamespace(),
        release_diagnostics_for_judge=release_diagnostics_for_judge,
    )

    assert result is had_runnable_model
    assert events == ["write_iteration_results", "release_diagnostics", "registry_update"]
    release_diagnostics_for_judge.assert_called_once()
    evaluator.artifact_store.write_iteration_results.assert_called_once()
    evaluator._publish_registry_status.assert_called_once()


def test_diagnostic_store_reopens_after_release_and_updates_artifact_store(tmp_path):
    """Released diagnostic writers should be recreated for the next iteration."""
    search = GeCCoModelSearch.__new__(GeCCoModelSearch)
    search.cfg = SimpleNamespace(
        judge=SimpleNamespace(diagnostic_store=SimpleNamespace(enabled=True))
    )
    search.run_context = SimpleNamespace(
        default_diagnostics_path=MagicMock(return_value=tmp_path / "diagnostics.duckdb")
    )
    search.shared_registry = object()
    search.judge_enabled = True
    search.artifact_store = SimpleNamespace(diagnostic_store=None)

    first_store = MagicMock(name="first_store")
    second_store = MagicMock(name="second_store")

    with patch("gecco.run_gecco._DiagnosticStore", side_effect=[first_store, second_store]) as store_cls:
        search._ensure_diagnostic_store_for_iteration()
        assert search.diagnostic_store is first_store
        assert search.artifact_store.diagnostic_store is first_store

        search._release_diagnostic_store_for_judge()
        first_store.close.assert_called_once()
        assert search.diagnostic_store is None
        assert search.artifact_store.diagnostic_store is None

        search._ensure_diagnostic_store_for_iteration()
        assert search.diagnostic_store is second_store
        assert search.artifact_store.diagnostic_store is second_store

    assert store_cls.call_args_list == [
        call(tmp_path / "diagnostics.duckdb"),
        call(tmp_path / "diagnostics.duckdb"),
    ]


@pytest.mark.parametrize(
    "generator_client, should_open_store",
    [("alpha", False), ("beta", True)],
)
def test_constructor_skips_diagnostic_store_open_for_cmg_generator(
    tmp_path, generator_client, should_open_store
):
    """Constructor-time store opening should respect CMG generator ownership."""
    tempdir = TemporaryDirectory(dir=tmp_path)
    try:
        run_context = run_gecco_module.RunContext(
            project_root=tmp_path,
            results_dir=tmp_path,
            temp_root=tmp_path,
            tempdir=tempdir,
            client_id="alpha",
        )
        cfg = SimpleNamespace(
            task=SimpleNamespace(name="phase4_task"),
            evaluation=SimpleNamespace(fit_type="group"),
            judge=SimpleNamespace(
                mode="static",
                diagnostic_store=SimpleNamespace(enabled=True),
            ),
            centralized_model_generation=SimpleNamespace(
                enabled=True,
                generator_client=generator_client,
            ),
        )

        store_instance = MagicMock(name="diagnostic_store")

        with patch(
            "gecco.run_gecco.RunContext.from_cfg", return_value=run_context
        ) as from_cfg_mock, patch(
            "gecco.run_gecco._DiagnosticStore", return_value=store_instance
        ) as store_cls:
            search = GeCCoModelSearch(
                model=MagicMock(),
                tokenizer=MagicMock(),
                cfg=cfg,
                df=SimpleNamespace(),
                prompt_builder=None,
                client_id="alpha",
                shared_registry=object(),
            )

        # Orchestrator-style construction resolves paths up front for the shared run context.
        from_cfg_mock.assert_called_once_with(
            cfg, client_id="alpha", config_path=None, results_dir=None
        )
        if should_open_store:
            store_cls.assert_called_once_with(run_context.default_diagnostics_path())
            assert search.diagnostic_store is store_instance
            assert search.artifact_store.diagnostic_store is store_instance
        else:
            store_cls.assert_not_called()
            assert search.diagnostic_store is None
            assert search.artifact_store.diagnostic_store is None
    finally:
        tempdir.cleanup()


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
            "metadata": {"source": "analysis"},
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
        "metadata": {"source": "analysis"},
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


def test_feedback_artifact_serialization_preserves_metadata(tmp_path):
    """Metadata should survive canonical artifact serialisation unchanged."""
    artifact = build_feedback_artifact(
        iteration=3,
        run_idx=2,
        tag="_meta",
        analysis_data={
            "trace": [],
            "full_trace": [],
            "best_bic": 88.1,
            "wall_time": 0.25,
            "metadata": {
                "shortcut_reason": "recovery_failure",
                "source_iteration": 1,
                "recovery_failures": [{"name": "candidate_a", "mean_r": 0.1}],
            },
        },
        synthesized_feedback={"default": "Stay with the earlier recommendation."},
        verdict_payloads=[],
        best_model=None,
        best_metric=None,
        include_best_model_code=False,
    )

    artifact_path = persist_feedback_artifact(artifact=artifact, results_dir=tmp_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload["metadata"] == artifact.metadata
    assert FeedbackArtifact.model_validate(payload).metadata == artifact.metadata


def test_recovery_failure_preserves_metadata_per_angle_and_recommendations(tmp_path):
    """Recovery shortcuts should keep structured metadata and verdict payloads."""
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    source_payload = {
        "iteration": 0,
        "run_idx": 0,
        "tag": "",
        "timestamp": "2026-06-04T12:00:00+00:00",
        "tool_call_count": 1,
        "wall_time_seconds": 1.0,
        "best_bic": 91.2,
        "tool_call_trace": [{"tool": "get_bic_trajectory", "result": "ok"}],
        "full_trace": [{"stage": "analysis"}],
        "metadata": {"source": "original", "provenance": {"iteration": 0}},
        "per_angle": [{"angle": "mechanism", "findings": "Too similar"}],
        "key_recommendations": ["Try a different mechanism family."],
        "synthesized_feedback": {"default": "Try a different mechanism family."},
        "personas": ["default"],
        "stuck_search": False,
        "short_circuit": False,
        "no_substantive_feedback": False,
        "random_feedback_only": False,
        "best_model_code_included": False,
    }
    (judge_dir / "iter0_run0.json").write_text(
        json.dumps(source_payload), encoding="utf-8"
    )

    judge = ToolUsingJudge.__new__(ToolUsingJudge)
    judge.mode = "llm"
    judge.results_dir = tmp_path
    judge.verbose = False
    judge.cfg = SimpleNamespace(centralized_model_generation=SimpleNamespace(enabled=False))

    artifact = run_orchestrated_judge_pipeline(
        judge=judge,
        cfg=judge.cfg,
        results_dir=tmp_path,
        iteration=1,
        run_idx=0,
        tag="",
        best_model=None,
        best_metric=77.7,
        recovery_failures=[
            {
                "name": "candidate_b",
                "mean_r": 0.12,
                "per_param_r": {"alpha": 0.05, "beta": 0.11},
            }
        ],
        prev_had_success=False,
    )

    assert artifact.short_circuit is True
    assert artifact.metadata["source"] == "original"
    assert artifact.metadata["shortcut_reason"] == "recovery_failure"
    assert artifact.metadata["source_iteration"] == 0
    assert artifact.per_angle == [{"angle": "mechanism", "findings": "Too similar"}]
    assert artifact.key_recommendations == ["Try a different mechanism family."]
    assert artifact.synthesized_feedback == {
        "default": artifact.synthesized_feedback["default"]
    }
    assert "{'default':" not in artifact.synthesized_feedback["default"]
    assert '"default":' not in artifact.synthesized_feedback["default"]


def test_static_mode_ignores_recovery_shortcut_verdict_prose(tmp_path):
    """Static mode should not reuse shortcut verdict text from old recovery failures."""
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    (judge_dir / "iter0_run0.json").write_text(
        json.dumps(
            {
                "iteration": 0,
                "run_idx": 0,
                "tag": "",
                "timestamp": "2026-06-04T12:00:00+00:00",
                "tool_call_count": 1,
                "wall_time_seconds": 1.0,
                "best_bic": 91.2,
                "tool_call_trace": [],
                "full_trace": [],
                "metadata": {"shortcut_reason": "recovery_failure"},
                "per_angle": [{"angle": "mechanism", "findings": "Too similar"}],
                "key_recommendations": ["Try a different mechanism family."],
                "synthesized_feedback": {
                    "default": (
                        "Update — previous iteration candidate(s) rejected for poor parameter recovery:\n"
                        "- candidate_a: mean r=0.10\n"
                        "Do not repropose these mechanisms without addressing the identifiability issues."
                    )
                },
                "personas": ["default"],
                "stuck_search": False,
                "short_circuit": True,
                "no_substantive_feedback": False,
                "random_feedback_only": False,
                "best_model_code_included": False,
            }
        ),
        encoding="utf-8",
    )

    cfg = SimpleNamespace(
        llm=SimpleNamespace(provider="openai", base_model="gpt-test"),
        judge=SimpleNamespace(
            mode="static",
            context=SimpleNamespace(
                attempted_models=False,
                performance=True,
                best_model_code=False,
                diagnostic=False,
            ),
            output=SimpleNamespace(persona_synthesis=False),
        ),
    )
    judge = ToolUsingJudge(
        cfg=cfg,
        diagnostic_store=_SummaryOnlyStore(),
        model=object(),
        tokenizer=None,
        results_dir=tmp_path,
    )
    judge._try_shortcut_from_recovery_failure = MagicMock(
        side_effect=AssertionError("static mode should not reuse recovery shortcut prose")
    )

    analysis = judge.get_feedback_analysis(
        iteration=1,
        run_idx=0,
        tag="",
        best_metric=77.7,
        recovery_failures=[{"name": "candidate_b", "mean_r": 0.12, "per_param_r": {}}],
        prev_had_success=False,
    )

    assert "short_circuit" not in analysis
    assert "Update — previous iteration candidate(s) rejected for poor parameter recovery" not in analysis["analysis_text"]
    assert "Previous verdict (state unchanged since iter 0)" not in analysis["analysis_text"]


def test_local_single_worker_without_client_id_uses_default_feedback(tmp_path):
    """Single-worker synthesis should fall back to default feedback when no client id exists."""
    judge = MagicMock()
    judge.get_feedback_analysis.return_value = {
        "iteration": 0,
        "analysis_text": "analysis text",
        "trace": [],
        "full_trace": [],
        "best_bic": 100.0,
        "is_stuck": False,
        "wall_time": 1.0,
        "short_circuit": False,
    }
    judge.synthesize_for_persona.return_value = (
        "Default feedback only.",
        {"per_angle": [], "key_recommendations": []},
    )

    cfg = SimpleNamespace(
        judge=SimpleNamespace(
            mode="static",
            context=SimpleNamespace(
                attempted_models=False,
                performance=True,
                best_model_code=False,
                diagnostic=False,
            ),
            output=SimpleNamespace(persona_synthesis=False),
        ),
        clients={
            "explore": SimpleNamespace(
                llm=SimpleNamespace(feedback_guidance="Explore broadly.")
            )
        },
        centralized_model_generation=SimpleNamespace(enabled=False),
    )

    artifact = run_orchestrated_judge_pipeline(
        judge=judge,
        cfg=cfg,
        results_dir=tmp_path,
        iteration=0,
        run_idx=0,
        tag="",
        best_model=None,
        best_metric=None,
        recovery_failures=None,
        prev_had_success=True,
    )

    judge.synthesize_for_persona.assert_called_once_with(
        judge.get_feedback_analysis.return_value,
        persona_name="default",
        persona_suffix="",
        persona_config=None,
    )
    assert artifact.synthesized_feedback == {"default": "Default feedback only."}
    assert artifact.feedback_for_persona("explore") == "Default feedback only."


def test_deterministic_synthesis_is_persisted_unchanged(tmp_path):
    """Narrow deterministic synthesis should persist exactly as returned."""
    cfg = load_config(
        Path(__file__).resolve().parents[1]
        / "config"
        / "archive"
        / "two_step_factors_gemini3flash_capabilities_summary_only.yaml"
    )
    judge = ToolUsingJudge(
        cfg=cfg,
        diagnostic_store=_SummaryOnlyStore(),
        model=object(),
        tokenizer=None,
        results_dir=tmp_path,
    )

    analysis = judge.get_feedback_analysis(iteration=0, run_idx=0, tag="", best_metric=95.0)

    artifact = run_orchestrated_judge_pipeline(
        judge=judge,
        cfg=cfg,
        results_dir=tmp_path,
        iteration=0,
        run_idx=0,
        tag="",
        best_model=None,
        best_metric=95.0,
        recovery_failures=None,
        prev_had_success=True,
    )

    assert analysis["narrow_deterministic"] is True
    assert artifact.synthesized_feedback == {"default": analysis["analysis_text"]}
    assert artifact.key_recommendations == []
    assert artifact.personas == ["default"]


def test_persona_synthesis_fanout_only_runs_when_enabled(tmp_path):
    """Persona fan-out should be disabled unless the capability is enabled."""
    judge = MagicMock()
    judge.get_feedback_analysis.return_value = {
        "iteration": 0,
        "analysis_text": "analysis text",
        "trace": [],
        "full_trace": [],
        "best_bic": 100.0,
        "is_stuck": False,
        "wall_time": 1.0,
        "short_circuit": False,
    }
    judge.synthesize_for_persona.return_value = (
        "Default feedback only.",
        {"per_angle": [], "key_recommendations": []},
    )

    cfg = SimpleNamespace(
        judge=SimpleNamespace(
            mode="static",
            context=SimpleNamespace(attempted_models=False, performance=True, best_model_code=False, diagnostic=False),
            output=SimpleNamespace(persona_synthesis=False),
        ),
        clients={
            "explore": SimpleNamespace(llm=SimpleNamespace(feedback_guidance="Explore.")),
            "exploit": SimpleNamespace(llm=SimpleNamespace(feedback_guidance="Exploit.")),
        },
        centralized_model_generation=SimpleNamespace(enabled=False),
    )

    artifact = run_orchestrated_judge_pipeline(
        judge=judge,
        cfg=cfg,
        results_dir=tmp_path,
        iteration=0,
        run_idx=0,
        tag="",
        best_model=None,
        best_metric=None,
        recovery_failures=None,
        prev_had_success=True,
    )

    judge.synthesize_for_persona.assert_called_once()
    assert judge.synthesize_for_persona.call_args.kwargs["persona_name"] == "default"
    assert artifact.personas == ["default"]


def test_persona_synthesis_fanout_runs_for_cmg_generator_compatibility(tmp_path):
    """CMG should still synthesise feedback for the generator persona."""
    judge = MagicMock()
    judge.get_feedback_analysis.return_value = {
        "iteration": 0,
        "analysis_text": "analysis text",
        "trace": [],
        "full_trace": [],
        "best_bic": 100.0,
        "is_stuck": False,
        "wall_time": 1.0,
        "short_circuit": False,
    }
    judge.synthesize_for_persona.return_value = (
        "Generator-specific feedback.",
        {"per_angle": [], "key_recommendations": []},
    )

    cfg = SimpleNamespace(
        judge=SimpleNamespace(
            mode="static",
            context=SimpleNamespace(attempted_models=False, performance=True, best_model_code=False, diagnostic=False),
            output=SimpleNamespace(persona_synthesis=True),
        ),
        clients={
            "generator": SimpleNamespace(
                llm=SimpleNamespace(feedback_guidance="Focus on diversity.")
            ),
            "explore": SimpleNamespace(llm=SimpleNamespace(feedback_guidance="Explore.")),
        },
        centralized_model_generation=SimpleNamespace(
            enabled=True,
            generator_client="generator",
        ),
    )

    artifact = run_orchestrated_judge_pipeline(
        judge=judge,
        cfg=cfg,
        results_dir=tmp_path,
        iteration=0,
        run_idx=0,
        tag="",
        best_model=None,
        best_metric=None,
        recovery_failures=None,
        prev_had_success=True,
    )

    judge.synthesize_for_persona.assert_called_once()
    assert judge.synthesize_for_persona.call_args.kwargs["persona_name"] == "generator"
    assert artifact.synthesized_feedback == {"generator": "Generator-specific feedback."}


def test_persona_synthesis_uses_dict_backed_client_suffix(tmp_path):
    """Dict-backed client configs should feed persona suffixes into synthesis."""
    judge = MagicMock()
    judge.get_feedback_analysis.return_value = {
        "iteration": 0,
        "analysis_text": "analysis text",
        "trace": [],
        "full_trace": [],
        "best_bic": 100.0,
        "is_stuck": False,
        "wall_time": 1.0,
        "short_circuit": False,
    }
    judge.synthesize_for_persona.return_value = (
        "Generator-specific feedback.",
        {"per_angle": [], "key_recommendations": []},
    )

    cfg = SimpleNamespace(
        judge=SimpleNamespace(
            mode="static",
            context=SimpleNamespace(attempted_models=False, performance=True, best_model_code=False, diagnostic=False),
            output=SimpleNamespace(persona_synthesis=False),
        ),
        clients={
            "generator": {
                "llm": {
                    "feedback_guidance": "Focus on diverse candidate families."
                }
            }
        },
        centralized_model_generation=SimpleNamespace(
            enabled=True,
            generator_client="generator",
        ),
    )

    artifact = run_orchestrated_judge_pipeline(
        judge=judge,
        cfg=cfg,
        results_dir=tmp_path,
        iteration=0,
        run_idx=0,
        tag="",
        best_model=None,
        best_metric=None,
        recovery_failures=None,
        prev_had_success=True,
    )

    assert judge.synthesize_for_persona.call_args.kwargs["persona_name"] == "generator"
    assert judge.synthesize_for_persona.call_args.kwargs["persona_suffix"] == (
        "Focus on diverse candidate families."
    )
    assert artifact.synthesized_feedback == {"generator": "Generator-specific feedback."}


def test_empty_capabilities_produces_explicit_no_feedback_trace(tmp_path):
    """Empty capabilities should yield an explicit no-feedback shortcut artifact."""
    judge = MagicMock()
    judge.get_feedback_analysis.return_value = {
        "iteration": 0,
        "analysis_text": "",
        "trace": [],
        "full_trace": [],
        "best_bic": None,
        "is_stuck": False,
        "trajectory": [],
        "best_bic_str": "N/A",
        "wall_time": 0.0,
        "short_circuit": True,
        "no_capabilities": True,
    }
    judge.synthesize_for_persona.side_effect = AssertionError(
        "synthesis should be skipped for empty capabilities"
    )

    cfg = SimpleNamespace(
        judge=SimpleNamespace(
            mode="off",
            context=SimpleNamespace(attempted_models=False, performance=False, best_model_code=False, diagnostic=False),
            output=SimpleNamespace(persona_synthesis=False),
        ),
        clients={
            "explore": SimpleNamespace(llm=SimpleNamespace(feedback_guidance="Explore."))
        },
        centralized_model_generation=SimpleNamespace(enabled=False),
    )

    artifact = run_orchestrated_judge_pipeline(
        judge=judge,
        cfg=cfg,
        results_dir=tmp_path,
        iteration=0,
        run_idx=0,
        tag="",
        best_model=None,
        best_metric=None,
        recovery_failures=None,
        prev_had_success=True,
    )

    assert artifact.short_circuit is True
    assert artifact.no_substantive_feedback is True
    assert artifact.synthesized_feedback == {"default": ""}


def test_shortcut_path_does_not_write_competing_trace_schema(tmp_path):
    """Recovery shortcuts should avoid the legacy shortcut trace writer."""
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    (judge_dir / "iter0_run0.json").write_text(
        json.dumps(
            {
                "iteration": 0,
                "run_idx": 0,
                "tag": "",
                "timestamp": "2026-06-04T12:00:00+00:00",
                "tool_call_count": 1,
                "wall_time_seconds": 1.0,
                "best_bic": 91.2,
                "tool_call_trace": [],
                "full_trace": [],
                "metadata": {"source": "original"},
                "per_angle": [],
                "key_recommendations": [],
                "synthesized_feedback": {"default": "Existing feedback."},
                "personas": ["default"],
                "stuck_search": False,
                "short_circuit": False,
                "no_substantive_feedback": False,
                "random_feedback_only": False,
                "best_model_code_included": False,
            }
        ),
        encoding="utf-8",
    )

    judge = ToolUsingJudge.__new__(ToolUsingJudge)
    judge.mode = "llm"
    judge.results_dir = tmp_path
    judge.verbose = False
    judge.cfg = SimpleNamespace(centralized_model_generation=SimpleNamespace(enabled=False))
    judge._save_trace = MagicMock(side_effect=AssertionError("legacy shortcut trace used"))

    artifact = run_orchestrated_judge_pipeline(
        judge=judge,
        cfg=judge.cfg,
        results_dir=tmp_path,
        iteration=1,
        run_idx=0,
        tag="",
        best_model=None,
        best_metric=77.7,
        recovery_failures=[{"name": "candidate_b", "mean_r": 0.12, "per_param_r": {}}],
        prev_had_success=False,
    )

    persisted = json.loads((judge_dir / "iter1_run0.json").read_text(encoding="utf-8"))

    assert artifact.short_circuit is True
    assert set(persisted) >= {
        "iteration",
        "run_idx",
        "tag",
        "timestamp",
        "tool_call_count",
        "wall_time_seconds",
        "metadata",
        "synthesized_feedback",
        "personas",
    }
    assert "verdict" not in persisted


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
        judge=SimpleNamespace(
            mode="static",
            context=SimpleNamespace(attempted_models=False, performance=True, best_model_code=True, diagnostic=False),
            output=SimpleNamespace(persona_synthesis=False),
        ),
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


def test_build_feedback_artifact_omits_bic_when_best_code_has_no_performance_context():
    """Code-only best-model appendices should stay free of metric wording."""
    cfg = SimpleNamespace(
        judge=SimpleNamespace(
            context=SimpleNamespace(
                attempted_models=False,
                performance=False,
                best_model_code=True,
                diagnostic=False,
            )
        )
    )

    artifact = build_feedback_artifact(
        iteration=1,
        run_idx=0,
        tag="",
        analysis_data={
            "trace": [],
            "full_trace": [],
            "best_bic": 77.7,
            "wall_time": 2.0,
            "is_stuck": False,
        },
        synthesized_feedback={"default": "Try a more mechanistically distinct update."},
        verdict_payloads=[],
        best_model="def best_model(x):\n    return x",
        best_metric=77.7,
        include_best_model_code=False,
        cfg=cfg,
    )

    feedback_text = artifact.synthesized_feedback["default"]

    assert artifact.best_bic is None
    assert artifact.best_model_code_included is True
    assert "Best model code so far:" in feedback_text
    assert "BIC" not in feedback_text
    assert "metric" not in feedback_text.lower()
    assert "performance" not in feedback_text.lower()
    assert "def best_model(x):" in feedback_text


def test_build_feedback_artifact_disables_best_bic_for_attempted_only_context():
    """Attempted-only artifacts should not persist performance metadata."""
    cfg = SimpleNamespace(
        judge=SimpleNamespace(
            context=SimpleNamespace(
                attempted_models=True,
                performance=False,
                best_model_code=False,
                diagnostic=False,
            )
        )
    )

    artifact = build_feedback_artifact(
        iteration=1,
        run_idx=0,
        tag="",
        analysis_data={
            "trace": [],
            "full_trace": [],
            "best_bic": 77.7,
            "wall_time": 2.0,
            "is_stuck": False,
        },
        synthesized_feedback={"default": "Models attempted this iteration."},
        verdict_payloads=[],
        best_model=None,
        best_metric=None,
        include_best_model_code=False,
        cfg=cfg,
    )

    assert artifact.best_bic is None
    assert artifact.model_dump()["best_bic"] is None


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
    """Missing shared registry should not revive the local judge bypass."""
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


def test_orchestrator_uses_shared_orchestrated_runner(tmp_path, monkeypatch):
    """Distributed orchestration should use DuckDB evidence directly."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    original_glob = Path.glob

    def _guarded_glob(self, pattern):
        if "json" in pattern.lower() or "bics" in pattern.lower():
            raise AssertionError(f"unexpected JSON scan: {pattern}")
        return original_glob(self, pattern)

    monkeypatch.setattr(Path, "glob", _guarded_glob)

    local_store = ArtifactStore(
        results_dir,
        DiagnosticStore(results_dir / "diagnostics.duckdb"),
        inspection_output_enabled=False,
    )
    local_store.write_iteration_results(
        iteration=0,
        run_idx=0,
        tag="_client0",
        iteration_results=[
            {
                "function_name": "client0_model",
                "metric_value": 12.0,
                "param_names": [],
            }
        ],
    )
    local_store.diagnostic_store.close()
    peer_store = ArtifactStore(
        results_dir,
        DiagnosticStore(results_dir / "diagnostics_1.duckdb"),
        inspection_output_enabled=False,
    )
    peer_store.write_iteration_results(
        iteration=0,
        run_idx=0,
        tag="_client1",
        iteration_results=[
            {
                "function_name": "client1_model",
                "metric_value": 10.0,
                "param_names": [],
            }
        ],
    )
    peer_store.diagnostic_store.close()

    cfg = SimpleNamespace(
        task=SimpleNamespace(name="phase4_trace"),
        loop=SimpleNamespace(max_iterations=1),
        llm=SimpleNamespace(provider="openai", base_model="gpt-test"),
        judge=SimpleNamespace(
            barrier=SimpleNamespace(
                orchestrator_wait_seconds=1,
                retry_wait_seconds=1,
            ),
            mode="static",
            context=SimpleNamespace(attempted_models=False, performance=True, best_model_code=True, diagnostic=False),
            output=SimpleNamespace(persona_synthesis=False),
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
    mock_registry.wait_for_clients_complete.return_value = 2
    mock_registry.count_clients_with_models.return_value = 2
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

    with patch(
        "gecco.cli.run_judge_orchestrator._build_judge_store_from_duckdb_sources",
        wraps=run_judge_orchestrator_module._build_judge_store_from_duckdb_sources,
    ) as build_store_mock:
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
                                    "gecco.cli.run_judge_orchestrator.ToolUsingJudge",
                                    return_value=MagicMock(),
                                ) as judge_cls:
                                    with patch(
                                        "gecco.cli.run_judge_orchestrator.run_orchestrated_judge_pipeline",
                                        return_value=artifact,
                                    ) as runner:
                                        with patch("gecco.cli.run_judge_orchestrator.init_sentry"):
                                            run_orchestrator(
                                                config="test.yaml",
                                                results_dir=str(results_dir),
                                                n_clients=2,
                                            )

    runner.assert_called_once()
    build_store_mock.assert_called_once_with(results_dir)
    assert runner.call_args.kwargs["best_model"] == "def distributed_best_model(x):\n    return x"
    assert runner.call_args.kwargs["best_metric"] == 98.0
    diagnostic_store = judge_cls.call_args.kwargs["diagnostic_store"]
    model_rows = diagnostic_store.fetchall(
        "SELECT name FROM models WHERE split = 'train' ORDER BY name"
    )
    assert [row["name"] for row in model_rows] == ["client0_model", "client1_model"]
    assert mock_registry.set_judge_feedback.call_args.kwargs["synthesized_feedback"] == {
        "explore": "Focus on a different mechanism family."
    }


def test_run_orchestrator_does_not_mark_judge_failure_after_transient_duckdb_lock(
    tmp_path,
):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "diagnostics_client0.duckdb").touch()
    (results_dir / "diagnostics_client1.duckdb").touch()

    cfg = SimpleNamespace(
        task=SimpleNamespace(name="phase4_trace"),
        loop=SimpleNamespace(max_iterations=1),
        llm=SimpleNamespace(provider="openai", base_model="gpt-test"),
        judge=SimpleNamespace(
            barrier=SimpleNamespace(
                orchestrator_wait_seconds=1,
                retry_wait_seconds=1,
            ),
            mode="static",
            context=SimpleNamespace(
                attempted_models=False,
                performance=True,
                best_model_code=True,
                diagnostic=False,
            ),
            output=SimpleNamespace(persona_synthesis=False),
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
    mock_registry.wait_for_clients_complete.return_value = 2
    mock_registry.count_clients_with_models.return_value = 2
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
    store = MagicMock()
    store.import_from_source_db.side_effect = [
        RuntimeError("Could not set lock on diagnostics store"),
        None,
        None,
    ]

    with patch(
        "gecco.cli.run_judge_orchestrator._build_judge_store_from_duckdb_sources",
        wraps=run_judge_orchestrator_module._build_judge_store_from_duckdb_sources,
    ) as build_store_mock:
        with patch(
            "gecco.cli.run_judge_orchestrator.DiagnosticStore",
            return_value=store,
        ) as diagnostic_store_cls:
            with patch("gecco.cli.run_judge_orchestrator.time.sleep") as sleep_mock:
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
                                                        results_dir=str(results_dir),
                                                        n_clients=2,
                                                    )

    build_store_mock.assert_called_once_with(results_dir)
    diagnostic_store_cls.assert_called_once_with(results_dir / "diagnostics_unified.duckdb")
    assert store.import_from_source_db.call_count == 3
    sleep_mock.assert_called_once()
    runner.assert_called_once()
    mock_registry.set_judge_failure.assert_not_called()
    mock_registry.set_judge_feedback.assert_called_once()


def test_build_judge_store_retries_transient_duckdb_lock(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "diagnostics_client0.duckdb").touch()
    (results_dir / "diagnostics_client1.duckdb").touch()

    store = MagicMock()
    store.import_from_source_db.side_effect = [
        RuntimeError("Could not set lock on diagnostics store"),
        None,
        None,
    ]

    with patch(
        "gecco.cli.run_judge_orchestrator.DiagnosticStore",
        return_value=store,
    ) as diagnostic_store_cls:
        with patch("gecco.cli.run_judge_orchestrator.time.sleep") as sleep_mock:
            built_store = run_judge_orchestrator_module._build_judge_store_from_duckdb_sources(
                results_dir
            )

    assert built_store is store
    diagnostic_store_cls.assert_called_once_with(results_dir / "diagnostics_unified.duckdb")
    assert store.import_from_source_db.call_count == 3
    assert sleep_mock.call_count == 1


def test_build_judge_store_does_not_retry_non_lock_import_error(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "diagnostics_client0.duckdb").touch()

    store = MagicMock()
    store.import_from_source_db.side_effect = RuntimeError("schema mismatch")

    with patch(
        "gecco.cli.run_judge_orchestrator.DiagnosticStore",
        return_value=store,
    ) as diagnostic_store_cls:
        with patch("gecco.cli.run_judge_orchestrator.time.sleep") as sleep_mock:
            with pytest.raises(RuntimeError, match="schema mismatch"):
                run_judge_orchestrator_module._build_judge_store_from_duckdb_sources(
                    results_dir
                )

    diagnostic_store_cls.assert_called_once_with(results_dir / "diagnostics_unified.duckdb")
    store.import_from_source_db.assert_called_once_with(results_dir / "diagnostics_client0.duckdb")
    sleep_mock.assert_not_called()


def test_build_judge_store_gives_up_after_lock_retry_limit(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "diagnostics_client0.duckdb").touch()

    store = MagicMock()
    store.import_from_source_db.side_effect = RuntimeError("Conflicting lock on source db")

    with patch(
        "gecco.cli.run_judge_orchestrator.DiagnosticStore",
        return_value=store,
    ) as diagnostic_store_cls:
        with patch("gecco.cli.run_judge_orchestrator.time.sleep") as sleep_mock:
            with pytest.raises(RuntimeError, match="Conflicting lock on source db"):
                run_judge_orchestrator_module._build_judge_store_from_duckdb_sources(
                    results_dir
                )

    diagnostic_store_cls.assert_called_once_with(results_dir / "diagnostics_unified.duckdb")
    assert store.import_from_source_db.call_count == 5
    assert sleep_mock.call_count == 4
