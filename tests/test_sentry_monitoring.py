"""Tests for Sentry monitoring helpers, CLI fallback capture, and orchestrator capture."""

import fcntl
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import sentry_sdk

from gecco.artifacts import ArtifactStore
from gecco.candidate_generation import CandidateGenerator
from gecco.candidate_evaluation import CandidateEvaluator
from gecco.coordination import SharedRegistry
from gecco.sentry_init import (
    capture_coordination_error,
    capture_fit_error,
    capture_operational_error,
    capture_recovery_failed,
    flush_sentry_events,
    init_sentry,
)


class TestInitSentry:
    def test_returns_false_when_no_dsn(self):
        with patch.dict(os.environ, clear=True):
            assert init_sentry() is False

    def test_returns_true_when_dsn_set(self):
        with patch.dict(os.environ, {"SENTRY_DSN": "https://key@o0.ingest.sentry.io/project"}):
            with patch("gecco.sentry_init.sentry_sdk.init") as mock_init:
                result = init_sentry()
                assert result is True
                mock_init.assert_called_once()
                assert mock_init.call_args.kwargs["environment"] == "production"

    def test_config_environment_cannot_override_production(self):
        cfg = SimpleNamespace(sentry=SimpleNamespace(environment="development"))
        with patch.dict(os.environ, {"SENTRY_DSN": "https://key@o0.ingest.sentry.io/project"}):
            with patch("gecco.sentry_init.sentry_sdk.init") as mock_init:
                result = init_sentry(cfg=cfg)
                assert result is True
                assert mock_init.call_args.kwargs["environment"] == "production"

    def test_flush_sentry_events_calls_sdk_flush(self):
        with patch("gecco.sentry_init.sentry_sdk.flush") as mock_flush:
            flush_sentry_events(timeout=3.5)
            mock_flush.assert_called_once_with(timeout=3.5)

    def test_flush_sentry_events_ignores_flush_failure(self):
        with patch("gecco.sentry_init.sentry_sdk.flush", side_effect=RuntimeError("flush failed")):
            flush_sentry_events(timeout=1.0)

    def test_init_without_config_does_not_raise(self):
        with patch.dict(os.environ, {"SENTRY_DSN": "https://key@o0.ingest.sentry.io/project"}):
            with patch("gecco.sentry_init.sentry_sdk.init"):
                try:
                    init_sentry()
                except Exception:
                    pytest.fail("init_sentry() without config raised unexpectedly")


class TestCaptureOperationalError:
    def test_capture_helpers_ignore_sentry_capture_failure(self):
        with patch.object(sentry_sdk, "capture_exception", side_effect=RuntimeError("sentry down")) as mock_capture:
            error = RuntimeError("primary failure")

            capture_operational_error(error, component="cli", operation="dispatch")
            capture_coordination_error(error=error, operation="write")
            capture_fit_error(iteration=1, model_name="model_a", error=error)
            capture_recovery_failed(iteration=2, model_name="model_b", error=error)

            assert mock_capture.call_count == 4

    def test_sets_context_and_fingerprint(self):
        with patch.object(sentry_sdk, "capture_exception") as mock_capture:
            error = ValueError("test error")
            capture_operational_error(
                error,
                component="test_component",
                operation="test_operation",
                severity="error",
                extra_field="extra_value",
            )
            mock_capture.assert_called_once()
            args, kwargs = mock_capture.call_args
            assert args[0] is error
            assert kwargs["fingerprint"] == [
                "operational-error",
                "test_component",
                "test_operation",
            ]
            assert kwargs["extras"]["component"] == "test_component"
            assert kwargs["extras"]["operation"] == "test_operation"
            assert kwargs["extras"]["severity"] == "error"
            assert kwargs["extras"]["extra_field"] == "extra_value"

    def test_accepts_different_severity(self):
        with patch.object(sentry_sdk, "capture_exception") as mock_capture:
            error = RuntimeError("warning test")
            capture_operational_error(
                error,
                component="cli",
                operation="handler_dispatch",
                severity="warning",
            )
            _, kwargs = mock_capture.call_args
            assert kwargs["extras"]["severity"] == "warning"

    def test_does_not_swallow_exception(self):
        with patch.object(sentry_sdk, "capture_exception"):
            error = ValueError("test")
            capture_operational_error(error, component="x", operation="y")
            # If we reach here, the helper didn't raise
            assert True


class TestExistingCaptureHelpers:
    def test_capture_fit_error_uses_expected_failure_context(self):
        with patch.object(sentry_sdk, "capture_exception") as mock_capture:
            error = Exception("fit failed")
            capture_fit_error(iteration=3, model_name="test_model", error=error, run=1)
            _, kwargs = mock_capture.call_args
            assert kwargs["fingerprint"] == ["fit-error", "3", "test_model"]
            assert kwargs["extras"].get("component") == "candidate_evaluation"
            assert kwargs["extras"].get("operation") == "fit"
            assert kwargs["extras"].get("severity") == "expected"
            assert kwargs["extras"].get("failure_class") == "expected_search_domain"
            assert kwargs["extras"].get("iteration") == 3
            assert kwargs["extras"].get("model_name") == "test_model"
            assert kwargs["extras"].get("run") == 1
            # No raw model code attached
            assert "code" not in kwargs["extras"]
            assert "prompt" not in kwargs["extras"]

    def test_capture_recovery_failed_expected_context(self):
        with patch.object(sentry_sdk, "capture_exception") as mock_capture:
            error = Exception("recovery failed")
            capture_recovery_failed(iteration=1, model_name="recovery_model", error=error)
            _, kwargs = mock_capture.call_args
            assert kwargs["fingerprint"] == ["recovery-failed", "1", "recovery_model"]
            assert kwargs["extras"].get("component") == "candidate_evaluation"
            assert kwargs["extras"].get("operation") == "recovery"
            assert kwargs["extras"].get("severity") == "expected"
            assert kwargs["extras"].get("failure_class") == "expected_search_domain"
            assert kwargs["extras"].get("iteration") == 1
            assert kwargs["extras"].get("model_name") == "recovery_model"

    def test_capture_coordination_error_sets_fingerprint(self):
        with patch.object(sentry_sdk, "capture_exception") as mock_capture:
            error = Exception("db error")
            capture_coordination_error(error=error, operation="update")
            _, kwargs = mock_capture.call_args
            assert kwargs["fingerprint"] == ["coordination-error", "update"]
            assert kwargs["extras"].get("operation") == "update"


class TestCliFallbackCapture:
    def test_cli_initializes_sentry_before_dispatch(self):
        from gecco.cli import main

        with patch("gecco.cli.init_sentry") as mock_init:
            with patch("gecco.cli.build_parser") as mock_build:
                mock_parser = MagicMock()
                mock_args = MagicMock()
                mock_args.handler.return_value = 0
                mock_parser.parse_args.return_value = mock_args

                def build_parser_side_effect():
                    assert mock_init.called
                    return mock_parser

                mock_build.side_effect = build_parser_side_effect

                result = main(["dummy"])

        assert result == 0
        mock_init.assert_called_once_with(component="cli", operation="startup")

    def test_cli_reports_uncaught_exception(self):
        with patch.object(sentry_sdk, "capture_exception") as mock_capture:
            error = RuntimeError("handler failure")
            from gecco.cli import main
            with patch("gecco.cli.init_sentry"):
                with patch("gecco.cli.build_parser") as mock_build:
                    mock_parser = MagicMock()
                    mock_args = MagicMock()
                    mock_args.handler.side_effect = error
                    mock_parser.parse_args.return_value = mock_args
                    mock_build.return_value = mock_parser
                    with pytest.raises(RuntimeError):
                        main(["dummy"])
            mock_capture.assert_called_once()
            args, kwargs = mock_capture.call_args
            assert args[0] is error
            assert kwargs["fingerprint"] == [
                "operational-error",
                "cli",
                "handler_dispatch",
            ]

    def test_cli_does_not_report_system_exit_zero(self):
        with patch.object(sentry_sdk, "capture_exception") as mock_capture:
            from gecco.cli import main
            with patch("gecco.cli.init_sentry"):
                with patch("gecco.cli.build_parser") as mock_build:
                    mock_parser = MagicMock()
                    mock_args = MagicMock()
                    mock_args.handler.side_effect = SystemExit(0)
                    mock_parser.parse_args.return_value = mock_args
                    mock_build.return_value = mock_parser
                    with pytest.raises(SystemExit) as exc_info:
                        main(["dummy"])
                    assert exc_info.value.code == 0
            mock_capture.assert_not_called()

    def test_cli_reports_nonzero_system_exit(self):
        with patch.object(sentry_sdk, "capture_exception") as mock_capture:
            from gecco.cli import main
            with patch("gecco.cli.init_sentry"):
                with patch("gecco.cli.build_parser") as mock_build:
                    mock_parser = MagicMock()
                    mock_args = MagicMock()
                    mock_args.handler.side_effect = SystemExit(1)
                    mock_parser.parse_args.return_value = mock_args
                    mock_build.return_value = mock_parser
                    with pytest.raises(SystemExit) as exc_info:
                        main(["dummy"])
                    assert exc_info.value.code == 1
            mock_capture.assert_called_once()

    def test_cli_does_not_report_system_exit_none(self):
        with patch.object(sentry_sdk, "capture_exception") as mock_capture:
            from gecco.cli import main
            with patch("gecco.cli.init_sentry"):
                with patch("gecco.cli.build_parser") as mock_build:
                    mock_parser = MagicMock()
                    mock_args = MagicMock()
                    mock_args.handler.side_effect = SystemExit(None)
                    mock_parser.parse_args.return_value = mock_args
                    mock_build.return_value = mock_parser
                    with pytest.raises(SystemExit):
                        main(["dummy"])
            mock_capture.assert_not_called()


def _mock_orchestrator_config():
    """Build a minimal mock config for orchestrator tests."""
    return SimpleNamespace(
        task=SimpleNamespace(name="test_task"),
        llm=SimpleNamespace(provider="openai", base_model="gpt-4o"),
        data=SimpleNamespace(
            path="test.csv",
            input_columns=["x"],
            splits={"train": 0.8, "test": 0.2},
            id_column="subject",
            narrative_template="test template {x}",
            data2text_function="identity",
        ),
        loop=SimpleNamespace(max_iterations=3, n_clients=2),
        judge=SimpleNamespace(
            barrier=SimpleNamespace(
                orchestrator_wait_seconds=10,
                retry_wait_seconds=5,
            )
        ),
        evaluation=SimpleNamespace(fit_type="group"),
        sentry=None,
        centralized_model_generation=None,
    )


class TestOrchestratorCapture:
    """Tests for judge orchestrator Sentry capture on critical failures."""

    def test_reports_duckdb_load_failure_and_halts(self):
        """DuckDB load failure: capture helper called, registry failure written,
        judge pipeline not invoked, command returns non-zero."""
        mm = MagicMock()
        with (
            patch("gecco.cli.run_judge_orchestrator.load_config") as mock_load_config,
            patch("gecco.cli.run_judge_orchestrator.load_llm", return_value=(mm, mm)),
            patch("gecco.cli.run_judge_orchestrator.load_data", return_value=mm),
            patch("gecco.cli.run_judge_orchestrator.split_by_participant") as mock_split,
            patch("gecco.cli.run_judge_orchestrator.get_data2text_function", return_value=MagicMock()),
            patch("gecco.cli.run_judge_orchestrator.PromptBuilderWrapper"),
            patch("gecco.cli.run_judge_orchestrator.SharedRegistry") as mock_registry_cls,
            patch("gecco.cli.run_judge_orchestrator.configure_temp_dirs"),
            patch("gecco.cli.run_judge_orchestrator.init_sentry"),
            patch("gecco.cli.run_judge_orchestrator.run_orchestrated_judge_pipeline") as mock_pipeline,
            patch.object(sentry_sdk, "capture_exception") as mock_capture,
        ):
            cfg = _mock_orchestrator_config()
            mock_load_config.return_value = cfg
            mock_split.return_value = {"prompt": mm}
            mock_registry = MagicMock()
            mock_registry.wait_for_clients_complete.return_value = 2
            mock_registry.count_clients_with_models.return_value = 1
            mock_registry_cls.return_value = mock_registry
            mock_capture.side_effect = RuntimeError("sentry down")

            with patch(
                "gecco.cli.run_judge_orchestrator._build_judge_store_from_duckdb_sources",
                side_effect=RuntimeError("duckdb load exploded"),
            ):
                from gecco.cli.run_judge_orchestrator import run_orchestrator

                result = run_orchestrator(
                    config="dummy.yaml",
                    results_dir=str(Path("/tmp/opencode/test_orch_duckdb")),
                )

                assert result == 1
                mock_registry.set_judge_failure.assert_called_once()
                failure_call = mock_registry.set_judge_failure.call_args
                assert "Diagnostic DuckDB load failed" in str(failure_call)
                mock_pipeline.assert_not_called()

            mock_capture.assert_called_once()
            _, kwargs = mock_capture.call_args
            assert kwargs["fingerprint"] == [
                "operational-error",
                "judge_orchestrator",
                "load_duckdb_store",
            ]

    def test_reports_final_judge_failure_after_retries(self):
        """Final judge failure after retries: capture helper called once,
        registry failure written, command returns non-zero."""
        expected_attempts = 3
        mm = MagicMock()
        call_order = []

        def _capture_side_effect(*args, **kwargs):
            call_order.append("capture")
            raise RuntimeError("sentry down")

        def _flush_side_effect(*args, **kwargs):
            call_order.append("flush")

        with (
            patch("gecco.cli.run_judge_orchestrator.load_config") as mock_load_config,
            patch("gecco.cli.run_judge_orchestrator.load_llm", return_value=(mm, mm)),
            patch("gecco.cli.run_judge_orchestrator.load_data", return_value=mm),
            patch("gecco.cli.run_judge_orchestrator.split_by_participant") as mock_split,
            patch("gecco.cli.run_judge_orchestrator.get_data2text_function", return_value=MagicMock()),
            patch("gecco.cli.run_judge_orchestrator.PromptBuilderWrapper"),
            patch("gecco.cli.run_judge_orchestrator.SharedRegistry") as mock_registry_cls,
            patch("gecco.cli.run_judge_orchestrator.configure_temp_dirs"),
            patch("gecco.cli.run_judge_orchestrator.init_sentry"),
            patch("gecco.cli.run_judge_orchestrator.DiagnosticStore") as mock_ds,
            patch("gecco.cli.run_judge_orchestrator._build_judge_store_from_duckdb_sources") as mock_build,
            patch("gecco.cli.run_judge_orchestrator.run_orchestrated_judge_pipeline") as mock_pipeline,
            patch("gecco.cli.run_judge_orchestrator.ToolUsingJudge"),
            patch.object(sentry_sdk, "capture_exception") as mock_capture,
            patch("gecco.cli.run_judge_orchestrator.flush_sentry_events") as mock_flush,
        ):
            cfg = _mock_orchestrator_config()
            mock_load_config.return_value = cfg
            mock_split.return_value = {"prompt": mm}
            mock_registry = MagicMock()
            mock_registry.wait_for_clients_complete.return_value = 2
            mock_registry.count_clients_with_models.return_value = 1
            mock_registry.read.return_value = {"global_best": None}
            mock_registry_cls.return_value = mock_registry
            mock_build.return_value = MagicMock()
            mock_pipeline.side_effect = RuntimeError("judge failed every time")
            mock_ds.return_value = MagicMock()
            mock_capture.side_effect = _capture_side_effect
            mock_flush.side_effect = _flush_side_effect

            from gecco.cli.run_judge_orchestrator import run_orchestrator

            result = run_orchestrator(
                config="dummy.yaml",
                results_dir=str(Path("/tmp/opencode/test_orch_final_judge")),
            )

            assert result == 1
            assert mock_pipeline.call_count == expected_attempts
            mock_registry.set_judge_failure.assert_called_once()
            mock_registry.set_judge_feedback.assert_not_called()
            failure_call = mock_registry.set_judge_failure.call_args
            assert "judge failed every time" in str(failure_call)

            mock_capture.assert_called_once()
            _, kwargs = mock_capture.call_args
            assert kwargs["fingerprint"] == [
                "operational-error",
                "judge_orchestrator",
                "final_judge_retry",
            ]
            assert kwargs["extras"]["attempt"] == 3
            assert kwargs["extras"]["max_attempts"] == 3
            assert kwargs["extras"]["exception_type"] == "RuntimeError"
            mock_flush.assert_called_once()
            assert call_order == ["capture", "flush"]

    def test_success_path_does_not_flush_or_capture(self):
        mm = MagicMock()
        artifact = SimpleNamespace(synthesized_feedback={"default": "ok"})
        with (
            patch("gecco.cli.run_judge_orchestrator.load_config") as mock_load_config,
            patch("gecco.cli.run_judge_orchestrator.load_llm", return_value=(mm, mm)),
            patch("gecco.cli.run_judge_orchestrator.load_data", return_value=mm),
            patch("gecco.cli.run_judge_orchestrator.split_by_participant") as mock_split,
            patch("gecco.cli.run_judge_orchestrator.get_data2text_function", return_value=MagicMock()),
            patch("gecco.cli.run_judge_orchestrator.PromptBuilderWrapper"),
            patch("gecco.cli.run_judge_orchestrator.SharedRegistry") as mock_registry_cls,
            patch("gecco.cli.run_judge_orchestrator.configure_temp_dirs"),
            patch("gecco.cli.run_judge_orchestrator.init_sentry"),
            patch("gecco.cli.run_judge_orchestrator.DiagnosticStore") as mock_ds,
            patch("gecco.cli.run_judge_orchestrator._build_judge_store_from_duckdb_sources") as mock_build,
            patch("gecco.cli.run_judge_orchestrator.run_orchestrated_judge_pipeline", return_value=artifact) as mock_pipeline,
            patch("gecco.cli.run_judge_orchestrator.ToolUsingJudge"),
            patch.object(sentry_sdk, "capture_exception") as mock_capture,
            patch("gecco.cli.run_judge_orchestrator.flush_sentry_events") as mock_flush,
        ):
            cfg = _mock_orchestrator_config()
            cfg.loop.max_iterations = 1
            mock_load_config.return_value = cfg
            mock_split.return_value = {"prompt": mm}
            mock_registry = MagicMock()
            mock_registry.wait_for_clients_complete.return_value = 2
            mock_registry.count_clients_with_models.return_value = 1
            mock_registry.read.return_value = {"global_best": None}
            mock_registry_cls.return_value = mock_registry
            mock_build.return_value = MagicMock()
            mock_ds.return_value = MagicMock()

            from gecco.cli.run_judge_orchestrator import run_orchestrator

            result = run_orchestrator(
                config="dummy.yaml",
                results_dir=str(Path("/tmp/opencode/test_orch_success")),
            )

            assert result is None
            mock_capture.assert_not_called()
            mock_flush.assert_not_called()
            mock_pipeline.assert_called_once()

    def test_duckdb_load_failure_skips_judge_pipeline(self):
        """Judge pipeline is not invoked after a DuckDB load failure."""
        mm = MagicMock()
        with (
            patch("gecco.cli.run_judge_orchestrator.load_config") as mock_load_config,
            patch("gecco.cli.run_judge_orchestrator.load_llm", return_value=(mm, mm)),
            patch("gecco.cli.run_judge_orchestrator.load_data", return_value=mm),
            patch("gecco.cli.run_judge_orchestrator.split_by_participant") as mock_split,
            patch("gecco.cli.run_judge_orchestrator.get_data2text_function", return_value=MagicMock()),
            patch("gecco.cli.run_judge_orchestrator.PromptBuilderWrapper"),
            patch("gecco.cli.run_judge_orchestrator.SharedRegistry") as mock_registry_cls,
            patch("gecco.cli.run_judge_orchestrator.configure_temp_dirs"),
            patch("gecco.cli.run_judge_orchestrator.init_sentry"),
            patch("gecco.cli.run_judge_orchestrator.run_orchestrated_judge_pipeline") as mock_pipeline,
        ):
            cfg = _mock_orchestrator_config()
            mock_load_config.return_value = cfg
            mock_split.return_value = {"prompt": mm}
            mock_registry = MagicMock()
            mock_registry.wait_for_clients_complete.return_value = 2
            mock_registry.count_clients_with_models.return_value = 1
            mock_registry_cls.return_value = mock_registry

            with patch(
                "gecco.cli.run_judge_orchestrator._build_judge_store_from_duckdb_sources",
                side_effect=RuntimeError("duckdb load exploded"),
            ):
                from gecco.cli.run_judge_orchestrator import run_orchestrator

                run_orchestrator(
                    config="dummy.yaml",
                    results_dir=str(Path("/tmp/opencode/test_orch_skip_judge")),
                )

            mock_pipeline.assert_not_called()


class TestOptionalDiagnosticCapture:
    """Tests for optional diagnostic failure handling in test evaluation."""

    def test_optional_diagnostic_failure_reports_and_continues(self):
        """Optional diagnostic failure should not block the returned result."""
        with (
            patch("gecco.cli.run_test_evaluation.run_fit") as mock_run_fit,
            patch(
                "gecco.offline_evaluation.individual_differences.evaluate_individual_differences"
            ) as mock_id,
        ):
            mock_run_fit.return_value = {
                "metric_name": "BIC",
                "metric_value": 100.0,
                "mean_nll": 2.5,
                "eval_metrics": [95.0, 105.0],
                "per_participant_nll": [2.0, 3.0],
                "param_names": ["alpha"],
                "parameter_values": None,
            }
            mock_id.side_effect = RuntimeError("individual differences computation failed")

            from gecco.cli.run_test_evaluation import fit_one_on_test

            candidate = {
                "function_name": "test_model",
                "code": "def test_model(x, params):\n    return 0.0",
                "val_mean_nll": 3.0,
            }
            cfg = _mock_orchestrator_config()
            cfg.individual_differences_eval = True

            result = fit_one_on_test(
                candidate=candidate,
                df_test=MagicMock(),
                cfg=cfg,
                id_eval_data=MagicMock(),
            )

            assert result is not None
            assert result["model_name"] == "test_model"

    def test_optional_diagnostic_failure_returns_normally_without_id_data(self):
        """Without id_eval_data, no capture call, normal return."""
        with (
            patch("gecco.cli.run_test_evaluation.run_fit") as mock_run_fit,
            patch.object(sentry_sdk, "capture_exception") as mock_capture,
        ):
            mock_run_fit.return_value = {
                "metric_name": "BIC",
                "metric_value": 100.0,
                "mean_nll": 2.5,
                "eval_metrics": [95.0, 105.0],
                "per_participant_nll": [2.0, 3.0],
                "param_names": ["alpha"],
                "parameter_values": None,
            }

            from gecco.cli.run_test_evaluation import fit_one_on_test

            candidate = {
                "function_name": "test_model",
                "code": "def test_model(x, params):\n    return 0.0",
                "val_mean_nll": 3.0,
            }
            cfg = _mock_orchestrator_config()

            result = fit_one_on_test(
                candidate=candidate,
                df_test=MagicMock(),
                cfg=cfg,
                id_eval_data=None,
            )

            assert result is not None
            assert result["test_individual_differences"] is None
            mock_capture.assert_not_called()

    def test_candidate_ppc_failure_reports_and_continues_when_capture_fails(self):
        with (
            patch("gecco.offline_evaluation.fit_generated_models.run_fit") as mock_run_fit,
            patch("gecco.offline_evaluation.utils.build_model_spec") as mock_build_spec,
            patch("gecco.offline_evaluation.ppc._get_participants") as mock_get_participants,
            patch("gecco.offline_evaluation.ppc.compute_ppc", side_effect=RuntimeError("ppc failed")),
            patch.object(sentry_sdk, "capture_exception", side_effect=RuntimeError("sentry down")) as mock_capture,
        ):
            mock_run_fit.return_value = {
                "metric_name": "BIC",
                "metric_value": 10.0,
                "mean_nll": 1.0,
                "eval_metrics": [],
                "per_participant_nll": [],
                "param_names": ["alpha"],
                "parameter_values": [{"alpha": 0.1}],
            }
            mock_build_spec.return_value = SimpleNamespace(param_names=["alpha"])
            mock_get_participants.return_value = (None, ["p1", "p2"])

            evaluator = CandidateEvaluator(MagicMock())
            cfg = _mock_orchestrator_config()
            result, should_stop = evaluator.fit_candidate_model(
                model_dict={
                    "func_name": "test_model",
                    "name": "test_model",
                    "code": "def test_model(x):\n    return x",
                    "parameters": [{"name": "alpha"}],
                },
                model_idx=0,
                n_models=1,
                it=1,
                run_idx=0,
                tag="tag",
                model_file=Path("/tmp/opencode/test_model.py"),
                baseline_bic=None,
                df=MagicMock(),
                cfg=cfg,
                ppc_enabled=True,
                ppc_simulator=MagicMock(),
                block_residuals_enabled=False,
                set_activity=None,
            )

            assert should_stop is False
            assert result is not None
            assert "ppc" not in result
            mock_capture.assert_called_once()

    def test_candidate_block_residuals_failure_reports_and_continues_when_capture_fails(self):
        with (
            patch("gecco.offline_evaluation.fit_generated_models.run_fit") as mock_run_fit,
            patch("gecco.offline_evaluation.utils.build_model_spec") as mock_build_spec,
            patch("gecco.offline_evaluation.ppc.compute_block_residuals", side_effect=RuntimeError("block residuals failed")),
            patch.object(sentry_sdk, "capture_exception", side_effect=RuntimeError("sentry down")) as mock_capture,
        ):
            mock_run_fit.return_value = {
                "metric_name": "BIC",
                "metric_value": 10.0,
                "mean_nll": 1.0,
                "eval_metrics": [],
                "per_participant_nll": [],
                "param_names": ["alpha"],
                "parameter_values": [{"alpha": 0.1}],
            }
            mock_build_spec.return_value = SimpleNamespace(param_names=["alpha"])

            evaluator = CandidateEvaluator(MagicMock())
            cfg = _mock_orchestrator_config()
            result, should_stop = evaluator.fit_candidate_model(
                model_dict={
                    "func_name": "test_model",
                    "name": "test_model",
                    "code": "def test_model(x):\n    return x",
                    "parameters": [{"name": "alpha"}],
                },
                model_idx=0,
                n_models=1,
                it=1,
                run_idx=0,
                tag="tag",
                model_file=Path("/tmp/opencode/test_model.py"),
                baseline_bic=None,
                df=MagicMock(),
                cfg=cfg,
                ppc_enabled=False,
                block_residuals_enabled=True,
                block_residuals_n_blocks=5,
                set_activity=None,
            )

            assert should_stop is False
            assert result is not None
            assert "block_residuals" not in result
            mock_capture.assert_called_once()

    def test_shared_registry_callback_exception_rolls_back_and_closes_when_capture_fails(self, tmp_path):
        registry = SharedRegistry(tmp_path / "shared_registry.duckdb")

        def _seed(connection):
            connection.execute("CREATE TABLE IF NOT EXISTS tx_test (value INTEGER)")
            connection.execute("DELETE FROM tx_test")

        registry._with_connection(write=True, operation="seed", callback=_seed)

        with patch.object(sentry_sdk, "capture_exception", side_effect=RuntimeError("sentry down")) as mock_capture:
            def _failing_callback(connection):
                connection.execute("INSERT INTO tx_test VALUES (1)")
                raise RuntimeError("callback failed")

            with pytest.raises(RuntimeError, match="callback failed"):
                registry._with_connection(
                    write=True,
                    operation="callback_failure",
                    callback=_failing_callback,
                )

        count = registry._with_connection(
            write=False,
            operation="verify_rollback",
            callback=lambda connection: connection.execute(
                "SELECT COUNT(*) FROM tx_test"
            ).fetchone()[0],
        )
        assert count == 0

        with registry.lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        assert mock_capture.call_count == 1

    def test_test_evaluation_diagnostic_store_write_failure_continues(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "shared_registry.duckdb").touch()

        with (
            patch("gecco.cli.run_test_evaluation.load_config") as mock_load_config,
            patch("gecco.cli.run_test_evaluation.SharedRegistry.open_existing") as mock_open_existing,
            patch("gecco.cli.run_test_evaluation.load_splits", return_value=[1, 2]),
            patch("gecco.cli.run_test_evaluation.collect_candidates", return_value=[{"function_name": "test_model", "code": "code", "val_mean_nll": 1.0}]),
            patch("gecco.cli.run_test_evaluation.fit_one_on_test", return_value={
                "model_name": "test_model",
                "val_nll": 1.0,
                "test_mean_BIC": 2.0,
                "test_mean_NLL": 3.0,
                "test_individual_BIC": [],
                "test_individual_NLL": [],
                "test_individual_differences": None,
            }),
            patch("gecco.diagnostic_store.store.DiagnosticStore") as mock_ds,
            patch("gecco.cli.run_test_evaluation.configure_temp_dirs"),
        ):
            cfg = _mock_orchestrator_config()
            cfg.evaluation.n_test_models = 1
            mock_load_config.return_value = cfg
            mock_registry = MagicMock()
            mock_registry.read.return_value = {"baseline": None}
            mock_open_existing.return_value = mock_registry
            mock_store = MagicMock()
            mock_store.write_top_model_test.side_effect = RuntimeError("diagnostic store write failed")
            mock_ds.return_value = mock_store

            from gecco.cli.run_test_evaluation import run_test_evaluation

            result = run_test_evaluation(
                config="dummy.yaml",
                results_dir=str(results_dir),
                write_store=True,
            )

            assert result is None
            mock_store.write_top_model_test.assert_called_once()


class TestCandidateGenerationCapture:
    def _build_generator(self):
        artifact_store = MagicMock(spec=ArtifactStore)
        return CandidateGenerator(artifact_store)

    def _build_runtime_inputs(self):
        cfg = SimpleNamespace(
            clients={},
            llm=SimpleNamespace(
                provider="openai",
                structured_output=True,
                analysis_scratchpad=True,
                reviewer=SimpleNamespace(enabled=False),
            ),
            validation=SimpleNamespace(retry_limit=1),
        )
        cmg_cfg = SimpleNamespace(n_models=1)
        prompt_builder = SimpleNamespace(
            build_input_prompt=MagicMock(return_value="build one model"),
        )
        shared_registry = MagicMock()
        return cfg, cmg_cfg, prompt_builder, shared_registry

    def test_candidate_generation_failure_reports_status_and_reraises(self):
        generator = self._build_generator()
        cfg, cmg_cfg, prompt_builder, shared_registry = self._build_runtime_inputs()
        failure = RuntimeError("generation failed")

        with patch.object(sentry_sdk, "capture_exception") as mock_capture:
            with pytest.raises(RuntimeError, match="generation failed"):
                generator.generate_iteration(
                    iteration=7,
                    run_idx=3,
                    feedback="keep it simple",
                    cmg_cfg=cmg_cfg,
                    tag="_demo",
                    client_id=42,
                    naive_enabled=False,
                    prompt_builder=prompt_builder,
                    generate_text=MagicMock(side_effect=failure),
                    model=object(),
                    tokenizer=object(),
                    cfg=cfg,
                    shared_registry=shared_registry,
                    participant=None,
                    set_activity=None,
                )

        mock_capture.assert_called_once()
        args, kwargs = mock_capture.call_args
        assert args[0] is failure
        assert kwargs["fingerprint"] == [
            "operational-error",
            "candidate_generation",
            "generate_iteration",
        ]
        extras = kwargs["extras"]
        assert extras["component"] == "candidate_generation"
        assert extras["operation"] == "generate_iteration"
        assert extras["severity"] == "error"
        assert extras["iteration"] == 7
        assert extras["run"] == 3
        assert extras["client_id"] == "42"
        assert extras["tag"] == "_demo"
        assert shared_registry.set_generator_status.call_count == 1
        shared_registry.set_generator_status.assert_called_once_with(
            iteration=7,
            client_id=42,
            status="failed",
            n_candidates=0,
            error="generation failed",
        )
        for forbidden_key in (
            "feedback",
            "code_text",
            "parsed_models",
            "prompt",
            "generated_code",
            "model",
            "tokenizer",
            "cfg",
        ):
            assert forbidden_key not in extras

    def test_candidate_generation_reporting_failure_does_not_mask_original_failure(self):
        generator = self._build_generator()
        cfg, cmg_cfg, prompt_builder, shared_registry = self._build_runtime_inputs()
        failure = RuntimeError("generation failed")

        with patch.object(sentry_sdk, "capture_exception", side_effect=RuntimeError("sentry down")) as mock_capture:
            with pytest.raises(RuntimeError, match="generation failed"):
                generator.generate_iteration(
                    iteration=11,
                    run_idx=4,
                    feedback="keep it simple",
                    cmg_cfg=cmg_cfg,
                    tag="_demo",
                    client_id=42,
                    naive_enabled=False,
                    prompt_builder=prompt_builder,
                    generate_text=MagicMock(side_effect=failure),
                    model=object(),
                    tokenizer=object(),
                    cfg=cfg,
                    shared_registry=shared_registry,
                    participant=None,
                    set_activity=None,
                )

        mock_capture.assert_called_once()
        shared_registry.set_generator_status.assert_called_once_with(
            iteration=11,
            client_id=42,
            status="failed",
            n_candidates=0,
            error="generation failed",
        )
