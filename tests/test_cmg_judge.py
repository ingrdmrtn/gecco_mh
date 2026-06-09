"""Tests for CMG judge orchestrator feedback keying."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_cfg(cmg_enabled=True):
    cmg_cfg = SimpleNamespace(
        enabled=cmg_enabled,
        generator_client="generator",
        n_models=2,
    )
    return SimpleNamespace(
        task=SimpleNamespace(name="test_judge"),
        loop=SimpleNamespace(max_iterations=1),
        llm=SimpleNamespace(provider="openrouter", base_model="test-model"),
        judge=SimpleNamespace(
            capabilities=["performance_summary"],
            barrier=SimpleNamespace(
                orchestrator_wait_seconds=1,
                retry_wait_seconds=1,
            ),
            ppc=SimpleNamespace(enabled=False),
        ),
        data=SimpleNamespace(
            path="dummy.csv",
            input_columns=["choice_1"],
            id_column="participant",
            splits={"prompt": "[1:2]"},
            data2text_function="narrative",
            narrative_template="trial {choice_1}",
        ),
        clients=SimpleNamespace(
            generator=SimpleNamespace(
                llm=SimpleNamespace(feedback_guidance="Focus on diversity.")
            )
        ),
        centralized_model_generation=cmg_cfg,
    )


def test_cmg_short_circuit_feedback_keyed_by_generator():
    """In CMG mode, short-circuit judge feedback must be keyed by generator_client."""
    from gecco.cli.run_judge_orchestrator import run_orchestrator

    cfg = _make_cfg(cmg_enabled=True)

    mock_registry = MagicMock()
    mock_registry.wait_for_clients_complete.return_value = 2
    mock_registry.count_clients_with_models.return_value = 1
    mock_registry.read.return_value = {"global_best": {}}

    artifact = SimpleNamespace(
        synthesized_feedback={
            "generator": "Search is stuck. Revert to simpler models."
        }
    )

    with patch("gecco.cli.run_judge_orchestrator.load_config", return_value=cfg):
        with patch("gecco.cli.run_judge_orchestrator.SharedRegistry", return_value=mock_registry):
            with patch("gecco.cli.run_judge_orchestrator.load_llm", return_value=(None, None)):
                with patch("gecco.cli.run_judge_orchestrator.load_data", return_value=MagicMock()):
                    with patch("gecco.cli.run_judge_orchestrator.split_by_participant", return_value={"prompt": MagicMock()}):
                        with patch("gecco.cli.run_judge_orchestrator.get_data2text_function", return_value=lambda *a, **k: "data text"):
                            with patch("gecco.cli.run_judge_orchestrator._build_judge_store_from_duckdb_sources", return_value=MagicMock()):
                                with patch("gecco.cli.run_judge_orchestrator.ToolUsingJudge", return_value=MagicMock()):
                                    with patch("gecco.cli.run_judge_orchestrator.run_orchestrated_judge_pipeline", return_value=artifact):
                                        with patch("gecco.cli.run_judge_orchestrator.init_sentry"):
                                            run_orchestrator(config="test.yaml")

    set_judge_calls = mock_registry.set_judge_feedback.call_args_list
    assert len(set_judge_calls) == 1
    call_kwargs = set_judge_calls[0].kwargs
    feedback = call_kwargs["synthesized_feedback"]
    assert isinstance(feedback, dict)
    assert "generator" in feedback
    assert feedback["generator"] == "Search is stuck. Revert to simpler models."
    assert "default" not in feedback


def test_non_cmg_short_circuit_feedback_keyed_by_default():
    """In non-CMG mode, short-circuit judge feedback should remain keyed by 'default'."""
    from gecco.cli.run_judge_orchestrator import run_orchestrator

    cfg = _make_cfg(cmg_enabled=False)
    cfg.loop = SimpleNamespace(max_iterations=1)
    cfg.centralized_model_generation = SimpleNamespace(enabled=False)

    mock_registry = MagicMock()
    mock_registry.wait_for_clients_complete.return_value = 2
    mock_registry.count_clients_with_models.return_value = 1
    mock_registry.read.return_value = {"global_best": {}}

    artifact = SimpleNamespace(
        synthesized_feedback={"default": "Search is stuck. Try again."}
    )

    with patch("gecco.cli.run_judge_orchestrator.load_config", return_value=cfg):
        with patch("gecco.cli.run_judge_orchestrator.SharedRegistry", return_value=mock_registry):
            with patch("gecco.cli.run_judge_orchestrator.load_llm", return_value=(None, None)):
                with patch("gecco.cli.run_judge_orchestrator.load_data", return_value=MagicMock()):
                    with patch("gecco.cli.run_judge_orchestrator.split_by_participant", return_value={"prompt": MagicMock()}):
                        with patch("gecco.cli.run_judge_orchestrator.get_data2text_function", return_value=lambda *a, **k: "data text"):
                            with patch("gecco.cli.run_judge_orchestrator._build_judge_store_from_duckdb_sources", return_value=MagicMock()):
                                with patch("gecco.cli.run_judge_orchestrator.ToolUsingJudge", return_value=MagicMock()):
                                    with patch("gecco.cli.run_judge_orchestrator.run_orchestrated_judge_pipeline", return_value=artifact):
                                        with patch("gecco.cli.run_judge_orchestrator.init_sentry"):
                                            run_orchestrator(config="test.yaml", n_clients=2)

    set_judge_calls = mock_registry.set_judge_feedback.call_args_list
    assert len(set_judge_calls) == 1
    call_kwargs = set_judge_calls[0].kwargs
    feedback = call_kwargs["synthesized_feedback"]
    assert isinstance(feedback, dict)
    assert "default" in feedback
    assert feedback["default"] == "Search is stuck. Try again."


def test_non_cmg_orchestrator_supports_dict_clients():
    """Non-CMG orchestrators should handle validated dict-based client configs."""
    from gecco.cli.run_judge_orchestrator import run_orchestrator

    cfg = _make_cfg(cmg_enabled=False)
    cfg.loop = SimpleNamespace(max_iterations=1)
    cfg.centralized_model_generation = SimpleNamespace(enabled=False)
    cfg.clients = {
        "explore": SimpleNamespace(
            llm=SimpleNamespace(feedback_guidance="Focus on mechanism diversity.")
        )
    }
    cfg.judge.capabilities = ["performance_summary", "persona_synthesis"]

    mock_registry = MagicMock()
    mock_registry.wait_for_clients_complete.return_value = 2
    mock_registry.count_clients_with_models.return_value = 1
    mock_registry.read.return_value = {"global_best": {}}

    artifact = SimpleNamespace(
        synthesized_feedback={"explore": "Use a simpler but more novel mechanism."}
    )

    with patch("gecco.cli.run_judge_orchestrator.load_config", return_value=cfg):
        with patch(
            "gecco.cli.run_judge_orchestrator.SharedRegistry",
            return_value=mock_registry,
        ):
            with patch("gecco.cli.run_judge_orchestrator.load_llm", return_value=(None, None)):
                with patch("gecco.cli.run_judge_orchestrator.load_data", return_value=MagicMock()):
                    with patch(
                        "gecco.cli.run_judge_orchestrator.split_by_participant",
                        return_value={"prompt": MagicMock()},
                    ):
                        with patch(
                            "gecco.cli.run_judge_orchestrator.get_data2text_function",
                            return_value=lambda *a, **k: "data text",
                        ):
                            with patch(
                                "gecco.cli.run_judge_orchestrator._build_judge_store_from_duckdb_sources",
                                return_value=MagicMock(),
                            ):
                                with patch(
                                    "gecco.cli.run_judge_orchestrator.ToolUsingJudge",
                                    return_value=MagicMock(),
                                ):
                                    with patch(
                                        "gecco.cli.run_judge_orchestrator.run_orchestrated_judge_pipeline",
                                        return_value=artifact,
                                    ):
                                        with patch("gecco.cli.run_judge_orchestrator.init_sentry"):
                                            run_orchestrator(config="test.yaml", n_clients=2)

    set_judge_calls = mock_registry.set_judge_feedback.call_args_list
    assert len(set_judge_calls) == 1
    feedback = set_judge_calls[0].kwargs["synthesized_feedback"]
    assert feedback == {"explore": "Use a simpler but more novel mechanism."}
