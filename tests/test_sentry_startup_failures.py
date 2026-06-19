"""Tests for Sentry startup failure handling and SLURM preflight."""

from pathlib import Path
from unittest.mock import patch

import pytest


def test_resolve_config_path_prefers_config_root_for_relative_subdirs(tmp_path):
    from gecco.cli.config_paths import resolve_config_path

    project_root = tmp_path / "project"
    project_root.mkdir()

    relative_path = resolve_config_path(
        "two_step_factors/deepseekv4flash/judge_off.yaml",
        project_root=project_root,
    )
    absolute_path = resolve_config_path(
        str(project_root / "config" / "two_step_factors" / "deepseekv4flash" / "judge_off.yaml"),
        project_root=project_root,
    )

    assert relative_path == project_root / "config" / "two_step_factors" / "deepseekv4flash" / "judge_off.yaml"
    assert absolute_path == project_root / "config" / "two_step_factors" / "deepseekv4flash" / "judge_off.yaml"


def test_slurm_preflight_prints_provider_only_on_success(tmp_path, capsys, monkeypatch):
    from gecco.cli import slurm_preflight

    project_root = tmp_path / "project"
    config_path = project_root / "config" / "two_step_factors" / "deepseekv4flash"
    config_path.mkdir(parents=True)
    (config_path / "judge_off.yaml").write_text("llm:\n  provider: api\n", encoding="utf-8")

    monkeypatch.setattr(slurm_preflight, "PROJECT_ROOT", project_root)
    with patch("gecco.cli.slurm_preflight.init_sentry") as mock_init:
        result = slurm_preflight.main(["--config", "two_step_factors/deepseekv4flash/judge_off.yaml"])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == "api\n"
    assert captured.err == ""
    mock_init.assert_called_once_with(config_name="two_step_factors/deepseekv4flash/judge_off.yaml", component="slurm_wrapper")


def test_slurm_preflight_reports_missing_config_before_exit(tmp_path, capsys, monkeypatch):
    from gecco.cli import slurm_preflight

    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(slurm_preflight, "PROJECT_ROOT", project_root)

    with (
        patch("gecco.cli.slurm_preflight.init_sentry") as mock_init,
        patch("gecco.cli.slurm_preflight.capture_operational_error") as mock_capture,
    ):
        result = slurm_preflight.main(["--config", "missing.yaml"])

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert "ERROR: Failed to detect provider" in captured.err
    mock_init.assert_called_once_with(config_name="missing.yaml", component="slurm_wrapper")
    mock_capture.assert_called_once()
    error = mock_capture.call_args.args[0]
    assert isinstance(error, FileNotFoundError)
    assert mock_capture.call_args.kwargs["component"] == "slurm_wrapper"
    assert mock_capture.call_args.kwargs["operation"] == "detect_provider"
    assert mock_capture.call_args.kwargs["config_name"] == "missing.yaml"
    assert Path(mock_capture.call_args.kwargs["config_path"]) == project_root / "config" / "missing.yaml"


@pytest.mark.parametrize(
    "script_name",
    [
        "run_gecco_distributed.sh",
        "run_judge_orchestrator.sh",
        "run_cmg_generator.sh",
        "run_cmg_evaluator.sh",
    ],
)
def test_slurm_wrapper_scripts_call_preflight_module(script_name):
    script_path = Path(__file__).resolve().parents[1] / "bash" / script_name
    text = script_path.read_text(encoding="utf-8")

    assert "gecco.cli.slurm_preflight" in text
    assert "--config \"$CONFIG\"" in text


def test_run_distributed_client_uses_shared_config_resolver():
    from gecco.cli import run_gecco_distributed

    resolved_path = Path("/tmp/opencode/resolved-distributed.yaml")

    with (
        patch("gecco.cli.run_gecco_distributed.resolve_config_path", return_value=resolved_path) as mock_resolve,
        patch("gecco.cli.run_gecco_distributed.load_config", side_effect=RuntimeError("stop")) as mock_load,
    ):
        with pytest.raises(RuntimeError, match="stop"):
            run_gecco_distributed.run_distributed_client(config="two_step_factors/deepseekv4flash/judge_off.yaml")

    mock_resolve.assert_called_once_with(
        "two_step_factors/deepseekv4flash/judge_off.yaml",
        project_root=run_gecco_distributed.PROJECT_ROOT,
    )
    mock_load.assert_called_once_with(resolved_path)


def test_run_orchestrator_uses_shared_config_resolver():
    from gecco.cli import run_judge_orchestrator

    resolved_path = Path("/tmp/opencode/resolved-orchestrator.yaml")

    with (
        patch("gecco.cli.run_judge_orchestrator.resolve_config_path", return_value=resolved_path) as mock_resolve,
        patch("gecco.cli.run_judge_orchestrator.load_config", side_effect=RuntimeError("stop")) as mock_load,
    ):
        with pytest.raises(RuntimeError, match="stop"):
            run_judge_orchestrator.run_orchestrator(config="two_step_factors/deepseekv4flash/judge_off.yaml")

    mock_resolve.assert_called_once_with(
        "two_step_factors/deepseekv4flash/judge_off.yaml",
        project_root=run_judge_orchestrator.PROJECT_ROOT,
    )
    mock_load.assert_called_once_with(resolved_path)
