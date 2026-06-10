"""Tests for CMG launcher dry-run behavior."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_cmg_cfg():
    """Return a minimal CMG-enabled config for launcher tests."""
    return SimpleNamespace(
        task=SimpleNamespace(name="test_cmg"),
        llm=SimpleNamespace(provider="openrouter", base_model="demo-model"),
        evaluation=SimpleNamespace(fit_type="group"),
        slurm={},
        centralized_model_generation=SimpleNamespace(
            enabled=True,
            generator_client="generator",
            n_models=2,
            run_final_evaluation=True,
        ),
    )


def test_slurm_dry_run_shows_sbatch_commands(mock_cmg_cfg, capsys, tmp_path):
    """Default (SLURM) with --dry-run should print actual sbatch commands without submitting."""
    from gecco.cli.launch_distributed import run_distributed_launcher

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "two_step_factors_cmg.yaml").write_text("task: {}\n", encoding="utf-8")

    with patch("gecco.cli.launch_distributed.PROJECT_ROOT", tmp_path):
        with patch("gecco.cli.launch_distributed.load_config", return_value=mock_cmg_cfg):
            with patch("gecco.cli.launch_distributed.init_sentry"):
                run_distributed_launcher(config="two_step_factors_cmg.yaml", dry_run=True)

    captured = capsys.readouterr()
    output = captured.out

    # Generator should be a non-array sbatch job using the shell wrapper
    assert "sbatch" in output
    assert "gecco-cmg-generator" in output
    assert "run_cmg_generator.sh" in output
    # Profile is passed as a positional arg to the shell script, not --client-profile
    gen_lines = [line for line in output.splitlines() if "gecco-cmg-generator" in line]
    assert any('"generator"' in line for line in gen_lines)

    # Evaluator array should be 0-(n_models-1) = 0-1
    assert "--array=0-1" in output
    assert "gecco-cmg-evaluator" in output
    assert "run_cmg_evaluator.sh" in output

    # Should NOT contain --array=0-2 (would be off-by-one)
    assert "--array=0-2" not in output

    # No evaluator should use the generator profile
    eval_lines = [line for line in output.splitlines() if "evaluator" in line.lower() or "--array" in line]
    for line in eval_lines:
        assert "--client-profile generator" not in line.lower() or "--array" not in line

    # Orchestrator job should be present
    assert "gecco-cmg-orchestrator" in output
    assert "run_judge_orchestrator.sh" in output

    # Final evaluation command should be scheduled by default
    assert "run_test_evaluation.sh" in output
    assert "results/test_cmg" in output

    # Dry-run marker
    assert "[Dry run]" not in output

    # No Rich object repr should leak into stdout
    assert "<rich.panel.Panel object" not in output

    # Readable panel fields should be present
    assert "Generator client:" in output
    assert "Evaluators:" in output


def test_slurm_dry_run_can_disable_final_eval(mock_cmg_cfg, capsys, tmp_path):
    """CLI override should suppress the final evaluation job."""
    from gecco.cli.launch_distributed import run_distributed_launcher

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "two_step_factors_cmg.yaml").write_text("task: {}\n", encoding="utf-8")

    disabled_cfg = SimpleNamespace(
        task=mock_cmg_cfg.task,
        llm=mock_cmg_cfg.llm,
        evaluation=mock_cmg_cfg.evaluation,
        slurm=mock_cmg_cfg.slurm,
        centralized_model_generation=SimpleNamespace(
            enabled=True,
            generator_client="generator",
            n_models=2,
            run_final_evaluation=False,
        ),
    )

    with patch("gecco.cli.launch_distributed.PROJECT_ROOT", tmp_path):
        with patch("gecco.cli.launch_distributed.load_config", return_value=disabled_cfg):
            with patch("gecco.cli.launch_distributed.init_sentry"):
                run_distributed_launcher(config="two_step_factors_cmg.yaml", dry_run=True)

    captured = capsys.readouterr()
    output = captured.out

    assert "Final eval:" in output
    assert "disabled" in output
    assert "run_test_evaluation.sh" not in output
