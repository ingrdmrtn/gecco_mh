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
        centralized_model_generation=SimpleNamespace(
            enabled=True,
            generator_client="generator",
            n_models=2,
        ),
    )


def test_slurm_dry_run_shows_sbatch_commands(mock_cmg_cfg, capsys):
    """Default (SLURM) with --dry-run should print actual sbatch commands without submitting."""
    from scripts.launch_cmg_distributed import main

    with patch.object(sys, "argv", [
        "launch_cmg_distributed.py",
        "--config", "two_step_factors_cmg.yaml",
        "--dry-run",
    ]):
        with patch("scripts.launch_cmg_distributed.load_config", return_value=mock_cmg_cfg):
            main()

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

    # Should NOT contain --array=0-2 (would be off-by-one)
    assert "--array=0-2" not in output

    # No evaluator should use the generator profile
    eval_lines = [line for line in output.splitlines() if "evaluator" in line.lower() or "--array" in line]
    for line in eval_lines:
        assert "--client-profile generator" not in line.lower() or "--array" not in line

    # Orchestrator job should be present
    assert "gecco-cmg-orchestrator" in output

    # Dry-run marker
    assert "[Dry run] No jobs were submitted." in output

    # No Rich object repr should leak into stdout
    assert "<rich.panel.Panel object" not in output

    # Readable panel fields should be present
    assert "Generator Client" in output
    assert "Evaluators" in output
