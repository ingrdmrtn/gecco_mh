"""Subprocess-based CLI smoke tests that do not require Slurm."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_gecco(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run the GeCCo CLI in a real subprocess and capture its output."""
    command = [sys.executable, "-m", "gecco", *args]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert (
        completed.returncode == 0
    ), f"command failed: {command}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    return completed


def test_cli_help_smoke_runs_real_entrypoint():
    """The top-level CLI should load and print its help text in a subprocess."""
    completed = _run_gecco("--help")

    assert "GeCCo command line interface" in completed.stdout
    assert "run" in completed.stdout
    assert "monitor" in completed.stdout


def test_distributed_dry_run_smoke_does_not_require_slurm(tmp_path):
    """Distributed dry-run should work even when sbatch is not on PATH."""
    env = os.environ.copy()
    env["PATH"] = str(tmp_path)

    completed = _run_gecco(
        "run",
        "distributed",
        "--config",
        "two_step_factors_distributed.yaml",
        "--dry-run",
        env=env,
    )

    assert "sbatch" in completed.stdout
    assert "run_gecco_distributed.sh" in completed.stdout
    assert "--array=0-6" in completed.stdout
    assert "<rich.panel.Panel object" not in completed.stdout
    assert "Submitted batch job" not in completed.stdout


def test_cmg_distributed_dry_run_smoke_does_not_require_slurm(tmp_path):
    """CMG dry-run should work even when sbatch is not on PATH."""
    env = os.environ.copy()
    env["PATH"] = str(tmp_path)

    completed = _run_gecco(
        "run",
        "distributed",
        "--config",
        "two_step_factors_cmg.yaml",
        "--dry-run",
        env=env,
    )

    assert "sbatch" in completed.stdout
    assert "run_cmg_generator.sh" in completed.stdout
    assert "run_cmg_evaluator.sh" in completed.stdout
    assert "--array=0-1" in completed.stdout
    assert "run_judge_orchestrator.sh" in completed.stdout
    assert "run_test_evaluation.sh" in completed.stdout
    assert "<rich.panel.Panel object" not in completed.stdout
    assert "Submitted batch job" not in completed.stdout
