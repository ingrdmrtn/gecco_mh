"""Local preview contract tests for the unified distributed launcher."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _write_config_tree(root: Path, name: str) -> None:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / name).write_text("task: {}\n", encoding="utf-8")


def test_local_preview_prints_runnable_regular_commands(tmp_path, capsys):
    """Non-CMG local preview should print internal client and orchestrator commands."""
    from gecco.cli.launch_distributed import run_distributed_launcher

    _write_config_tree(tmp_path, "demo.yaml")
    cfg = SimpleNamespace(
        task=SimpleNamespace(name="demo-task"),
        llm=SimpleNamespace(provider="openrouter", base_model="demo-model"),
        loop=SimpleNamespace(n_clients=2),
        judge=SimpleNamespace(capabilities=["performance_summary"]),
        clients={"alpha": SimpleNamespace(), "beta": SimpleNamespace()},
        evaluation=SimpleNamespace(fit_type="group"),
        slurm={},
        centralized_model_generation=SimpleNamespace(enabled=False),
    )

    with patch("gecco.cli.launch_distributed.PROJECT_ROOT", tmp_path):
        with patch("gecco.cli.launch_distributed.load_config", return_value=cfg):
            with patch("gecco.cli.launch_distributed.init_sentry"):
                with patch("gecco.cli.launch_distributed.LaunchExecutor") as executor_mock:
                    run_distributed_launcher(config="demo.yaml", local=True)

    output = capsys.readouterr().out

    assert "sbatch" not in output
    assert "python -m gecco internal distributed-client" in output
    assert "--client-profile \"alpha\"" in output
    assert "--client-profile \"beta\"" in output
    assert "python -m gecco internal judge-orchestrate" in output
    executor_mock.assert_not_called()


def test_local_preview_prints_runnable_cmg_commands(tmp_path, capsys):
    """CMG local preview should print generator, evaluator, and orchestrator commands."""
    from gecco.cli.launch_distributed import run_distributed_launcher

    _write_config_tree(tmp_path, "demo-cmg.yaml")
    cfg = SimpleNamespace(
        task=SimpleNamespace(name="demo-task"),
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

    with patch("gecco.cli.launch_distributed.PROJECT_ROOT", tmp_path):
        with patch("gecco.cli.launch_distributed.load_config", return_value=cfg):
            with patch("gecco.cli.launch_distributed.init_sentry"):
                with patch("gecco.cli.launch_distributed.LaunchExecutor") as executor_mock:
                    run_distributed_launcher(config="demo-cmg.yaml", local=True)

    output = capsys.readouterr().out

    assert "sbatch" not in output
    assert "python -m gecco internal distributed-client" in output
    assert "--client-profile \"generator\"" in output
    assert "Evaluator 0" in output
    assert "Evaluator 1" in output
    assert "python -m gecco internal judge-orchestrate" in output
    assert "python -m gecco internal test-evaluation" in output
    executor_mock.assert_not_called()
