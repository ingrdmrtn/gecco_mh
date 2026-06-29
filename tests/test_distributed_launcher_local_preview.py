"""Local preview contract tests for the unified distributed launcher."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _write_config_tree(root: Path, name: str) -> None:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / name
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("task: {}\n", encoding="utf-8")


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
                    with patch("gecco.cli.launch_distributed._new_run_id", return_value="auto-run"):
                        run_distributed_launcher(config="demo.yaml", local=True)

    output = capsys.readouterr().out

    assert "sbatch" not in output
    assert "python -m gecco internal distributed-client" in output
    assert "--client-profile \"alpha\"" in output
    assert "--client-profile \"beta\"" in output
    assert "Run ID: auto-run" in output
    assert '--results-dir "results/demo/auto-run"' in output
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


def test_dry_run_regular_launch_uses_nested_paths(tmp_path, capsys):
    """Dry-run regular launches should mirror config paths in logs and results."""
    from gecco.cli.launch_distributed import run_distributed_launcher

    _write_config_tree(tmp_path, "two_step_factors/deepseekv4flash/judge_off.yaml")
    cfg = SimpleNamespace(
        task=SimpleNamespace(name="judge_off"),
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
            with patch("gecco.cli.launch_distributed.get_provider_spec") as provider_spec_mock:
                provider_spec_mock.return_value = SimpleNamespace(label="OpenRouter", key="openrouter")
                with patch("gecco.cli.launch_distributed.init_sentry"):
                    run_distributed_launcher(
                        config="two_step_factors/deepseekv4flash/judge_off.yaml",
                        run_id="run-123",
                        dry_run=True,
                        launch_orchestrator=True,
                    )

    output = capsys.readouterr().out
    normalized_output = " ".join(output.split())

    assert "logs/two_step_factors/deepseekv4flash/judge_off/run-123/gecco-client-%A_%a.out" in normalized_output
    assert "logs/two_step_factors/deepseekv4flash/judge_off/run-123/gecco-orchestrator-%j.err" in normalized_output
    assert "Results dir: results/two_step_factors/deepseekv4flash/judge_off/run-123" in normalized_output
    assert "python -m gecco monitor --task judge_off --results-dir results/two_step_factors/deepseekv4flash/judge_off/run-123 --watch 10" in normalized_output


def test_dry_run_regular_launch_accepts_config_prefixed_paths(tmp_path, capsys):
    """Launcher config resolution should accept already-prefixed config paths."""
    from gecco.cli.launch_distributed import run_distributed_launcher

    _write_config_tree(tmp_path, "two_step_factors/deepseekv4flash/judge_off.yaml")
    cfg = SimpleNamespace(
        task=SimpleNamespace(name="judge_off"),
        llm=SimpleNamespace(provider="openrouter", base_model="demo-model"),
        loop=SimpleNamespace(n_clients=2),
        judge=SimpleNamespace(capabilities=["performance_summary"]),
        clients={"alpha": SimpleNamespace(), "beta": SimpleNamespace()},
        evaluation=SimpleNamespace(fit_type="group"),
        slurm={},
        centralized_model_generation=SimpleNamespace(enabled=False),
    )

    with patch("gecco.cli.launch_distributed.PROJECT_ROOT", tmp_path):
        with patch("gecco.cli.launch_distributed.load_config", return_value=cfg) as load_config_mock:
            with patch("gecco.cli.launch_distributed.get_provider_spec") as provider_spec_mock:
                provider_spec_mock.return_value = SimpleNamespace(label="OpenRouter", key="openrouter")
                with patch("gecco.cli.launch_distributed.init_sentry"):
                    run_distributed_launcher(
                        config="config/two_step_factors/deepseekv4flash/judge_off.yaml",
                        run_id="run-123",
                        dry_run=True,
                    )

    output = capsys.readouterr().out
    normalized_output = " ".join(output.split())

    load_config_mock.assert_called_once_with(
        tmp_path / "config" / "two_step_factors" / "deepseekv4flash" / "judge_off.yaml"
    )
    assert "logs/two_step_factors/deepseekv4flash/judge_off/run-123/gecco-client-%A_%a.out" in normalized_output
    assert "Results dir: results/two_step_factors/deepseekv4flash/judge_off/run-123" in normalized_output


def test_dry_run_cmg_launch_uses_nested_paths(tmp_path, capsys):
    """Dry-run CMG launches should mirror config paths in logs and results."""
    from gecco.cli.launch_distributed import run_distributed_launcher

    _write_config_tree(tmp_path, "two_step_factors/deepseekv4flash/judge_off.yaml")
    cfg = SimpleNamespace(
        task=SimpleNamespace(name="judge_off"),
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
            with patch("gecco.cli.launch_distributed.get_provider_spec") as provider_spec_mock:
                provider_spec_mock.return_value = SimpleNamespace(label="OpenRouter", key="openrouter")
                with patch("gecco.cli.launch_distributed.init_sentry"):
                    run_distributed_launcher(
                        config="two_step_factors/deepseekv4flash/judge_off.yaml",
                        run_id="run-123",
                        dry_run=True,
                    )

    output = capsys.readouterr().out
    normalized_output = " ".join(output.split())

    assert "logs/two_step_factors/deepseekv4flash/judge_off/run-123/gecco-cmg-generator-%j.out" in normalized_output
    assert "logs/two_step_factors/deepseekv4flash/judge_off/run-123/gecco-cmg-evaluator-%A_%a.err" in normalized_output
    assert 'bash/run_test_evaluation.sh "two_step_factors/deepseekv4flash/judge_off.yaml" "results/two_step_factors/deepseekv4flash/judge_off/run-123"' in normalized_output
    assert "python -m gecco monitor --task judge_off --results-dir results/two_step_factors/deepseekv4flash/judge_off/run-123 --watch 10" in normalized_output
