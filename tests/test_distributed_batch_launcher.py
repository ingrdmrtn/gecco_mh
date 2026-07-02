from __future__ import annotations

import re
from datetime import datetime
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import ANY, patch

import pytest

from gecco.cli import build_parser
from gecco.cli.launch_distributed_batch import run_distributed_batch_launcher
from gecco.cli.launcher_utils import LaunchExecutor as RealLaunchExecutor


def _make_cfg():
    return SimpleNamespace(
        task=SimpleNamespace(name="demo-task"),
        llm=SimpleNamespace(provider="openrouter", base_model="demo-model"),
        loop=SimpleNamespace(n_clients=2),
        judge=None,
        centralized_model_generation=SimpleNamespace(enabled=False),
        clients={"alpha": SimpleNamespace(), "beta": SimpleNamespace()},
        slurm={},
        evaluation=SimpleNamespace(fit_type="group"),
    )


@contextmanager
def _patched_batch_environment(project_root, cfg):
    provider_spec = SimpleNamespace(label="OpenRouter", key="openrouter")
    with patch("gecco.cli.launch_distributed.PROJECT_ROOT", project_root), patch(
        "gecco.cli.launch_distributed_batch.launch_distributed_cli.PROJECT_ROOT",
        project_root,
    ), patch("gecco.cli.launch_distributed.load_config", return_value=cfg), patch(
        "gecco.cli.launch_distributed.get_provider_spec", return_value=provider_spec
    ), patch("gecco.cli.launch_distributed.init_sentry", return_value=False), patch(
        "gecco.cli.launch_distributed.get_judge_mode", return_value="off"
    ):
        yield


def _make_fake_runner(seen_commands: list[str]):
    def fake_runner(command: str):
        seen_commands.append(command)
        job_id = str(100 + len(seen_commands))
        return SimpleNamespace(returncode=0, stdout=f"Submitted batch job {job_id}\n", stderr="")

    return fake_runner


def _root_commands(commands: list[str]) -> list[str]:
    return [
        command
        for command in commands
        if any(
            marker in command
            for marker in (
                "run_gecco_distributed.sh",
                "run_cmg_generator.sh",
                "run_cmg_evaluator.sh",
                "run_judge_orchestrator.sh",
            )
        )
    ]


def _commands_with_marker(commands: list[str], marker: str) -> list[str]:
    return [command for command in commands if marker in command]


def test_batch_launcher_expands_explicit_config_list(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "a.yaml").write_text("task: {}\n", encoding="utf-8")
    (config_dir / "b.yaml").write_text("task: {}\n", encoding="utf-8")

    cfg = _make_cfg()
    seen_commands: list[str] = []
    real_executor = RealLaunchExecutor(runner=_make_fake_runner(seen_commands), printer=lambda *_: None)

    with _patched_batch_environment(tmp_path, cfg), patch(
        "gecco.cli.launch_distributed_batch.LaunchExecutor", return_value=real_executor
    ):
        run_distributed_batch_launcher(configs=["config/a.yaml", "config/b.yaml"])

    root_commands = _root_commands(seen_commands)
    assert len(root_commands) == 2
    assert "config/a.yaml" in root_commands[0]
    assert "config/b.yaml" in root_commands[1]


def test_batch_launcher_passes_submission_controls_without_changing_plan(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "a.yaml").write_text("task: {}\n", encoding="utf-8")

    cfg = _make_cfg()
    seen_commands: list[str] = []
    real_executor = RealLaunchExecutor(runner=_make_fake_runner(seen_commands), printer=lambda *_: None)

    with _patched_batch_environment(tmp_path, cfg), patch(
        "gecco.cli.launch_distributed_batch.LaunchExecutor", return_value=real_executor
    ) as launch_executor_mock:
        run_distributed_batch_launcher(
            configs=["config/a.yaml"],
            submit_delay_seconds=2.5,
            sbatch_retry_attempts=5,
            sbatch_retry_backoff_seconds=4.5,
        )

    launch_executor_mock.assert_called_once_with(
        printer=ANY,
        submit_delay_seconds=2.5,
        sbatch_retry_attempts=5,
        sbatch_retry_backoff_seconds=4.5,
    )
    root_commands = _root_commands(seen_commands)
    assert len(root_commands) == 1
    assert "config/a.yaml" in root_commands[0]


def test_batch_launcher_help_mentions_submission_controls(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "distributed-batch", "--help"])

    output = capsys.readouterr().out
    assert "--submit-delay-seconds" in output
    assert "default: 1.0" in output
    assert "--sbatch-retry-attempts" in output
    assert "default: 3" in output
    assert "--sbatch-retry-backoff-seconds" in output
    assert "default: 2.0" in output


def test_batch_launcher_expands_config_dir_sorted_and_ignores_non_yaml(tmp_path, capsys):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "b.yaml").write_text("task: {}\n", encoding="utf-8")
    (config_dir / "a.yaml").write_text("task: {}\n", encoding="utf-8")
    (config_dir / "ignore.txt").write_text("ignore me\n", encoding="utf-8")

    cfg = _make_cfg()
    seen_commands: list[str] = []
    real_executor = RealLaunchExecutor(runner=_make_fake_runner(seen_commands), printer=lambda *_: None)

    with _patched_batch_environment(tmp_path, cfg), patch(
        "gecco.cli.launch_distributed_batch.LaunchExecutor", return_value=real_executor
    ):
        run_distributed_batch_launcher(config_dir="config")

    output = capsys.readouterr().out

    root_commands = _root_commands(seen_commands)
    assert len(root_commands) == 2
    assert "config/a.yaml" in root_commands[0]
    assert "config/b.yaml" in root_commands[1]
    assert all("ignore.txt" not in command for command in seen_commands)
    assert "Distributed batch summary" in output
    assert "Configs found in config" in output
    assert "Distributed batch complete" in output


def test_batch_launcher_creates_unique_replicate_run_ids(tmp_path, capsys):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "demo.yaml").write_text("task: {}\n", encoding="utf-8")
    (config_dir / "alt.yaml").write_text("task: {}\n", encoding="utf-8")

    cfg = _make_cfg()
    seen_commands: list[str] = []
    real_executor = RealLaunchExecutor(runner=_make_fake_runner(seen_commands), printer=lambda *_: None)

    with _patched_batch_environment(tmp_path, cfg), patch(
        "gecco.cli.launch_distributed_batch.LaunchExecutor", return_value=real_executor
    ), patch("gecco.cli.launch_distributed_batch.datetime") as datetime_mock:
        datetime_mock.now.return_value = datetime(2026, 7, 1, 12, 34, 56)
        run_distributed_batch_launcher(
            configs=["config/demo.yaml", "config/alt.yaml"],
            replicates=3,
            max_concurrent_configs=1,
        )

    output = capsys.readouterr().out

    root_commands = _root_commands(seen_commands)
    assert len(root_commands) == 6
    assert sum("config/demo.yaml" in command for command in root_commands) == 3
    assert sum("config/alt.yaml" in command for command in root_commands) == 3

    run_ids = set(
        re.findall(r"batch-20260701-123456-[^\"/ ]+-rep-\d{3}", "\n".join(root_commands))
    )
    assert run_ids == {
        "batch-20260701-123456-demo-rep-001",
        "batch-20260701-123456-demo-rep-002",
        "batch-20260701-123456-demo-rep-003",
        "batch-20260701-123456-alt-rep-001",
        "batch-20260701-123456-alt-rep-002",
        "batch-20260701-123456-alt-rep-003",
    }
    assert "Distributed batch summary" in output
    assert "Configs to launch (explicit order)" in output
    assert "Pipeline 1/6" in output
    assert "lane dependencies: no" in output
    assert "lane dependencies: yes" in output
    assert "Distributed batch complete" in output
    assert "job ID:" in output


def test_batch_launcher_chains_orchestrated_runs_wait_on_all_terminal_jobs(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "a.yaml").write_text("task: {}\n", encoding="utf-8")
    (config_dir / "b.yaml").write_text("task: {}\n", encoding="utf-8")

    cfg = _make_cfg()
    seen_commands: list[str] = []
    real_executor = RealLaunchExecutor(runner=_make_fake_runner(seen_commands), printer=lambda *_: None)

    with _patched_batch_environment(tmp_path, cfg), patch(
        "gecco.cli.launch_distributed_batch.LaunchExecutor", return_value=real_executor
    ):
        run_distributed_batch_launcher(
            configs=["config/a.yaml", "config/b.yaml"],
            launch_orchestrator=True,
            max_concurrent_configs=1,
        )

    root_commands = _root_commands(seen_commands)
    assert len(root_commands) == 4
    assert "--dependency=" not in root_commands[0]
    assert "--dependency=" not in root_commands[1]
    assert "--dependency=afterany:103:102" in root_commands[2]
    assert "--dependency=afterany:103:102" in root_commands[3]

    test_eval_commands = _commands_with_marker(seen_commands, "run_test_evaluation.sh")
    assert len(test_eval_commands) == 2
    assert "--dependency=afterok:101" in test_eval_commands[0]
    assert "--dependency=afterok:104" in test_eval_commands[1]


def test_batch_launcher_chains_cmg_without_final_eval_waits_on_all_terminal_jobs(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "a.yaml").write_text("task: {}\n", encoding="utf-8")
    (config_dir / "b.yaml").write_text("task: {}\n", encoding="utf-8")

    cfg = SimpleNamespace(
        task=SimpleNamespace(name="demo-task"),
        llm=SimpleNamespace(provider="openrouter", base_model="demo-model"),
        loop=SimpleNamespace(n_clients=2),
        judge=None,
        centralized_model_generation=SimpleNamespace(
            enabled=True,
            generator_client="generator",
            n_models=2,
            run_final_evaluation=False,
        ),
        clients={"alpha": SimpleNamespace(), "beta": SimpleNamespace()},
        slurm={},
        evaluation=SimpleNamespace(fit_type="group"),
    )
    seen_commands: list[str] = []
    real_executor = RealLaunchExecutor(runner=_make_fake_runner(seen_commands), printer=lambda *_: None)

    with _patched_batch_environment(tmp_path, cfg), patch(
        "gecco.cli.launch_distributed_batch.LaunchExecutor", return_value=real_executor
    ):
        run_distributed_batch_launcher(
            configs=["config/a.yaml", "config/b.yaml"],
            max_concurrent_configs=1,
        )

    root_commands = _root_commands(seen_commands)
    assert len(root_commands) == 6
    assert "--dependency=" not in root_commands[0]
    assert "--dependency=" not in root_commands[1]
    assert "--dependency=" not in root_commands[2]
    assert "--dependency=afterany:101:102:103" in root_commands[3]
    assert "--dependency=afterany:101:102:103" in root_commands[4]
    assert "--dependency=afterany:101:102:103" in root_commands[5]
    assert all("run_test_evaluation.sh" not in command for command in seen_commands)


def test_batch_launcher_chains_cmg_final_eval_on_current_pipeline_jobs(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "a.yaml").write_text("task: {}\n", encoding="utf-8")
    (config_dir / "b.yaml").write_text("task: {}\n", encoding="utf-8")

    cfg = SimpleNamespace(
        task=SimpleNamespace(name="demo-task"),
        llm=SimpleNamespace(provider="openrouter", base_model="demo-model"),
        loop=SimpleNamespace(n_clients=2),
        judge=None,
        centralized_model_generation=SimpleNamespace(
            enabled=True,
            generator_client="generator",
            n_models=2,
            run_final_evaluation=True,
        ),
        clients={"alpha": SimpleNamespace(), "beta": SimpleNamespace()},
        slurm={},
        evaluation=SimpleNamespace(fit_type="group"),
    )
    seen_commands: list[str] = []
    real_executor = RealLaunchExecutor(runner=_make_fake_runner(seen_commands), printer=lambda *_: None)

    with _patched_batch_environment(tmp_path, cfg), patch(
        "gecco.cli.launch_distributed_batch.LaunchExecutor", return_value=real_executor
    ):
        run_distributed_batch_launcher(
            configs=["config/a.yaml", "config/b.yaml"],
            max_concurrent_configs=1,
        )

    root_commands = _root_commands(seen_commands)
    assert len(root_commands) == 6
    assert "--dependency=" not in root_commands[0]
    assert "--dependency=" not in root_commands[1]
    assert "--dependency=" not in root_commands[2]
    assert "--dependency=afterany:104" in root_commands[3]
    assert "--dependency=afterany:104" in root_commands[4]
    assert "--dependency=afterany:104" in root_commands[5]

    final_eval_commands = _commands_with_marker(seen_commands, "run_test_evaluation.sh")
    assert len(final_eval_commands) == 2
    assert "--dependency=afterok:101:102:103" in final_eval_commands[0]
    assert "--dependency=afterok:105:106:107" in final_eval_commands[1]


def test_batch_launcher_chains_whole_pipelines_by_max_concurrent_configs(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for name in ["a.yaml", "b.yaml", "c.yaml", "d.yaml", "e.yaml"]:
        (config_dir / name).write_text("task: {}\n", encoding="utf-8")

    cfg = _make_cfg()
    seen_commands: list[str] = []
    real_executor = RealLaunchExecutor(runner=_make_fake_runner(seen_commands), printer=lambda *_: None)

    with _patched_batch_environment(tmp_path, cfg), patch(
        "gecco.cli.launch_distributed_batch.LaunchExecutor", return_value=real_executor
    ):
        run_distributed_batch_launcher(config_dir="config", max_concurrent_configs=2)

    root_commands = _root_commands(seen_commands)
    assert len(root_commands) == 5
    assert "--dependency=" not in root_commands[0]
    assert "--dependency=" not in root_commands[1]
    assert "--dependency=afterany:102" in root_commands[2]
    assert "--dependency=afterany:104" in root_commands[3]
    assert "--dependency=afterany:106" in root_commands[4]
    assert not any(re.search(r"--array=[^ ]*%\d+", command) for command in seen_commands)


def test_batch_launcher_supports_afterok_policy(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "a.yaml").write_text("task: {}\n", encoding="utf-8")
    (config_dir / "b.yaml").write_text("task: {}\n", encoding="utf-8")

    cfg = _make_cfg()
    seen_commands: list[str] = []
    real_executor = RealLaunchExecutor(runner=_make_fake_runner(seen_commands), printer=lambda *_: None)

    with _patched_batch_environment(tmp_path, cfg), patch(
        "gecco.cli.launch_distributed_batch.LaunchExecutor", return_value=real_executor
    ):
        run_distributed_batch_launcher(config_dir="config", dependency_policy="afterok")

    root_commands = _root_commands(seen_commands)
    assert len(root_commands) == 2
    assert "--dependency=afterok:102" in root_commands[1]
    assert "--dependency=afterany" not in root_commands[1]


def test_batch_launcher_reports_dry_run_summary(tmp_path, capsys):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "a.yaml").write_text("task: {}\n", encoding="utf-8")

    cfg = _make_cfg()
    seen_commands: list[str] = []
    real_executor = RealLaunchExecutor(runner=_make_fake_runner(seen_commands), printer=lambda *_: None)

    with _patched_batch_environment(tmp_path, cfg), patch(
        "gecco.cli.launch_distributed_batch.LaunchExecutor", return_value=real_executor
    ), patch("gecco.cli.launch_distributed_batch.datetime") as datetime_mock:
        datetime_mock.now.return_value = datetime(2026, 7, 1, 12, 34, 56)
        run_distributed_batch_launcher(configs=["config/a.yaml"], dry_run=True)

    output = capsys.readouterr().out

    assert "Distributed batch summary" in output
    assert "dry-run" in output
    assert "Pipeline 1/1" in output
    assert "lane dependencies: no" in output
    assert "Distributed batch complete" in output
    assert "previewed" in output
    assert "job ID:" not in output


def test_batch_launcher_requires_configs_or_config_dir():
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["run", "distributed-batch"])

    assert excinfo.value.code == 2
