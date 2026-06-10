"""CLI contract tests for Phase 2 cleanup work."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest


def test_run_distributed_routes_to_launcher():
    """The distributed launch path should live under the unified CLI."""
    from gecco.cli import build_parser
    from gecco.cli.launch_distributed import main as launch_distributed_main

    args = build_parser().parse_args(["run", "distributed", "--config", "demo.yaml"])

    assert args.handler is launch_distributed_main
    assert args.config == "demo.yaml"


def test_run_cmg_distributed_routes_to_cmg_launcher():
    """The CMG launch path should live under the unified CLI."""
    from gecco.cli import build_parser
    from gecco.cli.launch_cmg_distributed import main as launch_cmg_distributed_main

    args = build_parser().parse_args(
        ["run", "cmg-distributed", "--config", "demo.yaml"]
    )

    assert args.handler is launch_cmg_distributed_main
    assert args.config == "demo.yaml"


def test_judge_orchestrate_routes_to_orchestrator():
    """The judge orchestrator should be reachable through the CLI family."""
    from gecco.cli import build_parser
    from gecco.cli.run_judge_orchestrator import main as orchestrator_main

    args = build_parser().parse_args(
        ["judge", "orchestrate", "--config", "demo.yaml"]
    )

    assert args.handler is orchestrator_main
    assert args.config == "demo.yaml"


def test_monitor_routes_to_monitor_runtime():
    """Monitoring should be available as a top-level CLI command."""
    from gecco.cli import build_parser
    from gecco.cli.monitor_distributed import main as monitor_main

    args = build_parser().parse_args(["monitor", "--task", "demo-task"])

    assert args.handler is monitor_main
    assert args.task == "demo-task"


def test_reset_routes_to_reset_runtime():
    """Reset should be available as a top-level CLI command."""
    from gecco.cli import build_parser
    from gecco.cli.reset_distributed import main as reset_main

    args = build_parser().parse_args(["reset", "config/demo.yaml"])

    assert args.handler is reset_main
    assert args.config == "config/demo.yaml"


def test_run_local_client_routes_to_local_client_runtime():
    """Local client execution should live under the unified CLI."""
    from gecco.cli import build_parser
    from gecco.cli.run_local_client import main as local_client_main

    args = build_parser().parse_args(["run", "local-client", "--config", "demo.yaml"])

    assert args.handler is local_client_main
    assert args.config == "demo.yaml"


def test_internal_test_evaluation_routes_to_runtime():
    """Internal test-evaluation should remain reachable through the CLI."""
    from gecco.cli import build_parser
    from gecco.cli.run_test_evaluation import main as test_evaluation_main

    args = build_parser().parse_args(
        ["internal", "test-evaluation", "--config", "demo.yaml", "--results-dir", "results/demo"]
    )

    assert args.handler is test_evaluation_main
    assert args.config == "demo.yaml"
    assert args.results_dir == "results/demo"


def test_run_distributed_handler_passes_typed_arguments_directly():
    """The distributed CLI handler should pass parsed values straight through."""
    from gecco.cli import build_parser

    args = build_parser().parse_args(
        [
            "run",
            "distributed",
            "--config",
            "x.yaml",
            "--vllm-tp",
            "4",
            "--launch-vllm",
        ]
    )

    with patch("gecco.cli.launch_distributed.run_distributed_launcher") as run_launcher:
        args.handler(args)

    run_launcher.assert_called_once_with(
        config="x.yaml",
        profiles=None,
        extra_clients=0,
        launch_vllm=True,
        vllm_model=None,
        vllm_tp=4,
        vllm_port=8000,
        vllm_url=None,
        conda_env=None,
        partition=None,
        cpus_per_task=None,
        mem=None,
        dry_run=False,
        launch_orchestrator=False,
    )


def test_run_distributed_infers_orchestrator_launch_from_validated_config(tmp_path):
    """The launcher should infer orchestrator mode from validated config state."""
    from gecco.cli.launch_distributed import run_distributed_launcher
    from gecco.cli.launcher_utils import LaunchExecutor as RealLaunchExecutor

    project_root = tmp_path
    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "demo.yaml").write_text("task: {}\n", encoding="utf-8")

    cfg = SimpleNamespace(
        task=SimpleNamespace(name="demo-task"),
        llm=SimpleNamespace(provider="openrouter", base_model="demo-model"),
        loop=SimpleNamespace(max_iterations=1, n_clients=2),
        judge=SimpleNamespace(capabilities=["performance_summary"]),
        centralized_model_generation=SimpleNamespace(enabled=False),
        clients={"alpha": SimpleNamespace(), "beta": SimpleNamespace()},
        slurm={},
        evaluation=SimpleNamespace(fit_type="group"),
    )

    seen_commands: list[str] = []

    def fake_runner(command: str):
        seen_commands.append(command)
        if "launch_vllm_server.sh" in command:
            return SimpleNamespace(returncode=0, stdout="12345;cluster\n", stderr="")
        if "run_gecco_distributed.sh" in command and "--array=" in command:
            return SimpleNamespace(returncode=0, stdout="Submitted batch job 2001\n", stderr="")
        if "run_judge_orchestrator.sh" in command:
            return SimpleNamespace(returncode=0, stdout="Submitted batch job 2002\n", stderr="")
        if "run_test_evaluation.sh" in command:
            return SimpleNamespace(returncode=0, stdout="Submitted batch job 2003\n", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    real_executor = RealLaunchExecutor(runner=fake_runner, printer=lambda *_: None)

    with patch("gecco.cli.launch_distributed.PROJECT_ROOT", project_root):
        with patch("gecco.cli.launch_distributed.load_config", return_value=cfg) as load_config_mock:
            with patch("gecco.cli.launch_distributed.get_provider_spec") as provider_spec_mock:
                provider_spec_mock.return_value = SimpleNamespace(label="OpenRouter", key="openrouter")
                with patch("gecco.cli.launch_distributed.init_sentry"):
                    with patch("gecco.cli.launch_distributed.LaunchExecutor", return_value=real_executor):
                        run_distributed_launcher(config="demo.yaml", launch_vllm=True)

    load_config_mock.assert_called_once_with(project_root / "config" / "demo.yaml")
    assert seen_commands[0] == 'sbatch --parsable  bash/launch_vllm_server.sh "demo-model" 8000 1'
    assert seen_commands[1].startswith('sbatch --array=0-1 --cpus-per-task=48 --dependency=afterok:12345')
    assert seen_commands[1].endswith('bash/run_gecco_distributed.sh "demo.yaml" "alpha,beta" "" ""')
    assert seen_commands[2].startswith('sbatch --dependency=afterok:12345 --cpus-per-task=8')
    assert seen_commands[2].endswith('bash/run_judge_orchestrator.sh "demo.yaml" "" "2" ""')
    assert seen_commands[3].startswith('sbatch --dependency=afterok:2001 --cpus-per-task=8')
    assert seen_commands[3].endswith('bash/run_test_evaluation.sh "demo.yaml" "results/demo-task" ""')
    assert len(seen_commands) == 4


def test_run_cmg_distributed_builds_expected_commands(tmp_path):
    """The CMG launcher should preserve its sbatch command strings."""
    from gecco.cli.launch_cmg_distributed import run_cmg_distributed_launcher
    from gecco.cli.launcher_utils import LaunchExecutor as RealLaunchExecutor

    project_root = tmp_path
    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "demo.yaml").write_text("task: {}\n", encoding="utf-8")

    cfg = SimpleNamespace(
        task=SimpleNamespace(name="demo-task"),
        llm=SimpleNamespace(provider="openrouter", base_model="demo-model"),
        loop=SimpleNamespace(max_iterations=1),
        evaluation=SimpleNamespace(fit_type="group"),
        centralized_model_generation=SimpleNamespace(
            enabled=True,
            generator_client="generator",
            n_models=2,
            run_final_evaluation=True,
        ),
        slurm={},
    )

    seen_commands: list[str] = []

    def fake_runner(command: str):
        seen_commands.append(command)
        if "run_cmg_generator.sh" in command:
            return SimpleNamespace(returncode=0, stdout="Submitted batch job 5001\n", stderr="")
        if "run_cmg_evaluator.sh" in command and "--array=0-1" in command:
            return SimpleNamespace(returncode=0, stdout="Submitted batch job 5002\n", stderr="")
        if "run_judge_orchestrator.sh" in command:
            return SimpleNamespace(returncode=0, stdout="Submitted batch job 5003\n", stderr="")
        if "run_test_evaluation.sh" in command:
            return SimpleNamespace(returncode=0, stdout="Submitted batch job 5004\n", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    real_executor = RealLaunchExecutor(runner=fake_runner, printer=lambda *_: None)

    with patch("gecco.cli.launch_cmg_distributed.PROJECT_ROOT", project_root):
        with patch("gecco.cli.launch_cmg_distributed.load_config", return_value=cfg):
            with patch("gecco.cli.launch_cmg_distributed.console.print"):
                with patch("gecco.cli.launch_cmg_distributed.LaunchExecutor", return_value=real_executor):
                    run_cmg_distributed_launcher(config="demo.yaml", dry_run=False)

    assert seen_commands[0].startswith("sbatch --job-name=gecco-cmg-generator")
    assert seen_commands[1].startswith("sbatch --array=0-1 --job-name=gecco-cmg-evaluator")
    assert seen_commands[1].endswith('bash/run_cmg_evaluator.sh "demo.yaml" "" ""')
    assert seen_commands[2].startswith("sbatch --job-name=gecco-cmg-orchestrator")
    assert seen_commands[3].startswith("sbatch --dependency=afterok:5001:5002:5003 --cpus-per-task=8")
    assert len(seen_commands) == 4


def test_cli_entrypoint_functions_are_importable_and_callable():
    """Extracted runtime entrypoints should be exposed as direct callables."""
    from gecco.cli.launch_cmg_distributed import run_cmg_distributed_launcher
    from gecco.cli.launch_distributed import run_distributed_launcher
    from gecco.cli.monitor_distributed import run_monitor
    from gecco.cli.reset_distributed import run_reset
    from gecco.cli.run_gecco_distributed import run_distributed_client
    from gecco.cli.run_judge_orchestrator import run_orchestrator
    from gecco.cli.run_local_client import run_local_client
    from gecco.cli.run_test_evaluation import run_test_evaluation

    assert callable(run_distributed_launcher)
    assert callable(run_cmg_distributed_launcher)
    assert callable(run_local_client)
    assert callable(run_monitor)
    assert callable(run_reset)
    assert callable(run_distributed_client)
    assert callable(run_test_evaluation)
    assert callable(run_orchestrator)


@pytest.mark.parametrize(
    "legacy_name",
    [
        "launch_distributed",
        "run_gecco_distributed",
        "run_judge_orchestrator",
        "launch_cmg_distributed",
        "monitor_distributed",
        "reset_distributed",
    ],
)
def test_legacy_script_names_are_not_valid_cli_commands(legacy_name):
    """Removed legacy entrypoint names should not remain as CLI aliases."""
    from gecco.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args([legacy_name])
