"""Launcher utility contract tests."""

from types import SimpleNamespace

from gecco.cli.launcher_utils import (
    LaunchCommand,
    LaunchExecutor,
    LaunchPlan,
    build_afterok_dependency,
    _parse_job_id,
)


def test_launch_plan_executes_in_order_and_propagates_dependencies():
    """A launch plan should preserve order and chain afterok dependencies."""
    seen_commands: list[str] = []

    def fake_runner(command: str):
        seen_commands.append(command)
        if "launch_vllm_server.sh" in command:
            return SimpleNamespace(returncode=0, stdout="12345;cluster\n", stderr="")
        job_id = str(100 + len(seen_commands))
        return SimpleNamespace(returncode=0, stdout=f"Submitted batch job {job_id}\n", stderr="")

    executor = LaunchExecutor(runner=fake_runner, printer=lambda *_: None)
    plan = LaunchPlan(
        commands=(
            LaunchCommand(label="server", command="sbatch bash/launch_vllm_server.sh"),
            LaunchCommand(
                label="clients",
                command="sbatch {dependency} bash/run_gecco_distributed.sh",
                dependency_labels=("server",),
            ),
        )
    )

    results = plan.execute(executor)

    assert seen_commands == [
        "sbatch bash/launch_vllm_server.sh",
        "sbatch --dependency=afterok:12345 bash/run_gecco_distributed.sh",
    ]
    assert [result.job_id for result in results] == ["12345", "102"]
    assert results[1].submitted is True


def test_parse_job_id_accepts_parsable_slurm_output():
    """Slurm parsable output should still yield the job id."""
    assert _parse_job_id("12345") == "12345"
    assert _parse_job_id("12345;cluster") == "12345"


def test_launch_executor_skips_submission_in_dry_run():
    """Dry-run mode should print without calling the runner."""
    called = False

    def fake_runner(_command: str):
        nonlocal called
        called = True
        raise AssertionError("runner should not be called in dry run")

    executor = LaunchExecutor(runner=fake_runner, printer=lambda *_: None)
    plan = LaunchPlan(commands=(LaunchCommand(label="clients", command="sbatch bash/run.sh"),))

    results = executor.execute(plan, dry_run=True)

    assert called is False
    assert results[0].submitted is False
    assert results[0].job_id is None
    assert results[0].command == "sbatch bash/run.sh"


def test_launch_executor_leaves_dependency_flag_blank_without_prior_job_ids():
    """Dependency labels should only expand when prior jobs have IDs."""
    seen_commands: list[str] = []

    def fake_runner(command: str):
        seen_commands.append(command)
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 555\n", stderr="")

    executor = LaunchExecutor(runner=fake_runner, printer=lambda *_: None)
    assert build_afterok_dependency({}, ["missing"]) == ""
    plan = LaunchPlan(commands=(LaunchCommand(label="later", command="sbatch bash/run.sh"),))

    results = executor.execute(plan)

    assert seen_commands == ["sbatch bash/run.sh"]
    assert results[0].job_id == "555"
