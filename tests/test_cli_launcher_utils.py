"""Launcher utility contract tests."""

from types import SimpleNamespace

from gecco.cli.launcher_utils import (
    LaunchCommand,
    LaunchExecutor,
    LaunchPlan,
    SubmissionResult,
    build_afterok_dependency,
    _parse_job_id,
)


def test_launch_plan_executes_in_order_and_propagates_dependencies():
    """A launch plan should preserve order and chain afterok dependencies."""
    seen_commands: list[str] = []

    def fake_runner(command: str):
        seen_commands.append(command)
        job_id = str(100 + len(seen_commands))
        return SimpleNamespace(returncode=0, stdout=f"Submitted batch job {job_id}\n", stderr="")

    executor = LaunchExecutor(runner=fake_runner, printer=lambda *_: None)
    plan = LaunchPlan(
        commands=(
            LaunchCommand(label="server", command="sbatch bash/run_generator.sh"),
            LaunchCommand(
                label="clients",
                command="sbatch {dependency} bash/run_clients.sh",
                dependency_labels=("server",),
            ),
        )
    )

    results = plan.execute(executor)

    assert seen_commands == [
        "sbatch bash/run_generator.sh",
        "sbatch --dependency=afterok:101 bash/run_clients.sh",
    ]
    assert [result.job_id for result in results] == ["101", "102"]
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


def test_launch_executor_prefers_current_pipeline_dependencies_over_prior_lane_snapshot():
    """Lane-chained jobs should keep using the current pipeline's submitted job ids."""
    seen_commands: list[str] = []

    def fake_runner(command: str):
        seen_commands.append(command)
        job_id = str(100 + len(seen_commands))
        return SimpleNamespace(returncode=0, stdout=f"Submitted batch job {job_id}\n", stderr="")

    executor = LaunchExecutor(runner=fake_runner, printer=lambda *_: None)
    prior_results = {
        "client_array": SubmissionResult(label="client_array", command="", submitted=True, job_id="800"),
        "test_evaluation": SubmissionResult(label="test_evaluation", command="", submitted=True, job_id="900"),
        "orchestrator": SubmissionResult(label="orchestrator", command="", submitted=True, job_id="901"),
    }
    plan = LaunchPlan(
        commands=(
            LaunchCommand(
                label="client_array",
                command="sbatch {dependency} bash/run_clients.sh",
                dependency_labels=("test_evaluation", "orchestrator"),
                dependency_policy="afterany",
            ),
            LaunchCommand(
                label="test_evaluation",
                command="sbatch {dependency} bash/run_test_evaluation.sh",
                dependency_labels=("client_array",),
                dependency_policy="afterok",
                required_dependency_labels=("client_array",),
            ),
        )
    )

    results = executor.execute(plan, prior_results=prior_results)

    assert seen_commands == [
        "sbatch --dependency=afterany:900:901 bash/run_clients.sh",
        "sbatch --dependency=afterok:101 bash/run_test_evaluation.sh",
    ]
    assert [result.job_id for result in results] == ["101", "102"]
