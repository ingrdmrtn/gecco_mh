"""Launcher utility contract tests."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

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


def test_launch_executor_retries_transient_sbatch_socket_timeout_and_succeeds():
    """Transient sbatch socket timeouts should be retried before failing."""
    attempts: list[str] = []
    printed: list[str] = []

    def fake_runner(command: str):
        attempts.append(command)
        if len(attempts) == 1:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="sbatch: error: Batch job submission failed: Socket timed out on send/recv operation\n",
            )
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 777\n", stderr="")

    executor = LaunchExecutor(
        runner=fake_runner,
        printer=printed.append,
        sbatch_retry_attempts=3,
        sbatch_retry_backoff_seconds=0.1,
    )

    with patch("gecco.cli.launcher_utils.time.sleep") as sleep_mock:
        result = executor.submit(LaunchCommand(label="clients", command="sbatch bash/run.sh"))

    assert len(attempts) == 2
    assert result.job_id == "777"
    assert any("WARN: transient sbatch submission failure" in line for line in printed)
    assert any("attempt 1/3" in line and "0.1s" in line for line in printed)
    assert any("Socket timed out on send/recv operation" in line for line in printed)
    sleep_mock.assert_called_once_with(0.1)


def test_launch_executor_uses_exponential_backoff_for_repeated_transient_failures():
    """Retry delays should double on each subsequent transient sbatch failure."""
    attempts: list[str] = []
    printed: list[str] = []

    def fake_runner(command: str):
        attempts.append(command)
        if len(attempts) < 3:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="sbatch: error: Batch job submission failed: Socket timed out on send/recv operation\n",
            )
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 777\n", stderr="")

    executor = LaunchExecutor(
        runner=fake_runner,
        printer=printed.append,
        sbatch_retry_attempts=3,
        sbatch_retry_backoff_seconds=0.1,
    )

    with patch("gecco.cli.launcher_utils.time.sleep") as sleep_mock:
        result = executor.submit(LaunchCommand(label="clients", command="sbatch bash/run.sh"))

    assert len(attempts) == 3
    assert result.job_id == "777"
    assert sleep_mock.call_args_list == [((0.1,), {}), ((0.2,), {})]
    assert any("attempt 1/3" in line and "0.1s" in line for line in printed)
    assert any("attempt 2/3" in line and "0.2s" in line for line in printed)
    assert any("Socket timed out on send/recv operation" in line for line in printed)


def test_launch_executor_raises_after_exhausting_transient_sbatch_retries():
    """Persistent transient sbatch failures should still fail after retry budget is exhausted."""
    attempts: list[str] = []
    printed: list[str] = []

    def fake_runner(command: str):
        attempts.append(command)
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="sbatch: error: Batch job submission failed: Socket timed out on send/recv operation\n",
        )

    executor = LaunchExecutor(
        runner=fake_runner,
        printer=printed.append,
        sbatch_retry_attempts=3,
        sbatch_retry_backoff_seconds=0.1,
    )

    with patch("gecco.cli.launcher_utils.time.sleep") as sleep_mock, pytest.raises(SystemExit):
        executor.submit(LaunchCommand(label="clients", command="sbatch bash/run.sh"))

    assert len(attempts) == 3
    assert sum("WARN: transient sbatch submission failure" in line for line in printed) == 2
    assert any("attempt 1/3" in line and "0.1s" in line for line in printed)
    assert any("attempt 2/3" in line and "0.2s" in line for line in printed)
    assert any("ERROR: sbatch: error: Batch job submission failed" in line for line in printed)
    assert sleep_mock.call_args_list == [((0.1,), {}), ((0.2,), {})]


def test_launch_executor_does_not_retry_non_transient_sbatch_failure():
    """Non-transient sbatch failures should fail fast without retrying."""
    attempts: list[str] = []
    printed: list[str] = []

    def fake_runner(command: str):
        attempts.append(command)
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="sbatch: error: Invalid account or account/partition combination specified\n",
        )

    executor = LaunchExecutor(
        runner=fake_runner,
        printer=printed.append,
        sbatch_retry_attempts=3,
        sbatch_retry_backoff_seconds=0.1,
    )

    with patch("gecco.cli.launcher_utils.time.sleep") as sleep_mock, pytest.raises(SystemExit):
        executor.submit(LaunchCommand(label="clients", command="sbatch bash/run.sh"))

    assert attempts == ["sbatch bash/run.sh"]
    assert sleep_mock.call_count == 0
    assert any("ERROR: sbatch: error: Invalid account" in line for line in printed)


def test_launch_executor_does_not_retry_non_sbatch_failure():
    """Failures from non-sbatch commands should not be retried."""
    attempts: list[str] = []
    printed: list[str] = []

    def fake_runner(command: str):
        attempts.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="boom\n")

    executor = LaunchExecutor(
        runner=fake_runner,
        printer=printed.append,
        sbatch_retry_attempts=3,
        sbatch_retry_backoff_seconds=0.1,
    )

    with patch("gecco.cli.launcher_utils.time.sleep") as sleep_mock, pytest.raises(SystemExit):
        executor.submit(LaunchCommand(label="local", command="python script.py"))

    assert attempts == ["python script.py"]
    assert sleep_mock.call_count == 0
    assert any("ERROR: boom" in line for line in printed)


def test_launch_executor_sleeps_between_successful_submissions():
    """Successful submissions should be paced before the next submission."""
    seen_commands: list[str] = []

    def fake_runner(command: str):
        seen_commands.append(command)
        job_id = str(100 + len(seen_commands))
        return SimpleNamespace(returncode=0, stdout=f"Submitted batch job {job_id}\n", stderr="")

    executor = LaunchExecutor(
        runner=fake_runner,
        printer=lambda *_: None,
        submit_delay_seconds=2.0,
    )
    plan = LaunchPlan(
        commands=(
            LaunchCommand(label="server", command="sbatch bash/run_generator.sh"),
            LaunchCommand(label="clients", command="sbatch bash/run_clients.sh"),
        )
    )

    with patch("gecco.cli.launcher_utils.time.sleep") as sleep_mock:
        results = executor.execute(plan)

    assert [result.job_id for result in results] == ["101", "102"]
    sleep_mock.assert_called_once_with(2.0)


def test_launch_executor_does_not_sleep_when_remaining_command_is_skipped():
    """Skipped commands should not create pacing delays."""
    seen_commands: list[str] = []

    def fake_runner(command: str):
        seen_commands.append(command)
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 777\n", stderr="")

    executor = LaunchExecutor(
        runner=fake_runner,
        printer=lambda *_: None,
        submit_delay_seconds=2.0,
    )
    plan = LaunchPlan(
        commands=(
            LaunchCommand(label="server", command="sbatch bash/run_generator.sh"),
            LaunchCommand(
                label="clients",
                command="sbatch bash/run_clients.sh",
                required_dependency_labels=("missing",),
            ),
        )
    )

    with patch("gecco.cli.launcher_utils.time.sleep") as sleep_mock:
        results = executor.execute(plan)

    assert seen_commands == ["sbatch bash/run_generator.sh"]
    assert [result.job_id for result in results] == ["777"]
    sleep_mock.assert_not_called()


def test_launch_executor_skips_submission_in_dry_run():
    """Dry-run mode should print without calling the runner."""
    called = False

    def fake_runner(_command: str):
        nonlocal called
        called = True
        raise AssertionError("runner should not be called in dry run")

    executor = LaunchExecutor(runner=fake_runner, printer=lambda *_: None, submit_delay_seconds=2.0)
    plan = LaunchPlan(
        commands=(
            LaunchCommand(label="server", command="sbatch bash/run_generator.sh"),
            LaunchCommand(label="clients", command="sbatch bash/run_clients.sh"),
        )
    )

    with patch("gecco.cli.launcher_utils.time.sleep") as sleep_mock:
        results = executor.execute(plan, dry_run=True)

    assert called is False
    assert results[0].submitted is False
    assert results[0].job_id is None
    assert results[0].command == "sbatch bash/run_generator.sh"
    sleep_mock.assert_not_called()
