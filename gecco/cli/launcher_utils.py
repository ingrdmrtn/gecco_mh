"""Shared helpers for GeCCo launcher CLIs."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


DEFAULT_SBATCH_RETRY_ATTEMPTS = 3
DEFAULT_SBATCH_RETRY_BACKOFF_SECONDS = 5.0
DEFAULT_SUBMIT_DELAY_SECONDS = 5.0


@dataclass(slots=True, frozen=True)
class LaunchCommand:
    """A single shell command in a launch plan."""

    label: str
    command: str
    dependency_labels: tuple[str, ...] = ()
    dependency_policy: str = "afterok"
    required_dependency_labels: tuple[str, ...] = ()
    dependency_fallback: str | None = None
    lane_dependency: bool = False


@dataclass(slots=True, frozen=True)
class LaunchPlan:
    """An ordered set of launcher commands."""

    commands: tuple[LaunchCommand, ...]

    def execute(
        self,
        executor: "LaunchExecutor",
        dry_run: bool = False,
        on_result: Callable[[SubmissionResult], None] | None = None,
        prior_results: dict[str, SubmissionResult] | None = None,
    ) -> list[SubmissionResult]:
        """Execute the plan with a launcher executor."""
        return executor.execute(
            self,
            dry_run=dry_run,
            on_result=on_result,
            prior_results=prior_results,
        )


@dataclass(slots=True)
class SubmissionResult:
    """Result of submitting one launch command."""

    label: str
    command: str
    submitted: bool
    job_id: str | None = None
    stdout: str = ""
    stderr: str = ""


def build_afterok_dependency(
    prior_results: Mapping[str, SubmissionResult], dependency_labels: Sequence[str]
) -> str:
    """Build an ``afterok`` dependency string from previous submission results."""
    return build_dependency("afterok", prior_results, dependency_labels)


def build_dependency(
    dependency_policy: str,
    prior_results: Mapping[str, SubmissionResult],
    dependency_labels: Sequence[str],
) -> str:
    """Build a SLURM dependency string from previous submission results."""
    if dependency_policy not in {"afterok", "afterany"}:
        raise ValueError(f"Unsupported dependency policy: {dependency_policy}")
    job_ids = [
        prior_results[label].job_id
        for label in dependency_labels
        if label in prior_results and prior_results[label].job_id
    ]
    return f"--dependency={dependency_policy}:{':'.join(job_ids)}" if job_ids else ""


class LaunchExecutor:
    """Submit launcher commands and parse SBATCH job identifiers."""

    def __init__(
        self,
        runner: Callable[[str], Any] | None = None,
        printer: Callable[[str], None] = print,
        sbatch_retry_attempts: int = DEFAULT_SBATCH_RETRY_ATTEMPTS,
        sbatch_retry_backoff_seconds: float = DEFAULT_SBATCH_RETRY_BACKOFF_SECONDS,
        submit_delay_seconds: float = DEFAULT_SUBMIT_DELAY_SECONDS,
    ):
        self.runner = runner or _default_runner
        self.printer = printer
        self.sbatch_retry_attempts = max(1, sbatch_retry_attempts)
        self.sbatch_retry_backoff_seconds = max(0.0, sbatch_retry_backoff_seconds)
        self.submit_delay_seconds = max(0.0, submit_delay_seconds)

    def submit(self, command: LaunchCommand, dry_run: bool = False) -> SubmissionResult:
        """Run one command or print it in dry-run mode."""
        self.printer(f"  $ {command.command}")
        if dry_run:
            return SubmissionResult(
                label=command.label,
                command=command.command,
                submitted=False,
            )

        for attempt in range(1, self.sbatch_retry_attempts + 1):
            result = self.runner(command.command)
            stdout = getattr(result, "stdout", "") or ""
            stderr = getattr(result, "stderr", "") or ""
            if getattr(result, "returncode", 0) == 0:
                break

            error_text = str(stderr).strip() or str(stdout).strip()
            should_retry = (
                attempt < self.sbatch_retry_attempts
                and _is_retryable_sbatch_submission_error(command.command, error_text)
            )
            if should_retry:
                delay = self.sbatch_retry_backoff_seconds * (2 ** (attempt - 1))
                self.printer(
                    "  WARN: transient sbatch submission failure "
                    f"(attempt {attempt}/{self.sbatch_retry_attempts}); retrying in "
                    f"{delay:g}s: {error_text}"
                )
                time.sleep(delay)
                continue

            self.printer(f"  ERROR: {error_text}")
            raise SystemExit(1)

        stdout = stdout.strip()
        job_id = _parse_job_id(stdout)
        return SubmissionResult(
            label=command.label,
            command=command.command,
            submitted=True,
            job_id=job_id,
            stdout=stdout,
            stderr=str(stderr).strip(),
        )

    def execute(
        self,
        plan: LaunchPlan,
        dry_run: bool = False,
        on_result: Callable[[SubmissionResult], None] | None = None,
        prior_results: dict[str, SubmissionResult] | None = None,
    ) -> list[SubmissionResult]:
        """Run all commands in a plan sequentially."""
        results: list[SubmissionResult] = []
        initial_prior_results = dict(prior_results or {})
        prior_results = dict(initial_prior_results)
        for command_index, command in enumerate(plan.commands):
            if command.required_dependency_labels:
                missing_required = [
                    label
                    for label in command.required_dependency_labels
                    if label not in prior_results or not prior_results[label].job_id
                ]
                if missing_required:
                    continue

            dependency_source = initial_prior_results if command.lane_dependency else prior_results

            dependency_flag = build_dependency(
                command.dependency_policy,
                dependency_source,
                command.dependency_labels,
            )
            if not dependency_flag and command.dependency_fallback:
                dependency_flag = command.dependency_fallback
            effective_command = command.command.replace("{dependency}", dependency_flag)
            result = self.submit(
                LaunchCommand(label=command.label, command=effective_command),
                dry_run=dry_run,
            )
            results.append(result)
            prior_results[result.label] = result
            if on_result is not None:
                on_result(result)
            if not dry_run and self.submit_delay_seconds > 0 and _has_future_submissions(
                plan.commands,
                command_index + 1,
                prior_results,
            ):
                time.sleep(self.submit_delay_seconds)
        return results


def _parse_job_id(output: str) -> str | None:
    output = output.strip()
    if not output:
        return None
    if output.startswith("Submitted batch job "):
        return output.split()[-1]
    first_line = output.splitlines()[0]
    if first_line and first_line[0].isdigit():
        return first_line.split(";", 1)[0]
    return None


def _is_retryable_sbatch_submission_error(command: str, error_text: str) -> bool:
    if not command.lstrip().startswith("sbatch "):
        return False

    normalized = error_text.lower()
    transient_markers = (
        "socket timed out on send/recv operation",
        "temporarily unable to accept job",
        "slurmctld: connection refused",
        "connection reset by peer",
        "unable to contact slurm controller",
    )
    return any(marker in normalized for marker in transient_markers)


def _has_future_submissions(
    commands: Sequence[LaunchCommand],
    start_index: int,
    prior_results: Mapping[str, SubmissionResult],
) -> bool:
    simulated_results = dict(prior_results)
    for command in commands[start_index:]:
        if command.required_dependency_labels:
            missing_required = [
                label
                for label in command.required_dependency_labels
                if label not in simulated_results or not simulated_results[label].job_id
            ]
            if missing_required:
                continue

        simulated_results[command.label] = SubmissionResult(
            label=command.label,
            command=command.command,
            submitted=True,
            job_id="pending",
        )
        return True
    return False


def _default_runner(command: str):
    return subprocess.run(command, shell=True, capture_output=True, text=True)
