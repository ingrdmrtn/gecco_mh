"""Shared helpers for GeCCo launcher CLIs."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


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
    ):
        self.runner = runner or _default_runner
        self.printer = printer

    def submit(self, command: LaunchCommand, dry_run: bool = False) -> SubmissionResult:
        """Run one command or print it in dry-run mode."""
        self.printer(f"  $ {command.command}")
        if dry_run:
            return SubmissionResult(
                label=command.label,
                command=command.command,
                submitted=False,
            )

        result = self.runner(command.command)
        stdout = getattr(result, "stdout", "") or ""
        stderr = getattr(result, "stderr", "") or ""
        if getattr(result, "returncode", 0) != 0:
            self.printer(f"  ERROR: {str(stderr).strip()}")
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
        for command in plan.commands:
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


def _default_runner(command: str):
    return subprocess.run(command, shell=True, capture_output=True, text=True)
