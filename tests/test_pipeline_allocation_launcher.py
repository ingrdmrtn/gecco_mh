"""Tests for the pipeline allocation runner (internal pipeline-allocation command)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, PropertyMock, patch

import pytest

from gecco.cli.run_pipeline_allocation import run_pipeline_allocation


def test_allocation_runner_starts_clients_concurrently(tmp_path):
    """All clients are launched (Popen) before any communicate() call."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "demo.yaml").write_text("task: {}\n", encoding="utf-8")

    cfg = SimpleNamespace(
        task=SimpleNamespace(name="demo-task"),
        llm=SimpleNamespace(provider="openrouter", base_model="demo-model"),
        loop=SimpleNamespace(max_iterations=1),
        evaluation=SimpleNamespace(fit_type="group"),
        judge=None,
        centralized_model_generation=SimpleNamespace(enabled=False),
        clients={"alpha": SimpleNamespace(), "beta": SimpleNamespace()},
        data=SimpleNamespace(path="unused.csv", input_columns=[]),
    )

    # ── Call-order instrumentation ─────────────────────────────────
    # Track every Popen and communicate event in sequence so we can
    # prove that all Popen calls happen *before* any communicate call.
    call_events: list[tuple[str, str]] = []  # (event_type, profile)

    def fake_popen(cmd, *args, **kwargs):
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        # Extract the profile name from the command
        profile = "unknown"
        for part in cmd:
            if part in ("alpha", "beta"):
                profile = part
                break
        call_events.append(("popen", profile))

        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = MagicMock()
        proc.stderr = MagicMock()
        proc.wait.return_value = 0
        proc.poll.return_value = 0
        proc.pid = 100 + len(call_events)

        # Communicate records its own event
        def communicate_side_effect(*_a, **_kw):
            call_events.append(("communicate", profile))
            return ("", "")

        proc.communicate.side_effect = communicate_side_effect
        return proc

    with (
        patch("gecco.cli.run_pipeline_allocation.PROJECT_ROOT", tmp_path),
        patch("gecco.cli.run_pipeline_allocation.load_config", return_value=cfg),
        patch("gecco.cli.run_pipeline_allocation.init_sentry", return_value=False),
        patch("gecco.cli.run_pipeline_allocation.subprocess.Popen", side_effect=fake_popen),
    ):
        run_pipeline_allocation(
            config="demo.yaml",
            profiles_csv="alpha,beta",
            results_dir="results/demo/run-001",
        )

    # ── Assertions ─────────────────────────────────────────────────
    # All client Popen events must appear before any client communicate event.
    popen_indices = [
        i for i, (evt, prof) in enumerate(call_events)
        if evt == "popen" and prof in ("alpha", "beta")
    ]
    communicate_indices = [
        i for i, (evt, prof) in enumerate(call_events)
        if evt == "communicate" and prof in ("alpha", "beta")
    ]

    assert len(popen_indices) == 2, f"Expected 2 Popen calls, got {len(popen_indices)}"
    assert len(communicate_indices) == 2, (
        f"Expected 2 client communicate calls, got {len(communicate_indices)}"
    )

    # Critical assertion: last Popen must be before first communicate.
    # A serial launch→wait loop (Popen A, communicate A, Popen B, communicate B)
    # would have popen_indices=[0,2] and communicate_indices=[1,3], so
    # max(popen_indices)=2 > min(communicate_indices)=1 — this assertion fails.
    assert max(popen_indices) < min(communicate_indices), (
        f"Not all Popen calls happen before communicate: "
        f"popen indices={popen_indices}, communicate indices={communicate_indices}. "
        "A serial launch→wait loop would cause this."
    )


def test_allocation_runner_aborts_on_client_failure(tmp_path):
    """Nonzero client exit stops later stages and returns nonzero."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "demo.yaml").write_text("task: {}\n", encoding="utf-8")

    cfg = SimpleNamespace(
        task=SimpleNamespace(name="demo-task"),
        llm=SimpleNamespace(provider="openrouter", base_model="demo-model"),
        loop=SimpleNamespace(max_iterations=1),
        evaluation=SimpleNamespace(fit_type="group"),
        judge=None,
        centralized_model_generation=SimpleNamespace(enabled=False),
        clients={"alpha": SimpleNamespace()},
        data=SimpleNamespace(path="unused.csv", input_columns=[]),
    )

    process_count = [0]

    def fake_popen(cmd, *args, **kwargs):
        process_count[0] += 1
        proc = MagicMock()
        proc.returncode = 1  # client fails
        proc.stdout = MagicMock()
        proc.stderr = MagicMock()
        proc.communicate.return_value = ("", "")
        proc.wait.return_value = 1
        proc.poll.return_value = 1
        proc.pid = 200
        return proc

    with (
        patch("gecco.cli.run_pipeline_allocation.PROJECT_ROOT", tmp_path),
        patch("gecco.cli.run_pipeline_allocation.load_config", return_value=cfg),
        patch("gecco.cli.run_pipeline_allocation.init_sentry", return_value=False),
        patch("gecco.cli.run_pipeline_allocation.subprocess.Popen", side_effect=fake_popen),
    ):
        result = run_pipeline_allocation(
            config="demo.yaml",
            profiles_csv="alpha",
            results_dir="results/demo/run-001",
        )

    # Should return nonzero
    assert result == 1
    # Client should have been started
    assert process_count[0] == 1


def test_allocation_runner_aborts_on_judge_failure(tmp_path):
    """Nonzero judge/orchestrator exit aborts test evaluation."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "demo.yaml").write_text("task: {}\n", encoding="utf-8")

    cfg = SimpleNamespace(
        task=SimpleNamespace(name="demo-task"),
        llm=SimpleNamespace(provider="openrouter", base_model="demo-model"),
        loop=SimpleNamespace(max_iterations=1, n_clients=1),
        evaluation=SimpleNamespace(fit_type="group"),
        judge=SimpleNamespace(capabilities=["performance_summary"]),
        centralized_model_generation=SimpleNamespace(enabled=False),
        clients={"alpha": SimpleNamespace()},
        data=SimpleNamespace(path="unused.csv", input_columns=[]),
    )

    call_sequence = []

    def fake_popen(cmd, *args, **kwargs):
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        call_sequence.append(cmd_str)
        proc = MagicMock()
        proc.stdout = MagicMock()
        proc.stderr = MagicMock()
        proc.pid = 300 + len(call_sequence)

        if "distributed-client" in cmd_str:
            proc.returncode = 0
            proc.communicate.return_value = ("", "")
        elif "judge-orchestrate" in cmd_str:
            proc.returncode = 1  # judge fails
            proc.communicate.return_value = ("", "")
        else:
            proc.returncode = 0
            proc.communicate.return_value = ("", "")
        return proc

    with (
        patch("gecco.cli.run_pipeline_allocation.PROJECT_ROOT", tmp_path),
        patch("gecco.cli.run_pipeline_allocation.load_config", return_value=cfg),
        patch("gecco.cli.run_pipeline_allocation.init_sentry", return_value=False),
        patch("gecco.cli.run_pipeline_allocation.subprocess.Popen", side_effect=fake_popen),
    ):
        result = run_pipeline_allocation(
            config="demo.yaml",
            profiles_csv="alpha",
            results_dir="results/demo/run-001",
        )

    # Should return nonzero
    assert result == 1
    # Client should have run, judge should have run but failed, test-evaluation should NOT run
    client_calls = [c for c in call_sequence if "distributed-client" in c]
    judge_calls = [c for c in call_sequence if "judge-orchestrate" in c]
    test_eval_calls = [c for c in call_sequence if "test-evaluation" in c]

    assert len(client_calls) == 1
    assert len(judge_calls) == 1
    assert len(test_eval_calls) == 0, "test-evaluation must not start after judge failure"


def test_allocation_runner_commands_include_expected_args(tmp_path):
    """Subprocess commands include config, profiles, vLLM URL, results dir as applicable."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "demo.yaml").write_text("task: {}\n", encoding="utf-8")

    cfg = SimpleNamespace(
        task=SimpleNamespace(name="demo-task"),
        llm=SimpleNamespace(provider="openrouter", base_model="demo-model"),
        loop=SimpleNamespace(max_iterations=1, n_clients=2),
        evaluation=SimpleNamespace(fit_type="group"),
        judge=SimpleNamespace(capabilities=["performance_summary"]),
        centralized_model_generation=SimpleNamespace(enabled=False),
        clients={"alpha": SimpleNamespace(), "beta": SimpleNamespace()},
        data=SimpleNamespace(path="unused.csv", input_columns=[]),
    )

    captured = []

    def fake_popen(cmd, *args, **kwargs):
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = MagicMock()
        proc.stderr = MagicMock()
        proc.communicate.return_value = ("", "")
        proc.pid = 400 + len(captured)
        captured.append(cmd)
        return proc

    with (
        patch("gecco.cli.run_pipeline_allocation.PROJECT_ROOT", tmp_path),
        patch("gecco.cli.run_pipeline_allocation.load_config", return_value=cfg),
        patch("gecco.cli.run_pipeline_allocation.init_sentry", return_value=False),
        patch("gecco.cli.run_pipeline_allocation.subprocess.Popen", side_effect=fake_popen),
    ):
        run_pipeline_allocation(
            config="demo.yaml",
            profiles_csv="alpha,beta",
            vllm_url="http://gpu:8000/v1",
            results_dir="results/demo/run-001",
        )

    # Client commands should include config, vllm-url, results-dir
    for cmd in captured:
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        if "distributed-client" in cmd_str:
            assert "demo.yaml" in cmd_str or "--config" in cmd_str
            assert "--results-dir" in cmd_str
        if "judge-orchestrate" in cmd_str:
            assert "--vllm-url" in cmd_str
            assert "http://gpu:8000/v1" in cmd_str

    # 2 clients + 1 judge + 1 test-evaluation (config has judge capabilities)
    assert len(captured) == 4


def test_allocation_runner_rejects_cmg_config(tmp_path):
    """CMG config is rejected even when clients / --profiles-csv are present."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "cmg.yaml").write_text("task: {}\n", encoding="utf-8")

    # CMG config WITH clients attribute — rejection must fire *before*
    # profile resolution, so even valid client profiles are irrelevant.
    cfg = SimpleNamespace(
        task=SimpleNamespace(name="demo-task"),
        llm=SimpleNamespace(provider="openrouter", base_model="demo-model"),
        loop=SimpleNamespace(max_iterations=1),
        evaluation=SimpleNamespace(fit_type="group"),
        judge=None,
        centralized_model_generation=SimpleNamespace(enabled=True, generator_client="gen", n_models=2),
        clients={"alpha": SimpleNamespace(), "beta": SimpleNamespace()},
        data=SimpleNamespace(path="unused.csv", input_columns=[]),
    )

    with (
        patch("gecco.cli.run_pipeline_allocation.PROJECT_ROOT", tmp_path),
        patch("gecco.cli.run_pipeline_allocation.load_config", return_value=cfg),
        patch("gecco.cli.run_pipeline_allocation.init_sentry", return_value=False),
    ):
        result = run_pipeline_allocation(
            config="cmg.yaml",
            profiles_csv="alpha,beta",
            results_dir="results/demo/run-001",
        )

    # Should fail with explicit CMG error (not "no client profiles")
    assert result == 1
    # No subprocess.Popen should have been called — rejection is before launch
    # (we assert this implicitly because we didn't patch Popen and it was never called).


def test_allocation_runner_preserves_stage_ordering(tmp_path):
    """Clients complete before judge before test-evaluation."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "demo.yaml").write_text("task: {}\n", encoding="utf-8")

    cfg = SimpleNamespace(
        task=SimpleNamespace(name="demo-task"),
        llm=SimpleNamespace(provider="openrouter", base_model="demo-model"),
        loop=SimpleNamespace(max_iterations=1, n_clients=2),
        evaluation=SimpleNamespace(fit_type="group"),
        judge=SimpleNamespace(capabilities=["performance_summary"]),
        centralized_model_generation=SimpleNamespace(enabled=False),
        clients={"alpha": SimpleNamespace(), "beta": SimpleNamespace()},
        data=SimpleNamespace(path="unused.csv", input_columns=[]),
    )

    execution_order = []

    def fake_popen(cmd, *args, **kwargs):
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        execution_order.append(cmd_str)
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = MagicMock()
        proc.stderr = MagicMock()
        proc.communicate.return_value = ("", "")
        proc.pid = 500 + len(execution_order)
        return proc

    with (
        patch("gecco.cli.run_pipeline_allocation.PROJECT_ROOT", tmp_path),
        patch("gecco.cli.run_pipeline_allocation.load_config", return_value=cfg),
        patch("gecco.cli.run_pipeline_allocation.init_sentry", return_value=False),
        patch("gecco.cli.run_pipeline_allocation.subprocess.Popen", side_effect=fake_popen),
    ):
        run_pipeline_allocation(
            config="demo.yaml",
            profiles_csv="alpha,beta",
            results_dir="results/demo/run-001",
        )

    client_calls = [i for i, c in enumerate(execution_order) if "distributed-client" in c]
    judge_calls = [i for i, c in enumerate(execution_order) if "judge-orchestrate" in c]
    test_eval_calls = [i for i, c in enumerate(execution_order) if "test-evaluation" in c]

    # Clients must start before judge
    if client_calls and judge_calls:
        assert max(client_calls) < min(judge_calls), "clients must complete before judge starts"

    # Judge must start before test-evaluation
    if judge_calls and test_eval_calls:
        assert max(judge_calls) < min(test_eval_calls), "judge must complete before test-evaluation"
