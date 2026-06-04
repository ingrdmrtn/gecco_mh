"""Focused tests for Phase 5 DuckDB canonical runtime state fixes."""

from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from gecco.baseline import fit_baseline_if_needed
from gecco.coordination import SharedRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_ROOT = PROJECT_ROOT / "gecco-mh-dashboard"

if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))


def _load_dashboard_module(module_name: str, relative_path: str):
    """Load a dashboard module directly from its file path."""
    module_path = DASHBOARD_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dashboard_data_adapter = _load_dashboard_module(
    "phase5_dashboard_data_adapter", "dashboard/data_adapter.py"
)
dashboard_config = _load_dashboard_module(
    "phase5_dashboard_config", "dashboard/config.py"
)


@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    """Return a fresh registry path for each test."""
    return tmp_path / "shared_registry"


@pytest.fixture
def registry(registry_path: Path) -> SharedRegistry:
    """Create a fresh DuckDB-backed registry."""
    return SharedRegistry(registry_path)


def test_shared_registry_read_path_uses_read_only_duckdb_and_skips_schema_ddl(
    registry: SharedRegistry,
):
    """Reads should use a read-only DuckDB connection and skip schema DDL."""
    connect_calls: list[bool] = []

    real_connect = sys.modules["gecco.coordination"].duckdb.connect

    def _wrapped_connect(*args, **kwargs):
        connect_calls.append(bool(kwargs.get("read_only", False)))
        return real_connect(*args, **kwargs)

    with patch("gecco.coordination.create_schema") as create_schema_mock:
        with patch("gecco.coordination.duckdb.connect", side_effect=_wrapped_connect):
            registry.read()

    assert connect_calls == [True]
    create_schema_mock.assert_not_called()


def test_shared_registry_multi_process_reads_do_not_create_schema(
    registry: SharedRegistry,
):
    """Concurrent readers should not attempt schema creation."""
    registry.update(
        client_id=0,
        iteration=0,
        results=[{"function_name": "m0", "metric_value": 1.0}],
        status="complete",
    )

    with patch("gecco.coordination.create_schema", side_effect=AssertionError("read path must not run DDL")):
        with ThreadPoolExecutor(max_workers=4) as executor:
            snapshots = list(executor.map(lambda _: registry.read(), range(4)))

    assert all(snapshot["iteration_history"][0]["iteration"] == 0 for snapshot in snapshots)


def test_normal_registry_writes_do_not_recreate_schema_after_initialization(
    registry: SharedRegistry,
):
    """Ordinary writes should reuse the initialised store without rerunning DDL."""
    with patch(
        "gecco.coordination.create_schema",
        side_effect=AssertionError("write path must not rerun schema creation"),
    ):
        registry.update(
            client_id=1,
            iteration=0,
            results=[{"function_name": "m1", "metric_value": 2.0}],
            status="complete",
        )

    assert registry.read()["iteration_history"][0]["iteration"] == 0


def test_runtime_iteration_history_counts_use_per_iteration_fields(
    registry: SharedRegistry,
):
    """Historical counts should come from per-iteration rows, not latest client state."""
    registry.update(
        client_id=0,
        iteration=0,
        results=[{"function_name": "m0", "metric_value": 10.0}],
        status="complete",
        had_runnable_model=True,
    )
    registry.update(
        client_id=0,
        iteration=1,
        results=[],
        status="retrying",
        had_runnable_model=False,
    )

    assert registry.count_clients_complete(0) == 1
    assert registry.count_clients_with_models(0) == 1

    row = registry._fetchone(
        "SELECT n_clients_complete, n_clients_with_models "
        "FROM runtime_coordination_view WHERE iteration = ?",
        [0],
    )

    assert row == {"n_clients_complete": 1, "n_clients_with_models": 1}


def test_fit_baseline_if_needed_writes_only_to_registry_and_never_touches_baseline_json(
    registry: SharedRegistry, tmp_path: Path
):
    """Baseline runtime state should be stored only in DuckDB."""
    cfg = SimpleNamespace(
        baseline=SimpleNamespace(model="def baseline_model(data):\n    return data"),
        llm=SimpleNamespace(template_model=None),
    )
    df_train = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    fit_result = {
        "metric_name": "BIC",
        "metric_value": 12.5,
        "param_names": ["alpha"],
        "eval_metrics": [1.0],
        "participant_n_trials": [5],
    }

    with patch(
        "gecco.offline_evaluation.fit_generated_models.run_fit_hierarchical",
        return_value=fit_result,
    ):
        result = fit_baseline_if_needed(
            cfg=cfg,
            df_train=df_train,
            registry=registry,
            id_eval_data=None,
        )

    assert result is not None
    assert registry.read()["baseline"]["metric_value"] == pytest.approx(12.5)
    assert not any(tmp_path.rglob("baseline.json"))
    assert not any(tmp_path.rglob("baseline.lock"))


def test_fit_baseline_if_needed_is_single_fit_under_concurrency(
    registry: SharedRegistry,
):
    """Only one client should perform the baseline fit under contention."""
    cfg = SimpleNamespace(
        baseline=SimpleNamespace(model="def baseline_model(data):\n    return data"),
        llm=SimpleNamespace(template_model=None),
    )
    df_train = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    fit_result = {
        "metric_name": "BIC",
        "metric_value": 12.5,
        "param_names": ["alpha"],
        "eval_metrics": [1.0],
        "participant_n_trials": [5],
    }
    call_count = 0

    def _slow_fit(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        time.sleep(0.1)
        return fit_result

    with patch(
        "gecco.offline_evaluation.fit_generated_models.run_fit_hierarchical",
        side_effect=_slow_fit,
    ):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    lambda _: fit_baseline_if_needed(
                        cfg=cfg,
                        df_train=df_train,
                        registry=registry,
                        id_eval_data=None,
                    ),
                    range(4),
                )
            )

    assert call_count == 1
    assert all(result is not None for result in results)
    assert registry.read()["baseline"]["metric_value"] == pytest.approx(12.5)


def test_baseline_read_after_initial_fit_uses_registry_state(
    registry: SharedRegistry,
):
    """Subsequent reads should reuse the stored baseline from DuckDB."""
    cfg = SimpleNamespace(
        baseline=SimpleNamespace(model="def baseline_model(data):\n    return data"),
        llm=SimpleNamespace(template_model=None),
    )
    df_train = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    fit_result = {
        "metric_name": "BIC",
        "metric_value": 12.5,
        "param_names": ["alpha"],
        "eval_metrics": [1.0],
        "participant_n_trials": [5],
    }

    with patch(
        "gecco.offline_evaluation.fit_generated_models.run_fit_hierarchical",
        return_value=fit_result,
    ):
        first = fit_baseline_if_needed(
            cfg=cfg,
            df_train=df_train,
            registry=registry,
            id_eval_data=None,
        )

    with patch(
        "gecco.offline_evaluation.fit_generated_models.run_fit_hierarchical",
        side_effect=AssertionError("baseline should already be stored"),
    ):
        second = fit_baseline_if_needed(
            cfg=cfg,
            df_train=df_train,
            registry=registry,
            id_eval_data=None,
        )

    assert first is not None
    assert second == registry.read()["baseline"]


def test_shared_registry_callers_use_shared_registry_duckdb_paths(tmp_path: Path):
    """CLI callers should construct registries from shared_registry.duckdb."""
    from gecco.cli.monitor_distributed import load_registry
    from gecco.cli.run_judge_orchestrator import run_orchestrator

    cfg = SimpleNamespace(
        task=SimpleNamespace(name="demo_task"),
        loop=SimpleNamespace(max_iterations=0),
        llm=SimpleNamespace(provider="test-provider", base_model="test-model"),
        judge=SimpleNamespace(barrier=SimpleNamespace(orchestrator_wait_seconds=1, retry_wait_seconds=1)),
        data=SimpleNamespace(
            path="dummy.csv",
            input_columns=["choice_1"],
            id_column="participant",
            splits={"prompt": "[1:2]"},
            data2text_function="narrative",
            narrative_template="trial {choice_1}",
        ),
        centralized_model_generation=SimpleNamespace(enabled=False),
    )

    monitor_results_dir = tmp_path / "monitor"
    monitor_results_dir.mkdir()
    (monitor_results_dir / "shared_registry.duckdb").touch()

    with patch("gecco.cli.monitor_distributed.SharedRegistry") as monitor_registry:
        monitor_registry.open_existing.return_value.read.return_value = {
            "client_entries": {}
        }
        load_registry(monitor_results_dir)

    monitor_registry.open_existing.assert_called_once_with(
        monitor_results_dir / "shared_registry.duckdb"
    )

    mock_registry = MagicMock()
    with patch("gecco.cli.run_judge_orchestrator.load_config", return_value=cfg):
        with patch("gecco.cli.run_judge_orchestrator.SharedRegistry", return_value=mock_registry) as registry_cls:
            with patch("gecco.cli.run_judge_orchestrator.load_llm", return_value=(None, None)):
                with patch("gecco.cli.run_judge_orchestrator.load_data", return_value=MagicMock()):
                    with patch("gecco.cli.run_judge_orchestrator.split_by_participant", return_value={"prompt": MagicMock()}):
                        with patch("gecco.cli.run_judge_orchestrator.get_data2text_function", return_value=lambda *a, **k: "data text"):
                            with patch("gecco.cli.run_judge_orchestrator.init_sentry"):
                                run_orchestrator(config="demo.yaml", results_dir=str(tmp_path / "judge"), n_clients=1)

    registry_cls.assert_called_once_with(str(tmp_path / "judge" / "shared_registry.duckdb"))


def test_dashboard_load_registry_snapshot_reads_shared_registry_duckdb(tmp_path: Path):
    """Dashboard registry loading should read the DuckDB registry."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "shared_registry.duckdb").touch()

    with patch.object(dashboard_data_adapter, "SharedRegistry") as registry_cls:
        registry_cls.open_existing.return_value.read.return_value = {"baseline": None}
        result = dashboard_data_adapter.load_registry_snapshot(results_dir)

    registry_cls.open_existing.assert_called_once_with(results_dir / "shared_registry.duckdb")
    assert result == {"baseline": None}


def test_available_tasks_detects_duckdb_registry_files(tmp_path: Path):
    """Dashboard task discovery should look for shared_registry.duckdb."""
    results_root = tmp_path / "results"
    (results_root / "task_a").mkdir(parents=True)
    (results_root / "task_b").mkdir(parents=True)
    (results_root / "task_a" / "shared_registry.duckdb").touch()
    (results_root / "task_b" / "legacy_registry.json").touch()

    with patch.object(dashboard_config, "project_root", return_value=tmp_path):
        tasks = dashboard_config.available_tasks()

    assert tasks == ["task_a"]
