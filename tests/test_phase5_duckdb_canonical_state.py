"""Focused tests for Phase 5 DuckDB canonical runtime state fixes."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
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


def _load_project_module(module_name: str, relative_path: str):
    """Load a project module directly from its file path."""
    module_path = PROJECT_ROOT / relative_path
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
test_fit_model_script = _load_project_module(
    "phase5_test_fit_model", "scripts/test_fit_model.py"
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


def test_abort_state_round_trips_across_registry_handles(registry: SharedRegistry):
    """Abort metadata should be visible from a separate registry handle."""
    registry.request_abort(
        client_id="client-a",
        iteration=7,
        reason="RuntimeError: client crashed",
    )

    other_handle = SharedRegistry.open_existing(registry.registry_path)
    abort = other_handle.get_abort()

    assert abort is not None
    assert abort["client_id"] == "client-a"
    assert abort["iteration"] == 7
    assert abort["reason"] == "RuntimeError: client crashed"
    assert abort["status"] == "failed"
    assert other_handle.read()["abort"] == abort
    with pytest.raises(RuntimeError, match="client-a"):
        other_handle.raise_if_aborted()


def test_request_abort_does_not_overwrite_existing_abort(registry: SharedRegistry):
    """The first abort record should remain authoritative once written."""
    registry.request_abort(
        client_id="client-a",
        iteration=7,
        reason="RuntimeError: client crashed",
    )

    other_handle = SharedRegistry.open_existing(registry.registry_path)
    other_handle.request_abort(
        client_id="client-b",
        iteration=8,
        reason="ValueError: later failure",
    )

    abort = registry.get_abort()
    assert abort is not None
    assert abort["client_id"] == "client-a"
    assert abort["iteration"] == 7
    assert abort["reason"] == "RuntimeError: client crashed"


def test_raise_if_aborted_is_noop_without_abort(registry: SharedRegistry):
    """Absent abort state should keep registry waits available."""
    assert registry.get_abort() is None
    assert registry.raise_if_aborted() is None


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


def test_fit_baseline_if_needed_writes_only_to_registry_and_never_touches_baseline_duckdb(
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
    assert not any(tmp_path.rglob("baseline.duckdb"))
    assert not any(tmp_path.rglob("baseline.duckdb.lock"))


def test_fit_baseline_if_needed_only_fits_training_data(
    registry: SharedRegistry,
):
    """Baseline fitting should only use the training split."""
    cfg = SimpleNamespace(
        baseline=SimpleNamespace(model="def baseline_model(data):\n    return data"),
        llm=SimpleNamespace(template_model=None),
    )
    df_train = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    train_fit_result = {
        "metric_name": "BIC",
        "metric_value": 12.5,
        "param_names": ["alpha"],
        "eval_metrics": [1.0],
        "participant_n_trials": [5],
    }
    calls: list[pd.DataFrame] = []

    def _run_fit(df, code, *, cfg, expected_func_name):
        assert code.startswith("def baseline_model")
        assert expected_func_name == "baseline_model"
        calls.append(df)
        if df is df_train:
            return train_fit_result
        raise AssertionError("unexpected data frame passed to run_fit")

    with patch(
        "gecco.offline_evaluation.fit_generated_models.run_fit_hierarchical",
        side_effect=_run_fit,
    ):
        result = fit_baseline_if_needed(
            cfg=cfg,
            df_train=df_train,
            registry=registry,
            id_eval_data=None,
        )

    assert result is not None
    assert calls == [df_train]
    assert result["metric_value"] == pytest.approx(12.5)
    assert result["val_mean_nll"] is None
    assert registry.read()["baseline"]["val_mean_nll"] is None


def test_fit_baseline_if_needed_training_only_result_keeps_val_nll_none(
    registry: SharedRegistry,
):
    """Baseline fitting should still succeed without a validation split."""
    cfg = SimpleNamespace(
        baseline=SimpleNamespace(model="def baseline_model(data):\n    return data"),
        llm=SimpleNamespace(template_model=None),
    )
    df_train = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    train_fit_result = {
        "metric_name": "BIC",
        "metric_value": 12.5,
        "param_names": ["alpha"],
        "eval_metrics": [1.0],
        "participant_n_trials": [5],
    }
    calls: list[pd.DataFrame] = []

    def _run_fit(df, code, *, cfg, expected_func_name):
        assert code.startswith("def baseline_model")
        assert expected_func_name == "baseline_model"
        calls.append(df)
        if df is df_train:
            return train_fit_result
        raise AssertionError("unexpected data frame passed to run_fit")

    with patch(
        "gecco.offline_evaluation.fit_generated_models.run_fit_hierarchical",
        side_effect=_run_fit,
    ):
        result = fit_baseline_if_needed(
            cfg=cfg,
            df_train=df_train,
            registry=registry,
            id_eval_data=None,
        )

    assert result is not None
    assert calls == [df_train]
    assert result["val_mean_nll"] is None
    assert registry.read()["baseline"]["val_mean_nll"] is None


def test_fit_baseline_if_needed_without_validation_keeps_val_nll_none(
    registry: SharedRegistry,
):
    """Baseline fitting should still succeed when no validation split is provided."""
    cfg = SimpleNamespace(
        baseline=SimpleNamespace(model="def baseline_model(data):\n    return data"),
        llm=SimpleNamespace(template_model=None),
    )
    df_train = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    fit_result = {
        "metric_name": "BIC",
        "metric_value": 12.5,
        "mean_nll": 1.25,
        "param_names": ["alpha"],
        "eval_metrics": [1.0],
        "participant_n_trials": [5],
    }

    with patch(
        "gecco.offline_evaluation.fit_generated_models.run_fit_hierarchical",
        return_value=fit_result,
    ) as run_fit_mock:
        result = fit_baseline_if_needed(
            cfg=cfg,
            df_train=df_train,
            registry=registry,
            id_eval_data=None,
        )

    assert result is not None
    run_fit_mock.assert_called_once()
    assert result["val_mean_nll"] is None
    assert registry.read()["baseline"]["val_mean_nll"] is None


def test_fit_baseline_if_needed_persists_executable_function_name(
    registry: SharedRegistry,
):
    """Baseline fits should persist the executable function name separately."""
    cfg = SimpleNamespace(
        baseline=SimpleNamespace(model="def hybrid_model(data):\n    return data"),
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
    assert result["function_name"] == "baseline_model"
    assert result["executable_function_name"] == "hybrid_model"
    assert registry.read()["baseline"]["executable_function_name"] == "hybrid_model"


def test_cached_baseline_read_includes_executable_function_name(
    registry: SharedRegistry,
):
    """Cached baseline reads should preserve the executable function name."""
    registry.set_baseline(
        {
            "function_name": "baseline_model",
            "executable_function_name": "hybrid_model",
            "metric_name": "BIC",
            "metric_value": 12.5,
            "param_names": ["alpha"],
            "eval_metrics": [1.0],
            "code": "def hybrid_model(data):\n    return data",
        }
    )

    snapshot = registry.read()
    assert snapshot["baseline"]["function_name"] == "baseline_model"
    assert snapshot["baseline"]["executable_function_name"] == "hybrid_model"

    cfg = SimpleNamespace(
        baseline=SimpleNamespace(model="def hybrid_model(data):\n    return data"),
        llm=SimpleNamespace(template_model=None),
    )

    with patch(
        "gecco.offline_evaluation.fit_generated_models.run_fit_hierarchical",
        side_effect=AssertionError("baseline should be loaded from cache"),
    ):
        cached = fit_baseline_if_needed(
            cfg=cfg,
            df_train=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
            registry=registry,
            id_eval_data=None,
        )

    assert cached is not None
    assert cached["function_name"] == "baseline_model"
    assert cached["executable_function_name"] == "hybrid_model"


def test_fit_baseline_if_needed_uses_dict_baseline_model(
    registry: SharedRegistry,
):
    """Dict-style baseline config should prefer baseline.model over template_model."""
    cfg = SimpleNamespace(
        baseline={"model": "def seven_param_model(a, b, c, d, e, f, g):\n    return a"},
        llm=SimpleNamespace(template_model="def template_model(x, y):\n    return x"),
    )
    df_train = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    train_fit_result = {
        "metric_name": "BIC",
        "metric_value": 12.5,
        "param_names": ["alpha"],
        "eval_metrics": [1.0],
        "participant_n_trials": [5],
    }
    calls: list[pd.DataFrame] = []

    def _run_fit(df, code, *, cfg, expected_func_name):
        assert code.startswith("def seven_param_model")
        assert expected_func_name == "seven_param_model"
        assert "template_model" not in code
        calls.append(df)
        if df is df_train:
            return train_fit_result
        raise AssertionError("unexpected data frame passed to run_fit")

    with patch(
        "gecco.offline_evaluation.fit_generated_models.run_fit_hierarchical",
        side_effect=_run_fit,
    ):
        result = fit_baseline_if_needed(
            cfg=cfg,
            df_train=df_train,
            registry=registry,
            id_eval_data=None,
        )

    assert result is not None
    assert calls == [df_train]
    assert result["metric_value"] == pytest.approx(12.5)
    assert result["val_mean_nll"] is None
    assert registry.read()["baseline"]["val_mean_nll"] is None


def test_run_distributed_client_splits_prompt_train_and_test_only(
    tmp_path: Path, monkeypatch
):
    """The distributed entrypoint should split non-prompt participants into train/test only."""
    from gecco.cli import run_gecco_distributed as dist

    project_root = tmp_path / "project_root"
    monkeypatch.setattr(dist, "PROJECT_ROOT", project_root)

    df = pd.DataFrame(
        {
            "participant_id": [0, 1, 2, 3, 4],
            "value": [10, 11, 12, 13, 14],
        }
    )
    prompt_df = df.iloc[[0]].copy()
    cfg = SimpleNamespace(
        task=SimpleNamespace(name="demo"),
        llm=SimpleNamespace(provider="provider", base_model="model", base_url=None),
        loop=SimpleNamespace(max_independent_runs=0, max_iterations=1),
        data=SimpleNamespace(
            path="unused",
            input_columns=[],
            id_column="participant_id",
            splits=[],
            data2text_function="noop",
            narrative_template="template",
            max_prompt_trials=None,
            value_mappings=None,
        ),
        evaluation=SimpleNamespace(fit_type="group", split_seed=42),
        metadata=SimpleNamespace(flag=False),
        baseline=SimpleNamespace(model="def baseline_model(data):\n    return data"),
    )
    registry_mock = MagicMock()
    registry_mock.request_abort = MagicMock()
    registry_mock.set_client_status = MagicMock()
    search_seen: dict[str, pd.DataFrame] = {}

    class FakeSearch:
        def __init__(self, model, tokenizer, cfg, df, prompt_builder, **kwargs):
            search_seen["search_df"] = df
            search_seen["search_kwargs"] = kwargs
            self.best_iter = -1
            self.results_dir = tmp_path / "results" / "demo"

        def close(self):
            return None

    def _fit_baseline_if_needed(*, cfg, df_train, registry, id_eval_data=None):
        search_seen["baseline_df_train"] = df_train
        return {"metric_name": "BIC", "metric_value": 1.0}

    monkeypatch.setattr(dist, "configure_temp_dirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(dist, "resolve_config_path", lambda config, project_root=None: config)
    monkeypatch.setattr(dist, "load_config", lambda path: cfg)
    monkeypatch.setattr(dist, "init_sentry", lambda **kwargs: None)
    monkeypatch.setattr(dist, "load_data", lambda *args, **kwargs: df)
    monkeypatch.setattr(dist, "split_by_participant", lambda *args, **kwargs: {"prompt": prompt_df})
    monkeypatch.setattr(dist, "get_data2text_function", lambda *args, **kwargs: (lambda *a, **k: "text"))
    monkeypatch.setattr(dist, "PromptBuilderWrapper", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(dist, "load_llm", lambda *args, **kwargs: (MagicMock(), MagicMock()))
    monkeypatch.setattr(dist, "GeCCoModelSearch", FakeSearch)
    monkeypatch.setattr("gecco.baseline.fit_baseline_if_needed", _fit_baseline_if_needed)
    monkeypatch.setattr(dist, "SharedRegistry", lambda *args, **kwargs: registry_mock)
    monkeypatch.setattr(dist, "console", MagicMock())

    result = dist.run_distributed_client(config="dummy.yaml")

    assert result is None
    assert search_seen["baseline_df_train"] is search_seen["search_df"]
    assert len(search_seen["search_df"]) == 2
    assert len(search_seen["baseline_df_train"]) == 2
    assert 0 not in search_seen["search_df"]["participant_id"].tolist()
    assert "df_val" not in search_seen["search_kwargs"]


def test_split_prompt_train_test_excludes_prompt_participants_and_uses_two_way_default_ratios():
    """The production split helper should partition only non-prompt participants."""
    from gecco.cli import run_gecco_distributed as dist

    df = pd.DataFrame(
        {
            "participant_id": [100, 101, 10, 11, 12, 13, 14],
            "value": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    prompt_df = df.iloc[[0, 1]].copy()
    data_cfg = SimpleNamespace(id_column="participant_id", splits=[])
    evaluation_cfg = SimpleNamespace(split_seed=42, train_ratio=0.7, test_ratio=0.3)

    with patch.object(dist, "split_by_participant", return_value={"prompt": prompt_df}) as split_mock:
        result = dist._split_prompt_train_test(df, data_cfg, evaluation_cfg)

    split_mock.assert_called_once_with(df, "participant_id", [])
    assert len(result) == 5

    df_prompt, df_train, df_test, train_ids, test_ids = result
    assert df_prompt.equals(prompt_df)
    assert train_ids == [14, 12, 13]
    assert test_ids == [11, 10]
    assert len(train_ids) == int(5 * 0.7)
    assert set(df_train["participant_id"]) == set(train_ids)
    assert set(df_test["participant_id"]) == set(test_ids)
    assert set(prompt_df["participant_id"]).isdisjoint(df_train["participant_id"])
    assert set(prompt_df["participant_id"]).isdisjoint(df_test["participant_id"])


def test_run_test_evaluation_uses_placeholder_for_missing_baseline_val_nll(
    tmp_path: Path, monkeypatch, capsys
):
    """Old baseline records with missing val NLL should still evaluate cleanly."""
    from gecco.cli import run_test_evaluation as test_eval

    project_root = tmp_path / "project_root"
    monkeypatch.setattr(test_eval, "PROJECT_ROOT", project_root)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    registry_path = results_dir / "shared_registry.duckdb"
    registry_path.touch()

    cfg = SimpleNamespace(
        data=SimpleNamespace(path="unused", input_columns=[], id_column="participant_id", splits=[]),
        evaluation=SimpleNamespace(n_test_models=1),
    )
    fake_registry = MagicMock()
    fake_registry.read.return_value = {
        "iteration_history": [],
        "baseline": {"code": "code", "function_name": "baseline_model", "val_mean_nll": None},
    }
    fit_entry = {
        "model_name": "baseline_model",
        "val_nll": None,
        "test_mean_BIC": 2.0,
        "test_mean_NLL": 3.0,
        "test_individual_BIC": [],
        "test_individual_NLL": [],
        "test_individual_differences": None,
    }

    with patch.object(test_eval, "load_config", return_value=cfg):
        with patch.object(test_eval, "load_splits", return_value=pd.DataFrame()):
            with patch.object(test_eval, "collect_candidates", return_value=[]):
                with patch.object(test_eval, "fit_one_on_test", return_value=fit_entry) as fit_mock:
                    with patch.object(test_eval, "SharedRegistry") as registry_cls:
                        registry_cls.open_existing.return_value = fake_registry
                        registry_cls.side_effect = AssertionError("constructor path must not be used")

                        result = test_eval.run_test_evaluation(
                            config="unused",
                            results_dir=str(results_dir),
                            write_store=False,
                        )

    captured = capsys.readouterr().out
    assert result is None
    fit_mock.assert_called_once()
    assert fit_mock.call_args.args[0]["client_id"] == "baseline"
    assert "val_nll=n/a" in captured


def test_run_test_evaluation_load_splits_returns_test_partition(
    tmp_path: Path, monkeypatch
):
    """The test-evaluation helper should return the held-out test partition."""
    from gecco.cli import run_test_evaluation as test_eval

    project_root = tmp_path / "project_root"
    monkeypatch.setattr(test_eval, "PROJECT_ROOT", project_root)

    df = pd.DataFrame(
        {
            "participant_id": [0, 1, 2, 3, 4],
            "value": [10, 11, 12, 13, 14],
        }
    )
    prompt_df = df.iloc[[0]].copy()
    train_df = df.iloc[[1, 2]].copy()
    test_df = df.iloc[[3, 4]].copy()
    cfg = SimpleNamespace(
        data=SimpleNamespace(path="unused", input_columns=[], id_column="participant_id", splits=[]),
        evaluation=SimpleNamespace(split_seed=42),
    )

    with patch.object(test_eval, "load_data", return_value=df):
        with patch.object(
            test_eval,
            "_split_prompt_train_test",
            return_value=(prompt_df, train_df, test_df, [1, 2], [3, 4]),
        ) as split_mock:
            result = test_eval.load_splits(cfg)

    split_mock.assert_called_once()
    assert result.equals(test_df)


def test_collect_candidates_ranks_by_development_metric_without_validation_fields():
    """Candidate collection should rank by dev metrics and keep distinct identities."""
    from gecco.cli import run_test_evaluation as test_eval

    registry = MagicMock()
    registry.read.return_value = {
        "iteration_history": [
            {
                "client_id": 0,
                "iteration": 0,
                "results": [
                    {
                        "function_name": "shared_model",
                        "metric_value": 9.0,
                        "code": "def shared_model():\n    return 1",
                        "param_names": ["alpha"],
                    },
                    {
                        "function_name": "shared_model",
                        "metric_value": 7.0,
                        "code": "def shared_model():\n    return 1",
                        "param_names": ["alpha"],
                    },
                ],
            },
            {
                "client_id": 1,
                "iteration": 0,
                "results": [
                    {
                        "function_name": "shared_model",
                        "metric_value": 8.0,
                        "code": "def shared_model():\n    return 3",
                        "param_names": ["alpha"],
                    },
                    {
                        "function_name": "unique_model",
                        "metric_value": 6.0,
                        "code": "def unique_model():\n    return 4",
                        "param_names": ["beta"],
                    },
                    {
                        "function_name": "ignored_model",
                        "metric_value": None,
                        "code": "def ignored_model():\n    return 5",
                        "param_names": ["gamma"],
                    },
                ],
            },
        ],
        "baseline": {},
    }
    cfg = SimpleNamespace(evaluation=SimpleNamespace(metric="BIC"))

    candidates = test_eval.collect_candidates(registry, cfg)

    assert len(candidates) == 3
    assert [candidate["selection_metric_value"] for candidate in candidates] == [6.0, 7.0, 8.0]
    assert all("val_mean_nll" not in candidate for candidate in candidates)
    assert [candidate["function_name"] for candidate in candidates] == [
        "unique_model",
        "shared_model",
        "shared_model",
    ]

    nll_registry = MagicMock()
    nll_registry.read.return_value = {
        "iteration_history": [
            {
                "client_id": 2,
                "iteration": 1,
                "results": [
                    {
                        "function_name": "nll_model",
                        "mean_nll": 2.5,
                        "code": "def nll_model():\n    return 1",
                        "param_names": [],
                    }
                ],
            }
        ],
        "baseline": {},
    }
    nll_cfg = SimpleNamespace(evaluation=SimpleNamespace(metric="NLL"))

    nll_candidates = test_eval.collect_candidates(nll_registry, nll_cfg)

    assert len(nll_candidates) == 1
    assert nll_candidates[0]["selection_metric_name"] == "mean_nll"
    assert nll_candidates[0]["selection_metric_value"] == pytest.approx(2.5)


def test_run_test_evaluation_default_evaluates_10_generated_plus_baseline(
    tmp_path: Path, monkeypatch
):
    """Default test-evaluation count should cover 10 generated models plus baseline."""
    from gecco.cli import run_test_evaluation as test_eval

    project_root = tmp_path / "project_root"
    monkeypatch.setattr(test_eval, "PROJECT_ROOT", project_root)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    registry_path = results_dir / "shared_registry.duckdb"
    registry_path.touch()

    cfg = SimpleNamespace(
        data=SimpleNamespace(path="unused", input_columns=[], id_column="participant_id", splits=[]),
        evaluation=SimpleNamespace(),
    )
    fake_registry = MagicMock()
    fake_registry.read.return_value = {
        "iteration_history": [],
        "baseline": {
            "function_name": "baseline_model",
            "executable_function_name": "hybrid_model",
            "code": "def hybrid_model(data):\n    return data",
            "val_mean_nll": 0.5,
            "param_names": [],
        },
    }
    candidate_names = [f"candidate_{idx}" for idx in range(11)]
    fit_calls: list[str] = []

    def _fit_one_on_test(candidate, df_test, cfg, id_eval_data=None):
        fit_calls.append(candidate["client_id"])
        return {"model_name": candidate["function_name"], "val_nll": 0.0, "test_mean_BIC": 1.0, "test_mean_NLL": 2.0, "test_individual_BIC": [], "test_individual_NLL": [], "test_individual_differences": None}

    with patch.object(test_eval, "load_config", return_value=cfg):
        with patch.object(test_eval, "load_splits", return_value=pd.DataFrame()):
            with patch.object(test_eval, "collect_candidates", return_value=[{"client_id": name, "function_name": name, "code": "def f():\n    return 1", "val_mean_nll": float(idx), "param_names": []} for idx, name in enumerate(candidate_names)]):
                with patch.object(test_eval, "SharedRegistry") as registry_cls:
                    registry_cls.open_existing.return_value = fake_registry
                    registry_cls.side_effect = AssertionError("constructor path must not be used")
                    with patch.object(test_eval, "fit_one_on_test", side_effect=_fit_one_on_test):
                        result = test_eval.run_test_evaluation(
                            config="unused",
                            results_dir=str(results_dir),
                            write_store=False,
                        )

    assert result is None
    assert fit_calls == ["baseline", *candidate_names[:10]]


def test_run_test_evaluation_explicit_1_evaluates_1_generated_plus_baseline(
    tmp_path: Path, monkeypatch
):
    """Explicit test-evaluation count should still leave baseline outside the limit."""
    from gecco.cli import run_test_evaluation as test_eval

    project_root = tmp_path / "project_root"
    monkeypatch.setattr(test_eval, "PROJECT_ROOT", project_root)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    registry_path = results_dir / "shared_registry.duckdb"
    registry_path.touch()

    cfg = SimpleNamespace(
        data=SimpleNamespace(path="unused", input_columns=[], id_column="participant_id", splits=[]),
        evaluation=SimpleNamespace(n_test_models=1),
    )
    fake_registry = MagicMock()
    fake_registry.read.return_value = {
        "iteration_history": [],
        "baseline": {
            "function_name": "baseline_model",
            "executable_function_name": "hybrid_model",
            "code": "def hybrid_model(data):\n    return data",
            "val_mean_nll": 0.5,
            "param_names": [],
        },
    }
    candidate_names = ["candidate_0", "candidate_1"]
    fit_calls: list[str] = []

    def _fit_one_on_test(candidate, df_test, cfg, id_eval_data=None):
        fit_calls.append(candidate["client_id"])
        return {"model_name": candidate["function_name"], "val_nll": 0.0, "test_mean_BIC": 1.0, "test_mean_NLL": 2.0, "test_individual_BIC": [], "test_individual_NLL": [], "test_individual_differences": None}

    with patch.object(test_eval, "load_config", return_value=cfg):
        with patch.object(test_eval, "load_splits", return_value=pd.DataFrame()):
            with patch.object(test_eval, "collect_candidates", return_value=[{"client_id": name, "function_name": name, "code": "def f():\n    return 1", "val_mean_nll": float(idx), "param_names": []} for idx, name in enumerate(candidate_names)]):
                with patch.object(test_eval, "SharedRegistry") as registry_cls:
                    registry_cls.open_existing.return_value = fake_registry
                    registry_cls.side_effect = AssertionError("constructor path must not be used")
                    with patch.object(test_eval, "fit_one_on_test", side_effect=_fit_one_on_test):
                        result = test_eval.run_test_evaluation(
                            config="unused",
                            results_dir=str(results_dir),
                            write_store=False,
                        )

    assert result is None
    assert fit_calls == ["baseline", "candidate_0"]


def test_fit_baseline_if_needed_is_single_fit_under_process_contention(
    registry: SharedRegistry,
):
    """Only one process should perform the baseline fit under contention."""
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

    ctx = mp.get_context("fork")
    call_count = ctx.Value("i", 0)

    def _slow_fit(*args, **kwargs):
        with call_count.get_lock():
            call_count.value += 1
        time.sleep(0.1)
        return fit_result

    barrier = ctx.Barrier(5)
    results_queue = ctx.Queue()

    def _worker() -> None:
        try:
            barrier.wait()
            local_registry = SharedRegistry.open_existing(registry.registry_path)
            result = fit_baseline_if_needed(
                cfg=cfg,
                df_train=df_train,
                registry=local_registry,
                id_eval_data=None,
            )
            results_queue.put(("ok", result))
        except Exception as exc:  # pragma: no cover - surfaced via queue assertions
            results_queue.put(("error", repr(exc)))

    with patch(
        "gecco.offline_evaluation.fit_generated_models.run_fit_hierarchical",
        side_effect=_slow_fit,
    ):
        processes = [ctx.Process(target=_worker) for _ in range(4)]
        for process in processes:
            process.start()

        barrier.wait()

        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0

    results = [results_queue.get(timeout=1) for _ in range(4)]
    assert all(status == "ok" for status, _ in results)
    assert all(result is not None for _, result in results)

    assert call_count.value == 1
    assert registry.read()["baseline"]["metric_value"] == pytest.approx(12.5)


def test_run_test_evaluation_opens_existing_registry_read_only(tmp_path: Path, monkeypatch):
    """The test-evaluation pipeline must open the provided registry snapshot read-only."""
    from gecco.cli import run_test_evaluation as test_eval

    project_root = tmp_path / "project_root"
    (project_root / "results").mkdir(parents=True)
    monkeypatch.setattr(test_eval, "PROJECT_ROOT", project_root)

    results_dir = tmp_path / "custom_results"
    results_dir.mkdir()
    registry_path = results_dir / "shared_registry.duckdb"
    registry_path.touch()

    cfg = SimpleNamespace(
        data=SimpleNamespace(path="unused", input_columns=[], id_column="participant_id", splits=[]),
        evaluation=SimpleNamespace(n_test_models=0),
    )
    fake_registry = MagicMock()
    fake_registry.read.return_value = {"iteration_history": [], "baseline": {}}

    with patch.object(test_eval, "load_config", return_value=cfg):
        with patch.object(test_eval, "load_splits", return_value=pd.DataFrame()):
            with patch.object(test_eval, "collect_candidates", return_value=[]):
                with patch.object(test_eval, "SharedRegistry") as registry_cls:
                    registry_cls.open_existing.return_value = fake_registry
                    registry_cls.side_effect = AssertionError("constructor path must not be used")

                    result = test_eval.run_test_evaluation(
                        config="unused",
                        results_dir=str(results_dir),
                        write_store=False,
                    )

    registry_cls.open_existing.assert_called_once_with(registry_path)
    assert result is None


def test_run_test_evaluation_writes_json_csv_and_rich_summary(
    tmp_path: Path, monkeypatch, capsys
):
    """Completed test evaluation should write JSON, CSV, and a Rich summary table."""
    from gecco.cli import run_test_evaluation as test_eval

    project_root = tmp_path / "project_root"
    monkeypatch.setattr(test_eval, "PROJECT_ROOT", project_root)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    registry_path = results_dir / "shared_registry.duckdb"
    registry_path.touch()

    cfg = SimpleNamespace(
        data=SimpleNamespace(path="unused", input_columns=[], id_column="participant_id", splits=[]),
        evaluation=SimpleNamespace(n_test_models=1),
    )
    fake_registry = MagicMock()
    fake_registry.read.return_value = {
        "iteration_history": [
            {
                "client_id": 7,
                "iteration": 3,
                "results": [
                    {
                        "function_name": "descriptive_model",
                        "display_name": "descriptive model",
                        "executable_function_name": "cognitive_model1",
                        "code": "@njit\ndef cognitive_model1(model_parameters):\n    return model_parameters[0]",
                        "val_mean_nll": 4.25,
                        "metric_value": 1.25,
                        "param_names": ["x"],
                        "candidate_index": 0,
                    }
                ],
            }
        ],
        "baseline": {},
    }

    fit_entry = {
        "metric_name": "BIC",
        "metric_value": 2.0,
        "mean_nll": 3.0,
        "eval_metrics": [2.0],
        "per_participant_nll": [3.0],
    }

    def _run_fit(df, code, *, cfg, expected_func_name):
        assert expected_func_name == "cognitive_model1"
        assert code == "@njit\ndef cognitive_model1(model_parameters):\n    return model_parameters[0]"
        return fit_entry

    with patch.object(test_eval, "load_config", return_value=cfg):
        with patch.object(test_eval, "load_splits", return_value=pd.DataFrame()):
            with patch.object(test_eval, "SharedRegistry") as registry_cls:
                registry_cls.open_existing.return_value = fake_registry
                registry_cls.side_effect = AssertionError("constructor path must not be used")
                with patch.object(test_eval, "run_fit", side_effect=_run_fit):
                    result = test_eval.run_test_evaluation(
                        config="unused",
                        results_dir=str(results_dir),
                        write_store=False,
                    )

    captured = capsys.readouterr().out
    json_path = results_dir / "bics" / "top_models_test.json"
    csv_path = results_dir / "bics" / "top_models_test.csv"

    assert result is None
    assert json_path.exists()
    assert csv_path.exists()
    assert "Test evaluation summary" in captured
    assert "Model name" in captured
    assert "descriptive model" in captured

    json_rows = json.loads(json_path.read_text(encoding="utf-8"))
    csv_rows = pd.read_csv(csv_path)

    assert json_rows[0]["model_name"] == "descriptive model"
    assert json_rows[0]["val_nll"] == 4.25
    assert csv_rows.loc[0, "model_name"] == "descriptive model"
    assert csv_rows.loc[0, "executable_function_name"] == "cognitive_model1"
    assert csv_rows.loc[0, "val_nll"] == pytest.approx(4.25)
    assert "test_individual_BIC" not in csv_rows.columns
    assert "test_individual_NLL" not in csv_rows.columns


def test_run_test_evaluation_uses_descriptive_model_name_without_executable_regression(
    tmp_path: Path, monkeypatch
):
    """Recoverable registry names should stay human-facing while execution stays callable-based."""
    from gecco.cli import run_test_evaluation as test_eval

    project_root = tmp_path / "project_root"
    monkeypatch.setattr(test_eval, "PROJECT_ROOT", project_root)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    registry_path = results_dir / "shared_registry.duckdb"
    registry_path.touch()

    code = "@njit\ndef cognitive_model1(model_parameters):\n    return model_parameters[0]"
    cfg = SimpleNamespace(
        data=SimpleNamespace(path="unused", input_columns=[], id_column="participant_id", splits=[]),
        evaluation=SimpleNamespace(n_test_models=1),
    )
    fake_registry = MagicMock()
    fake_registry.read.return_value = {
        "iteration_history": [
            {
                "client_id": 0,
                "iteration": 4,
                "results": [
                    {
                        "function_name": "cognitive_model1",
                        "candidate_index": 2,
                        "code": code,
                        "metric_value": 1.5,
                        "param_names": ["x"],
                    }
                ],
            }
        ],
        "candidate_generations": {
            "4": {
                "candidates": [
                    {
                        "index": 2,
                        "name": "perseveration net",
                        "func_name": "cognitive_model1",
                        "code": code,
                    }
                ]
            }
        },
        "baseline": {},
    }

    fit_entry = {
        "metric_name": "BIC",
        "metric_value": 2.0,
        "mean_nll": 3.0,
        "eval_metrics": [2.0],
        "per_participant_nll": [3.0],
    }

    def _run_fit(df, code, *, cfg, expected_func_name):
        assert expected_func_name == "cognitive_model1"
        assert code == "@njit\ndef cognitive_model1(model_parameters):\n    return model_parameters[0]"
        return fit_entry

    with patch.object(test_eval, "load_config", return_value=cfg):
        with patch.object(test_eval, "load_splits", return_value=pd.DataFrame()):
            with patch.object(test_eval, "SharedRegistry") as registry_cls:
                registry_cls.open_existing.return_value = fake_registry
                registry_cls.side_effect = AssertionError("constructor path must not be used")
                with patch.object(test_eval, "run_fit", side_effect=_run_fit):
                    result = test_eval.run_test_evaluation(
                        config="unused",
                        results_dir=str(results_dir),
                        write_store=False,
                    )

    json_rows = json.loads((results_dir / "bics" / "top_models_test.json").read_text(encoding="utf-8"))
    csv_rows = pd.read_csv(results_dir / "bics" / "top_models_test.csv")

    assert result is None
    assert json_rows[0]["model_name"] == "perseveration net"
    assert csv_rows.loc[0, "model_name"] == "perseveration net"
    assert csv_rows.loc[0, "executable_function_name"] == "cognitive_model1"


def test_run_test_evaluation_uses_executable_name_for_display_named_candidate(
    tmp_path: Path, monkeypatch
):
    """Display names in the registry should not be used as executable names."""
    from gecco.cli import run_test_evaluation as test_eval

    project_root = tmp_path / "project_root"
    monkeypatch.setattr(test_eval, "PROJECT_ROOT", project_root)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    registry_path = results_dir / "shared_registry.duckdb"
    registry_path.touch()

    cfg = SimpleNamespace(
        data=SimpleNamespace(path="unused", input_columns=[], id_column="participant_id", splits=[]),
        evaluation=SimpleNamespace(n_test_models=1),
    )
    fake_registry = MagicMock()
    fake_registry.read.return_value = {
        "iteration_history": [
            {
                "client_id": 0,
                "iteration": 0,
                "results": [
                        {
                            "function_name": "perseveration_net_mf",
                            "code": "@njit\ndef cognitive_model1(model_parameters):\n    x = model_parameters[0]\n    return x",
                            "metric_value": 1.5,
                            "param_names": ["x"],
                        }
                ],
            }
        ],
        "baseline": {},
    }

    fit_entry = {
        "metric_name": "BIC",
        "metric_value": 2.0,
        "mean_nll": 3.0,
        "eval_metrics": [2.0],
        "per_participant_nll": [3.0],
    }

    def _run_fit(df, code, *, cfg, expected_func_name):
        assert expected_func_name == "cognitive_model1"
        assert code == "@njit\ndef cognitive_model1(model_parameters):\n    x = model_parameters[0]\n    return x"
        return fit_entry

    with patch.object(test_eval, "load_config", return_value=cfg):
        with patch.object(test_eval, "load_splits", return_value=pd.DataFrame()):
            with patch.object(test_eval, "SharedRegistry") as registry_cls:
                registry_cls.open_existing.return_value = fake_registry
                registry_cls.side_effect = AssertionError("constructor path must not be used")
                with patch.object(test_eval, "run_fit", side_effect=_run_fit):
                    result = test_eval.run_test_evaluation(
                        config="unused",
                        results_dir=str(results_dir),
                        write_store=False,
                    )

    assert result is None
    assert json.loads((results_dir / "bics" / "top_models_test.json").read_text(encoding="utf-8"))[0]["model_name"] == "perseveration_net_mf"
    assert pd.read_csv(results_dir / "bics" / "top_models_test.csv").loc[0, "model_name"] == "perseveration_net_mf"
    assert pd.read_csv(results_dir / "bics" / "top_models_test.csv").loc[0, "executable_function_name"] == "cognitive_model1"


def test_shared_registry_update_preserves_naming_metadata(registry: SharedRegistry):
    """Registry snapshots should keep naming metadata for later test evaluation."""
    registry.update(
        client_id=1,
        iteration=2,
        results=[
            {
                "function_name": "perseveration_net_mf",
                "display_name": "perseveration_net_mf",
                "executable_function_name": "cognitive_model1",
                "candidate_index": 3,
                "metric_value": 1.0,
                "param_names": ["x"],
            }
        ],
        status="complete",
    )

    snapshot = registry.read()
    result = snapshot["iteration_history"][0]["results"][0]

    assert result["display_name"] == "perseveration_net_mf"
    assert result["executable_function_name"] == "cognitive_model1"
    assert result["candidate_index"] == 3


def test_build_model_spec_falls_back_to_user_defined_function_not_injected_njit():
    """Missing display names should resolve to user code, never injected helpers."""
    from gecco.offline_evaluation.utils import build_model_spec

    code = "@njit\ndef cognitive_model(model_parameters):\n    x = model_parameters[0]\n    return x + 1\n"

    spec = build_model_spec(code, expected_func_name="missing_model")

    assert spec.name == "missing_model"
    assert spec.func.__name__ == "cognitive_model"


def test_run_test_evaluation_does_not_use_default_results_dir(tmp_path: Path, monkeypatch):
    """A stale default results location must not be reused in place of the explicit run."""
    from gecco.cli import run_test_evaluation as test_eval

    project_root = tmp_path / "project_root"
    default_results_dir = project_root / "results" / "test-evaluation"
    default_results_dir.mkdir(parents=True)
    monkeypatch.setattr(test_eval, "PROJECT_ROOT", project_root)

    results_dir = tmp_path / "explicit_results"
    results_dir.mkdir()
    registry_path = results_dir / "shared_registry.duckdb"
    registry_path.touch()

    cfg = SimpleNamespace(
        data=SimpleNamespace(path="unused", input_columns=[], id_column="participant_id", splits=[]),
        evaluation=SimpleNamespace(n_test_models=0),
    )
    fake_registry = MagicMock()
    fake_registry.read.return_value = {"iteration_history": [], "baseline": {}}

    def _open_existing(path):
        assert path == registry_path
        assert path != default_results_dir / "shared_registry.duckdb"
        return fake_registry

    with patch.object(test_eval, "load_config", return_value=cfg):
        with patch.object(test_eval, "load_splits", return_value=pd.DataFrame()):
            with patch.object(test_eval, "collect_candidates", return_value=[]):
                with patch.object(test_eval, "SharedRegistry") as registry_cls:
                    registry_cls.open_existing.side_effect = _open_existing
                    registry_cls.side_effect = AssertionError("constructor path must not be used")

                    result = test_eval.run_test_evaluation(
                        config="unused",
                        results_dir=str(results_dir),
                        write_store=False,
                    )

    registry_cls.open_existing.assert_called_once_with(registry_path)
    registry_cls.assert_not_called()
    assert default_results_dir.exists()
    assert result is None


def test_test_fit_model_uses_shared_registry_duckdb(tmp_path: Path):
    """The script helper should read candidate code from the DuckDB registry."""
    results_dir = tmp_path / "script_results"
    results_dir.mkdir()
    (results_dir / "shared_registry.duckdb").touch()

    registry_snapshot = {
        "iteration_history": [
            {
                "client_id": 0,
                "iteration": 0,
                "results": [
                    {
                        "function_name": "candidate_model",
                        "metric_name": "BIC",
                        "metric_value": 12.3,
                        "code": "def candidate_model():\n    return 1",
                    }
                ],
            }
        ]
    }

    with patch.object(test_fit_model_script, "SharedRegistry") as registry_cls:
        registry_cls.open_existing.return_value.read.return_value = registry_snapshot
        code = test_fit_model_script.load_code_from_registry(results_dir)

    registry_cls.open_existing.assert_called_once_with(results_dir / "shared_registry.duckdb")
    assert code == "def candidate_model():\n    return 1"


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
    (results_root / "task_b" / "legacy_registry.txt").touch()

    with patch.object(dashboard_config, "project_root", return_value=tmp_path):
        tasks = dashboard_config.available_tasks()

    assert tasks == ["task_a"]
