"""Dashboard model comparison tests."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_ROOT = PROJECT_ROOT / "gecco-mh-dashboard"

if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))


from dashboard import components  # noqa: E402
import dashboard.data_adapter as dashboard_data_adapter  # noqa: E402
import dashboard.views as dashboard_views  # noqa: E402


class _ModelViewRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []
        self.selectbox_value: object | None = None

    def subheader(self, text: str) -> None:
        self.events.append(("subheader", text))

    def caption(self, text: str) -> None:
        self.events.append(("caption", text))

    def write(self, *args: object) -> None:
        self.events.append(("write", args))

    def dataframe(self, frame: pd.DataFrame, **kwargs: object) -> None:
        self.events.append(("dataframe", frame.copy()))

    def info(self, text: str) -> None:
        self.events.append(("info", text))

    def warning(self, text: str) -> None:
        self.events.append(("warning", text))

    def json(self, body: object) -> None:
        self.events.append(("json", body))

    def code(self, body: str, language: str = "text") -> None:
        self.events.append(("code", (language, body)))

    def metric(self, label: str, value: object, delta: object | None = None, help: object | None = None) -> None:
        self.events.append(("metric", (label, value, delta, help)))

    def selectbox(
        self,
        label: str,
        options: list[object],
        index: int = 0,
        format_func=None,
    ) -> object:
        rendered_options = [format_func(option) if format_func is not None else option for option in options]
        self.events.append(("selectbox", {"label": label, "options": list(options), "rendered": rendered_options, "index": index}))
        return self.selectbox_value if self.selectbox_value is not None else options[index]

    def expander(self, label: str, expanded: bool = False):
        self.events.append(("expander", (label, expanded)))

        class _Ctx:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()


def _diagnostics_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dashboard_model_key": "diagnostics.duckdb::1::train",
                "source_db": "diagnostics.duckdb",
                "db_path": "/tmp/diagnostics.duckdb",
                "model_id": 1,
                "split": "train",
                "name": "model_fast",
                "metric_name": "BIC",
                "metric_value": 9.25,
                "mean_r2": 0.92,
                "max_r2": 0.97,
                "status": "complete",
                "param_names": ["beta"],
                "detail": {
                    "code": "def model_fast(): return 1",
                    "parameter_recovery": {"passed": True, "mean_r": 0.8, "n_successful": 1, "per_param_r": {"beta": 1.0}, "simulation_error": None},
                    "validation_errors": [{"error_type": "shape", "error_message": "bad shape", "error_details": "x"}],
                    "individual_differences": {"mean_r2": 0.92, "max_r2": 0.97, "best_param": "beta", "per_param_r2": {"beta": 0.97}, "per_param_detail": {"beta": "ok"}},
                },
            },
            {
                "dashboard_model_key": "diagnostics.duckdb::2::train",
                "source_db": "diagnostics.duckdb",
                "db_path": "/tmp/diagnostics.duckdb",
                "model_id": 2,
                "split": "train",
                "name": "model_slow",
                "metric_name": "BIC",
                "metric_value": 12.5,
                "mean_r2": 0.88,
                "max_r2": 0.9,
                "status": "complete",
                "param_names": ["alpha"],
                "detail": {
                    "code": "def model_slow(): return 2",
                    "parameter_recovery": {"passed": False, "mean_r": 0.4, "n_successful": 0, "per_param_r": {"alpha": 0.2}, "simulation_error": "recovery error"},
                    "validation_errors": [{"error_type": "runtime", "error_message": "boom", "error_details": "trace"}],
                    "individual_differences": {"mean_r2": 0.88, "max_r2": 0.9, "best_param": "alpha", "per_param_r2": {"alpha": 0.9}, "per_param_detail": {"alpha": "ok"}},
                },
            },
            {
                "dashboard_model_key": "diagnostics.duckdb::3::train",
                "source_db": "diagnostics.duckdb",
                "db_path": "/tmp/diagnostics.duckdb",
                "model_id": 3,
                "split": "train",
                "name": "model_unknown",
                "metric_name": "BIC",
                "metric_value": 8.0,
                "mean_r2": None,
                "max_r2": None,
                "status": "unknown",
                "param_names": ["gamma"],
                "detail": {},
            },
            {
                "dashboard_model_key": "diagnostics.duckdb::4::train",
                "source_db": "diagnostics.duckdb",
                "db_path": "/tmp/diagnostics.duckdb",
                "model_id": 4,
                "split": "train",
                "name": "model_failed",
                "metric_name": "BIC",
                "metric_value": float("inf"),
                "mean_r2": None,
                "max_r2": None,
                "status": "error",
                "param_names": [],
                "detail": {
                    "validation_errors": [{"error_type": "fit", "error_message": "failed", "error_details": "traceback"}],
                },
            },
        ]
    )


def _registry_snapshot() -> dict[str, object]:
    return {
        "baseline": {"name": "baseline_model", "metric_value": 14.0},
        "iteration_history": [
            {
                "client_id": 7,
                "iteration": 4,
                "timestamp": "2025-01-01T00:00:00Z",
                "results": [
                    {
                        "function_name": "shared_model",
                        "metric_name": "BIC",
                        "metric_value": 13.0,
                        "param_names": ["alpha"],
                        "code": "def shared_model(): return 1",
                        "status": "complete",
                    },
                    {
                        "function_name": "shared_model",
                        "metric_name": "BIC",
                        "metric_value": 11.0,
                        "param_names": ["beta"],
                        "status": "unknown",
                    },
                ],
            }
        ],
    }


def test_build_model_comparison_frame_ranks_successes_before_failed_rows() -> None:
    frame = dashboard_data_adapter.build_model_comparison_frame(_diagnostics_summary_frame(), snapshot=None)

    assert frame is not None
    assert list(frame["dashboard_model_key"]) == [
        "diagnostics.duckdb::1::train",
        "diagnostics.duckdb::2::train",
        "diagnostics.duckdb::3::train",
        "diagnostics.duckdb::4::train",
    ]
    assert list(frame["display_rank"].head(2)) == [1, 2]
    assert pd.isna(frame.loc[frame["name"] == "model_unknown", "display_rank"]).all()
    assert pd.isna(frame.loc[frame["name"] == "model_failed", "display_rank"]).all()


def test_build_model_comparison_frame_uses_registry_fallback_identity() -> None:
    frame = dashboard_data_adapter.build_model_comparison_frame(None, snapshot=_registry_snapshot())

    assert frame is not None
    assert list(frame["dashboard_model_key"]) == ["registry::7::4::0", "registry::7::4::1"]
    assert list(frame["result_index"]) == [0, 1]
    assert list(frame["name"]) == ["shared_model", "shared_model"]


def test_render_models_tab_shows_selected_model_detail_and_collapsed_verbose_sections(monkeypatch) -> None:
    recorder = _ModelViewRecorder()
    recorder.selectbox_value = "diagnostics.duckdb::1::train"
    monkeypatch.setattr(dashboard_views, "st", recorder)
    monkeypatch.setattr(components, "st", recorder)

    frame = dashboard_data_adapter.build_model_comparison_frame(_diagnostics_summary_frame(), snapshot=_registry_snapshot())
    assert frame is not None

    dashboard_views.render_models_tab(frame, snapshot=_registry_snapshot(), top_n=2)

    captions = [value for kind, value in recorder.events if kind == "caption"]
    assert any("Baseline" in str(value) for value in captions)

    table_frames = [value for kind, value in recorder.events if kind == "dataframe"]
    assert table_frames
    table_frame = table_frames[0]
    assert list(table_frame["dashboard_model_key"]) == [
        "diagnostics.duckdb::1::train",
        "diagnostics.duckdb::2::train",
        "diagnostics.duckdb::3::train",
        "diagnostics.duckdb::4::train",
    ]
    assert list(table_frame["display_rank"].head(2)) == [1, 2]

    selectbox_events = [value for kind, value in recorder.events if kind == "selectbox"]
    assert selectbox_events
    assert selectbox_events[0]["options"] == [
        "diagnostics.duckdb::1::train",
        "diagnostics.duckdb::2::train",
        "diagnostics.duckdb::3::train",
        "diagnostics.duckdb::4::train",
    ]

    expander_labels = [value for kind, value in recorder.events if kind == "expander"]
    assert ("Error / recovery status", False) in expander_labels
    assert ("Model artifacts", False) in expander_labels
    assert ("Selected model details", False) in expander_labels
    assert ("Raw model detail payload", False) in expander_labels

    code_values = [value for kind, value in recorder.events if kind == "code"]
    assert any("def model_fast(): return 1" in body for _language, body in code_values)


def test_unknown_status_rows_are_appended_after_top_n_successes(monkeypatch) -> None:
    recorder = _ModelViewRecorder()
    recorder.selectbox_value = "diagnostics.duckdb::1::train"
    monkeypatch.setattr(dashboard_views, "st", recorder)
    monkeypatch.setattr(components, "st", recorder)

    snapshot = _registry_snapshot()
    frame = dashboard_data_adapter.build_model_comparison_frame(_diagnostics_summary_frame(), snapshot=snapshot)
    assert frame is not None

    dashboard_views.render_models_tab(frame, snapshot=snapshot, top_n=1)

    table_frames = [value for kind, value in recorder.events if kind == "dataframe"]
    assert table_frames
    table_frame = table_frames[0]
    assert list(table_frame["name"]) == ["model_fast", "model_unknown", "model_failed"]
    assert table_frame.iloc[0]["display_rank"] == 1
    assert pd.isna(table_frame.iloc[1]["display_rank"])
    assert pd.isna(table_frame.iloc[2]["display_rank"])

    selectbox_events = [value for kind, value in recorder.events if kind == "selectbox"]
    assert selectbox_events[0]["options"] == ["diagnostics.duckdb::1::train", "diagnostics.duckdb::3::train", "diagnostics.duckdb::4::train"]


def test_render_model_detail_shows_plain_error_and_recovery_failure(monkeypatch) -> None:
    recorder = _ModelViewRecorder()
    recorder.selectbox_value = "diagnostics.duckdb::5::train"
    monkeypatch.setattr(dashboard_views, "st", recorder)
    monkeypatch.setattr(components, "st", recorder)

    frame = pd.DataFrame(
        [
            {
                "dashboard_model_key": "diagnostics.duckdb::5::train",
                "source_db": "diagnostics.duckdb",
                "db_path": "/tmp/diagnostics.duckdb",
                "model_id": 5,
                "split": "train",
                "name": "model_problem",
                "metric_name": "BIC",
                "metric_value": 15.0,
                "status": "complete",
                "param_names": ["alpha"],
                "detail": {
                    "error": "fit failed",
                    "code": "def model_problem(): return 0",
                    "parameter_recovery": {"passed": False, "mean_r": 0.2, "n_successful": 0, "per_param_r": {}, "simulation_error": "recovery error"},
                },
            }
        ]
    )

    dashboard_views.render_models_tab(frame, top_n=1)

    warnings = [value for kind, value in recorder.events if kind == "warning"]
    assert any("fit failed" in str(value) for value in warnings)
    assert any("Parameter recovery failed" in str(value) for value in warnings)
    assert any("Recovery simulation error" in str(value) for value in warnings)


def test_render_model_detail_handles_missing_r2_cleanly(monkeypatch) -> None:
    recorder = _ModelViewRecorder()
    recorder.selectbox_value = "diagnostics.duckdb::6::train"
    monkeypatch.setattr(dashboard_views, "st", recorder)
    monkeypatch.setattr(components, "st", recorder)

    frame = pd.DataFrame(
        [
            {
                "dashboard_model_key": "diagnostics.duckdb::6::train",
                "source_db": "diagnostics.duckdb",
                "db_path": "/tmp/diagnostics.duckdb",
                "model_id": 6,
                "split": "train",
                "name": "model_no_r2",
                "metric_name": "BIC",
                "metric_value": 9.5,
                "status": "complete",
                "param_names": ["beta"],
                "detail": {"code": "def model_no_r2(): return 1"},
            }
        ]
    )

    dashboard_views.render_models_tab(frame, top_n=1)

    expander_labels = [value for kind, value in recorder.events if kind == "expander"]
    assert ("R² details", False) not in expander_labels


def test_render_model_detail_renders_recovery_failure_without_raw_debug_dependency(monkeypatch) -> None:
    recorder = _ModelViewRecorder()
    recorder.selectbox_value = "diagnostics.duckdb::2::train"
    monkeypatch.setattr(dashboard_views, "st", recorder)
    monkeypatch.setattr(components, "st", recorder)

    frame = _diagnostics_summary_frame().iloc[[1]].copy()
    dashboard_views.render_models_tab(frame, top_n=1)

    warnings = [value for kind, value in recorder.events if kind == "warning"]
    assert any("Parameter recovery failed" in str(value) for value in warnings)
    assert any("Recovery simulation error" in str(value) for value in warnings)
