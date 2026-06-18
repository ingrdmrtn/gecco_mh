"""Dashboard overview tests for the command-center slice."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_ROOT = PROJECT_ROOT / "gecco-mh-dashboard"

if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))


from dashboard import components  # noqa: E402
from dashboard import data_adapter, views as dashboard_views  # noqa: E402


class _ColumnStub:
    def __init__(self, events: list[tuple[str, object]], index: int) -> None:
        self._events = events
        self.index = index

    def __enter__(self):
        self._events.append(("column.enter", self.index))
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _ExpanderStub:
    def __init__(self, events: list[tuple[str, object]], label: str, expanded: bool) -> None:
        self._events = events
        self.label = label
        self.expanded = expanded

    def __enter__(self):
        self._events.append(("expander", (self.label, self.expanded)))
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _StreamlitStub:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def subheader(self, text: str) -> None:
        self.events.append(("subheader", text))

    def caption(self, text: str) -> None:
        self.events.append(("caption", text))

    def info(self, text: str) -> None:
        self.events.append(("info", text))

    def warning(self, text: str) -> None:
        self.events.append(("warning", text))

    def write(self, *args: object) -> None:
        self.events.append(("write", args))

    def metric(self, label: str, value: object, **kwargs: object) -> None:
        self.events.append(("metric", {"label": label, "value": value, **kwargs}))

    def dataframe(self, frame: pd.DataFrame, **kwargs: object) -> None:
        self.events.append(("dataframe", list(frame.columns)))

    def line_chart(self, data: object, **kwargs: object) -> None:
        if isinstance(data, pd.DataFrame):
            payload = list(data.columns)
        else:
            payload = data
        self.events.append(("line_chart", payload))

    def json(self, body: object) -> None:
        self.events.append(("json", body))

    def code(self, body: str, language: str = "text") -> None:
        self.events.append(("code", (language, body)))

    def columns(self, count: int):
        return [_ColumnStub(self.events, index) for index in range(count)]

    def expander(self, label: str, expanded: bool = False):
        return _ExpanderStub(self.events, label, expanded)


def _install_streamlit_stub(monkeypatch: pytest.MonkeyPatch) -> _StreamlitStub:
    stub = _StreamlitStub()
    monkeypatch.setattr(dashboard_views, "st", stub)
    monkeypatch.setattr(components, "st", stub)
    return stub


def _overview_snapshot() -> dict[str, object]:
    return {
        "global_best": {
            "metric_value": 9.25,
            "model_code": "def model_b(): return 1",
            "param_names": ["beta"],
            "client_id": 1,
            "iteration": 1,
        },
        "baseline": {
            "function_name": "baseline_model",
            "metric_name": "BIC",
            "metric_value": 14.0,
            "param_names": ["baseline"],
            "eval_metrics": [],
            "mean_r2": 0.81,
            "max_r2": 0.9,
            "best_param": "beta",
            "per_param_r2": {"beta": 0.81},
            "code": "def baseline_model(): return 0",
            "val_mean_nll": 2.0,
        },
        "client_entries": {
            "0": {"status": "running", "iteration": 5, "timestamp": "2025-01-01T00:00:00Z"},
            "1": {"status": "complete", "iteration": 7, "timestamp": "2025-01-01T01:00:00Z"},
            "2": {"status": "complete_no_success", "iteration": 7},
            "3": {"status": "recovery_failed", "iteration": 4, "last_error": "boom"},
        },
        "iteration_history": [
            {
                "client_id": 0,
                "iteration": 0,
                "timestamp": "2025-01-01T00:00:00Z",
                "results": [
                    {"function_name": "model_a", "metric_value": 12.5, "param_names": ["alpha"]},
                    {"function_name": "broken", "metric_value": float("inf")},
                ],
            },
            {
                "client_id": 1,
                "iteration": 1,
                "timestamp": "2025-01-01T01:00:00Z",
                "results": [
                    {"function_name": "model_b", "metric_value": 9.25, "param_names": ["beta"]},
                    {"function_name": "missing", "metric_value": None},
                ],
            },
            {
                "client_id": 1,
                "iteration": 2,
                "results": [],
            },
        ],
        "tried_param_sets": [{"alpha": 1}, {"beta": 2}],
    }


def test_build_overview_summary_uses_registry_state_and_handles_missing_values() -> None:
    summary = data_adapter.build_overview_summary(_overview_snapshot())

    assert summary["client_count"] == 4
    assert summary["running_clients"] == 1
    assert summary["complete_clients"] == 2
    assert summary["recovery_failed_clients"] == 1
    assert summary["iteration_count"] == 3
    assert summary["trajectory_points"] == 2
    assert summary["param_set_count"] == 2
    assert summary["best_bic"] == pytest.approx(9.25)
    assert summary["baseline_bic"] == pytest.approx(14.0)
    assert summary["bic_delta"] == pytest.approx(4.75)
    assert summary["bic_delta_pct"] == pytest.approx(33.9285714286)
    assert summary["best_source"] == "global_best"
    assert summary["best_model_name"] == "Global best"
    assert summary["best_model_code"] == "def model_b(): return 1"
    assert summary["best_client_id"] == 1
    assert summary["best_iteration"] == 1
    assert summary["best_param_names"] == ["beta"]
    assert summary["best_provenance"]["model_code"] == "def model_b(): return 1"
    assert summary["run_state_label"] == "Running"
    assert summary["health_label"] == "Needs attention"

    iteration_frame = data_adapter.build_iteration_df(_overview_snapshot())
    assert list(iteration_frame["iteration"]) == [0, 1, 2]
    assert list(iteration_frame["best_bic"].dropna()) == [12.5, 9.25]


def test_build_overview_summary_prefers_global_best_over_history_when_they_differ() -> None:
    snapshot = _overview_snapshot()
    snapshot["global_best"] = {
        "metric_value": 20.5,
        "model_code": "def canonical_best(): return 42",
        "param_names": ["canonical"],
        "client_id": 99,
        "iteration": 77,
    }
    snapshot["iteration_history"] = [
        {
            "client_id": 0,
            "iteration": 0,
            "timestamp": "2025-01-01T00:00:00Z",
            "results": [
                {"function_name": "history_best", "metric_value": 8.0, "param_names": ["alpha"]},
            ],
        },
    ]

    summary = data_adapter.build_overview_summary(snapshot)

    assert summary["best_source"] == "global_best"
    assert summary["best_bic"] == pytest.approx(20.5)
    assert summary["best_model_name"] == "Global best"
    assert summary["best_model_code"] == "def canonical_best(): return 42"
    assert summary["best_client_id"] == 99
    assert summary["best_iteration"] == 77
    assert summary["best_param_names"] == ["canonical"]
    assert summary["best_provenance"]["metric_value"] == 20.5
    assert summary["iteration_frame"].iloc[0]["best_bic"] == pytest.approx(8.0)


def test_render_overview_shows_command_center_and_hides_debug_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _install_streamlit_stub(monkeypatch)

    dashboard_views.render_overview(_overview_snapshot(), history=[{"captured_at": "2025-01-01T01:30:00Z", "task_name": "phase-one", "registry_available": True, "stats": {"complete": 2}, "results_dir": "/tmp/results"}])

    labels = [value for kind, value in stub.events if kind in {"subheader", "caption", "info", "warning"}]
    metric_labels = [event[1]["label"] for event in stub.events if event[0] == "metric"]

    assert any("GeCCo run overview" in str(label) for label in labels)
    assert any("Run state" in str(label) for label in labels)
    assert any("Best model" in str(label) for label in labels)
    assert any("Baseline comparison" in str(label) for label in labels)
    assert any("Run health" in str(label) for label in labels)
    assert any("BIC trajectory" in str(label) for label in labels)
    assert any("Session history" in str(label) for label in labels)
    assert "Clients" in metric_labels
    assert "Best BIC" in metric_labels
    assert "Baseline BIC" in metric_labels
    assert "Health" in metric_labels
    assert any(kind == "line_chart" for kind, _ in stub.events)
    assert any(kind == "dataframe" for kind, _ in stub.events)
    assert not any(kind == "json" for kind, _ in stub.events)
    assert any(kind == "expander" and value[0] == "Overview debug details" and value[1] is False for kind, value in stub.events)


def test_render_overview_handles_missing_best_baseline_and_empty_history(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _install_streamlit_stub(monkeypatch)

    dashboard_views.render_overview({"client_entries": {}, "iteration_history": [], "tried_param_sets": []}, history=[])

    info_text = [value for kind, value in stub.events if kind == "info"]
    assert any("No running registry snapshot" in str(text) for text in info_text)
    assert any("No global best model yet" in str(text) for text in info_text)
    assert any("Baseline comparison not available" in str(text) for text in info_text)
    assert any("No BIC trajectory available" in str(text) for text in info_text)
    assert any("No session history captured yet" in str(text) for text in info_text)
    assert not any(kind == "json" for kind, _ in stub.events)
