"""Dashboard client status tests for the live-status slice."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_ROOT = PROJECT_ROOT / "gecco-mh-dashboard"

if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))


from dashboard import components  # noqa: E402
from dashboard.data_adapter import build_client_df  # noqa: E402
import dashboard.views as dashboard_views  # noqa: E402


class _ClientViewStub:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def subheader(self, text: str) -> None:
        self.events.append(("subheader", text))

    def caption(self, text: str) -> None:
        self.events.append(("caption", text))

    def metric(self, label: str, value: object, **kwargs: object) -> None:
        self.events.append(("metric", (label, value, kwargs)))

    def dataframe(self, frame: pd.DataFrame, **kwargs: object) -> None:
        self.events.append(("dataframe", frame.copy()))

    def info(self, text: str) -> None:
        self.events.append(("info", text))

    def columns(self, count: int):
        self.events.append(("columns", count))
        return [nullcontext() for _ in range(count)]

    def expander(self, label: str, expanded: bool = False):
        self.events.append(("expander", (label, expanded)))

        class _Ctx:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()


def _install_view_stub(monkeypatch: pytest.MonkeyPatch) -> _ClientViewStub:
    stub = _ClientViewStub()
    monkeypatch.setattr(dashboard_views, "st", stub)
    monkeypatch.setattr(components, "st", stub)
    return stub


def _snapshot() -> dict[str, object]:
    return {
        "client_entries": {
            "2": {
                "status": "complete_no_success",
                "last_iteration": 6,
                "best_metric": 9.5,
                "activity": "finished without a runnable model",
                "updated_at": "2025-01-01T12:00:00",
            },
            "0": {
                "status": "running",
                "last_iteration": 3,
                "best_metric": 12.0,
                "activity": "searching",
                "updated_at": "2025-01-01T12:55:00",
            },
            "1": {
                "status": "complete",
                "iteration": 4,
                "best_metric": 8.25,
                "message": "finished",
                "timestamp": "2025-01-01T12:30:00",
            },
            "3": {
                "activity": "waiting for registry",
            },
            "4": {
                "status": "failed",
                "last_iteration": 7,
                "best_metric": None,
                "activity": "failed while evaluating",
                "updated_at": "2025-01-01T12:45:00",
            },
            "5": {
                "status": "validation_error",
                "last_iteration": 8,
                "best_metric": None,
                "activity": "validation mismatch",
                "updated_at": "2025-01-01T12:50:00",
            },
        }
    }


def test_build_client_df_preserves_live_status_fields_and_update_age() -> None:
    frame = build_client_df(_snapshot(), now=datetime(2025, 1, 1, 13, 0, 0))

    assert list(frame["client"]) == ["0", "1", "2", "3", "4", "5"]
    assert list(frame["status"]) == ["running", "complete", "complete_no_success", "unknown", "failed", "validation_error"]
    assert list(frame["status label"]) == [
        "Running",
        "Complete",
        "Complete (no success)",
        "Unknown",
        "Failed",
        "Validation error",
    ]
    assert list(frame["status tone"]) == ["info", "success", "success", "neutral", "error", "error"]
    assert list(frame["terminal"]) == [False, True, True, False, True, True]
    assert list(frame["last iteration"]) == [3, 4, 6, None, 7, 8]
    assert list(frame["best BIC"]) == [12.0, 8.25, 9.5, None, None, None]
    assert list(frame["updated age"]) == ["5m ago", "30m ago", "1h ago", "—", "15m ago", "10m ago"]
    assert frame.loc[3, "updated"] is None


def test_render_clients_tab_shows_summary_table_and_empty_state(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _install_view_stub(monkeypatch)
    fixed_now = datetime(2025, 1, 1, 13, 0, 0)
    monkeypatch.setattr(dashboard_views, "build_client_df", lambda snapshot: build_client_df(snapshot, now=fixed_now))

    dashboard_views.render_clients_tab(_snapshot())

    metric_labels = [payload[0] for kind, payload in stub.events if kind == "metric"]
    assert set(metric_labels) == {"Clients", "Running", "Complete", "Issues"}

    table_frame = next(payload for kind, payload in stub.events if kind == "dataframe")
    assert list(table_frame.columns) == ["client", "status", "activity", "last iteration", "best BIC", "updated"]
    assert list(table_frame["status"]) == [
        "Running",
        "Complete",
        "Complete (no success)",
        "Unknown",
        "Failed",
        "Validation error",
    ]
    assert list(table_frame["updated"]) == ["5m ago", "30m ago", "1h ago", "—", "15m ago", "10m ago"]
    assert any(kind == "subheader" and payload == "Clients" for kind, payload in stub.events)

    stub.events.clear()
    dashboard_views.render_clients_tab({"client_entries": {}})

    assert any(kind == "info" and payload == "No client rows available." for kind, payload in stub.events)
