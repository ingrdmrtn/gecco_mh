"""Dashboard foundation tests for shared theme and components."""

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


from dashboard import components, theme  # noqa: E402
import app as dashboard_app  # noqa: E402
import dashboard.views as dashboard_views  # noqa: E402


def test_status_metadata_and_formatting_are_stable() -> None:
    known_statuses = {
        "running": ("Running", "info", False),
        "complete": ("Complete", "success", True),
        "complete_no_success": ("Complete (no success)", "success", True),
        "success": ("Success", "success", True),
        "failed": ("Failed", "error", True),
        "validation_error": ("Validation error", "error", True),
        "fit_error": ("Fit error", "error", True),
        "recovery_failed": ("Recovery failed", "error", True),
        "error": ("Error", "error", True),
    }

    for raw_status, (label, tone, terminal) in known_statuses.items():
        metadata = components.status_metadata(raw_status)
        assert metadata["status"] == raw_status
        assert metadata["label"] == label
        assert metadata["tone"] == tone
        assert metadata["terminal"] is terminal

    neutral = components.status_metadata("new_status")
    assert neutral["status"] == "unknown"
    assert neutral["label"] == "Unknown"
    assert neutral["tone"] == "neutral"
    assert neutral["terminal"] is False

    assert components.format_value(None) == "—"
    assert components.format_value(12345) == "12,345"
    assert components.format_value(12.3400) == "12.34"
    assert components.format_value(" ready ") == "ready"


def test_theme_and_components_are_importable() -> None:
    assert callable(theme.apply_dashboard_theme)
    assert callable(components.sidebar_header)
    assert callable(components.section_header)
    assert callable(components.debug_details)


class _SidebarStub:
    def __init__(self, events: list[tuple[str, object]]) -> None:
        self._events = events
        self.selectbox_value = "phase-one"
        self.text_input_value = "/tmp/results"
        self.checkbox_value = False
        self.button_value = False

    def header(self, label: str) -> None:
        self._events.append(("sidebar.header", label))

    def selectbox(self, label: str, options: list[str], index: int = 0) -> str:
        self._events.append(("sidebar.selectbox", label))
        return self.selectbox_value or options[index]

    def text_input(self, label: str, value: str) -> str:
        self._events.append(("sidebar.text_input", label))
        return self.text_input_value or value

    def checkbox(self, label: str, value: bool = False) -> bool:
        self._events.append(("sidebar.checkbox", label))
        return self.checkbox_value

    def number_input(
        self,
        label: str,
        *,
        min_value: float | int,
        max_value: float | int,
        value: float | int,
        step: float | int,
    ) -> float | int:
        self._events.append(("sidebar.number_input", label))
        return value

    def button(self, label: str) -> bool:
        self._events.append(("sidebar.button", label))
        return self.button_value


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
        self.sidebar = _SidebarStub(self.events)

    def set_page_config(self, **kwargs: object) -> None:
        self.events.append(("set_page_config", kwargs))

    def title(self, text: str) -> None:
        self.events.append(("title", text))

    def caption(self, text: str) -> None:
        self.events.append(("caption", text))

    def tabs(self, labels: list[str]):
        self.events.append(("tabs", tuple(labels)))
        return [nullcontext() for _ in labels]

    def selectbox(self, label: str, options: list[str], index: int = 0, format_func=None) -> str:
        self.events.append(("selectbox", (label, tuple(options), index)))
        return options[index]

    def warning(self, text: str) -> None:
        self.events.append(("warning", text))

    def rerun(self) -> None:
        self.events.append(("rerun", None))

    def subheader(self, text: str) -> None:
        self.events.append(("subheader", text))

    def write(self, *args: object) -> None:
        self.events.append(("write", args))

    def metric(self, label: str, value: object) -> None:
        self.events.append(("metric", (label, value)))

    def dataframe(self, frame: pd.DataFrame, **kwargs: object) -> None:
        self.events.append(("dataframe", tuple(frame.columns)))

    def info(self, text: str) -> None:
        self.events.append(("info", text))

    def json(self, body: object) -> None:
        self.events.append(("json", body))

    def code(self, body: str, language: str = "text") -> None:
        self.events.append(("code", (language, body)))

    def expander(self, label: str, expanded: bool = False):
        return _ExpanderStub(self.events, label, expanded)


def _install_streamlit_stub(monkeypatch: pytest.MonkeyPatch) -> _StreamlitStub:
    stub = _StreamlitStub()
    monkeypatch.setattr(dashboard_app, "st", stub)
    monkeypatch.setattr(dashboard_views, "st", stub)
    monkeypatch.setattr(components, "st", stub)
    return stub


def _install_renderer_stubs(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    def _record(name: str):
        def _inner(*args, **kwargs):
            events.append(name)

        return _inner

    monkeypatch.setattr(dashboard_app, "render_overview_tab", _record("overview"))
    monkeypatch.setattr(dashboard_app, "render_models_tab", _record("models"))
    monkeypatch.setattr(dashboard_app, "render_clients_tab", _record("clients"))
    monkeypatch.setattr(dashboard_app, "render_results_tab", _record("results"))
    monkeypatch.setattr(dashboard_app, "render_judge_tab", _record("judge"))
    monkeypatch.setattr(dashboard_app, "render_waiting_state", _record("waiting"))


def test_manual_refresh_reruns_after_state_capture_without_sleep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_dir = tmp_path / "selected-results"
    results_dir.mkdir()
    (results_dir / "shared_registry.duckdb").touch()

    stub = _install_streamlit_stub(monkeypatch)
    stub.sidebar.checkbox_value = False
    stub.sidebar.button_value = True
    stub.sidebar.text_input_value = str(results_dir)

    events: list[str] = []
    monkeypatch.setattr(dashboard_app, "available_tasks", lambda: ["phase-one"])
    monkeypatch.setattr(dashboard_app, "apply_dashboard_theme", lambda: events.append("theme"))
    monkeypatch.setattr(
        dashboard_app,
        "load_registry_snapshot",
        lambda path: events.append("load_registry_snapshot") or {"client_entries": {}},
    )
    monkeypatch.setattr(dashboard_app, "load_diagnostics_summary", lambda results_dir: None)
    monkeypatch.setattr(dashboard_app, "load_diagnostics_model_rows", lambda results_dir, iteration=0: [])
    monkeypatch.setattr(dashboard_app, "normalize_judge_iterations", lambda snapshot: [])
    monkeypatch.setattr(dashboard_app, "summary_stats", lambda snapshot: {"complete": 0, "running": 0, "errors": 0, "recovery_failed": 0})
    monkeypatch.setattr(dashboard_app, "append_snapshot", lambda snapshot, max_points=None: events.append("append_snapshot") or [snapshot])
    monkeypatch.setattr(dashboard_app, "get_history", lambda: [{"captured_at": "now"}])
    monkeypatch.setattr(dashboard_app.time, "sleep", lambda seconds: events.append(f"sleep:{seconds}"))
    monkeypatch.setattr(stub, "rerun", lambda: events.append("rerun"))
    monkeypatch.setattr(dashboard_app, "render_waiting_state", lambda *args, **kwargs: events.append("waiting"))
    _install_renderer_stubs(monkeypatch, events)

    dashboard_app.main()

    assert events[:3] == ["theme", "load_registry_snapshot", "append_snapshot"]
    assert ("tabs", ("Overview", "Models", "Clients", "Artifacts", "Judge")) in stub.events
    assert "sleep:5.0" not in events
    assert events[-1] == "rerun"
    assert events.index("append_snapshot") < events.index("rerun")


def test_auto_refresh_ordering_preserves_append_sleep_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_dir = tmp_path / "selected-results"
    results_dir.mkdir()
    (results_dir / "shared_registry.duckdb").touch()

    stub = _install_streamlit_stub(monkeypatch)
    stub.sidebar.checkbox_value = True
    stub.sidebar.button_value = False
    stub.sidebar.text_input_value = str(results_dir)

    events: list[str] = []
    monkeypatch.setattr(dashboard_app, "available_tasks", lambda: ["phase-one"])
    monkeypatch.setattr(dashboard_app, "apply_dashboard_theme", lambda: events.append("theme"))
    monkeypatch.setattr(dashboard_app, "load_registry_snapshot", lambda path: events.append("load_registry_snapshot") or {"client_entries": {}})
    monkeypatch.setattr(dashboard_app, "load_diagnostics_summary", lambda results_dir: None)
    monkeypatch.setattr(dashboard_app, "load_diagnostics_model_rows", lambda results_dir, iteration=0: [])
    monkeypatch.setattr(dashboard_app, "normalize_judge_iterations", lambda snapshot: [])
    monkeypatch.setattr(dashboard_app, "summary_stats", lambda snapshot: {"complete": 0, "running": 0, "errors": 0, "recovery_failed": 0})
    monkeypatch.setattr(dashboard_app, "append_snapshot", lambda snapshot, max_points=None: events.append("append_snapshot") or [snapshot])
    monkeypatch.setattr(dashboard_app, "get_history", lambda: [{"captured_at": "now"}])
    monkeypatch.setattr(dashboard_app.time, "sleep", lambda seconds: events.append(f"sleep:{seconds}"))
    monkeypatch.setattr(stub, "rerun", lambda: events.append("rerun"))
    monkeypatch.setattr(dashboard_app, "render_waiting_state", lambda *args, **kwargs: events.append("waiting"))
    _install_renderer_stubs(monkeypatch, events)

    dashboard_app.main()

    assert events[:3] == ["theme", "load_registry_snapshot", "append_snapshot"]
    assert events[-2:] == ["sleep:5.0", "rerun"]
    assert events.index("append_snapshot") < events.index("sleep:5.0") < events.index("rerun")


def test_no_auto_refresh_when_registry_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_dir = tmp_path / "selected-results"
    results_dir.mkdir()

    stub = _install_streamlit_stub(monkeypatch)
    stub.sidebar.checkbox_value = True
    stub.sidebar.button_value = False
    stub.sidebar.text_input_value = str(results_dir)

    events: list[str] = []
    monkeypatch.setattr(dashboard_app, "available_tasks", lambda: ["phase-one"])
    monkeypatch.setattr(dashboard_app, "apply_dashboard_theme", lambda: events.append("theme"))
    monkeypatch.setattr(dashboard_app, "render_waiting_state", lambda *args, **kwargs: events.append("waiting"))
    monkeypatch.setattr(dashboard_app, "append_snapshot", lambda snapshot, max_points=None: events.append("append_snapshot") or [snapshot])
    monkeypatch.setattr(dashboard_app, "get_history", lambda: [])
    monkeypatch.setattr(dashboard_app, "load_registry_snapshot", lambda path: pytest.fail("load_registry_snapshot should not run when registry is missing"))
    monkeypatch.setattr(dashboard_app, "load_diagnostics_summary", lambda results_dir: None)
    monkeypatch.setattr(dashboard_app, "load_diagnostics_model_rows", lambda results_dir, iteration=0: [])
    monkeypatch.setattr(dashboard_app, "normalize_judge_iterations", lambda snapshot: [])
    monkeypatch.setattr(dashboard_app, "summary_stats", lambda snapshot: {"complete": 0, "running": 0, "errors": 0, "recovery_failed": 0})
    monkeypatch.setattr(dashboard_app.time, "sleep", lambda seconds: events.append(f"sleep:{seconds}"))
    monkeypatch.setattr(dashboard_app, "render_overview_tab", lambda *args, **kwargs: events.append("overview"))
    monkeypatch.setattr(dashboard_app, "render_models_tab", lambda *args, **kwargs: events.append("models"))
    monkeypatch.setattr(dashboard_app, "render_clients_tab", lambda *args, **kwargs: events.append("clients"))
    monkeypatch.setattr(dashboard_app, "render_results_tab", lambda *args, **kwargs: events.append("results"))
    monkeypatch.setattr(dashboard_app, "render_judge_tab", lambda *args, **kwargs: events.append("judge"))

    dashboard_app.main()

    assert "waiting" in events
    assert not any(event.startswith("sleep:") for event in events)
    assert "rerun" not in events


def test_app_loads_and_passes_artifact_and_trace_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_dir = tmp_path / "selected-results"
    results_dir.mkdir()
    (results_dir / "shared_registry.duckdb").touch()

    stub = _install_streamlit_stub(monkeypatch)
    stub.sidebar.checkbox_value = False
    stub.sidebar.button_value = False
    stub.sidebar.text_input_value = str(results_dir)

    events: list[str] = []
    feedback_rows = _dashboard_feedback_artifacts()
    trace_rows = _dashboard_judge_trace_artifacts()

    monkeypatch.setattr(dashboard_app, "available_tasks", lambda: ["phase-one"])
    monkeypatch.setattr(dashboard_app, "apply_dashboard_theme", lambda: events.append("theme"))
    monkeypatch.setattr(dashboard_app, "load_registry_snapshot", lambda path: events.append(f"registry:{Path(path)}") or {"client_entries": {}})
    monkeypatch.setattr(dashboard_app, "load_diagnostics_summary", lambda results_dir: None)
    monkeypatch.setattr(dashboard_app, "load_diagnostics_model_rows", lambda results_dir, iteration=0: [])
    monkeypatch.setattr(dashboard_app, "load_feedback_artifacts", lambda path: events.append(f"feedback:{Path(path)}") or feedback_rows)
    monkeypatch.setattr(dashboard_app, "load_judge_trace_artifacts", lambda path: events.append(f"trace:{Path(path)}") or trace_rows)
    monkeypatch.setattr(dashboard_app, "normalize_judge_iterations", lambda snapshot: [])
    monkeypatch.setattr(dashboard_app, "summary_stats", lambda snapshot: {"complete": 0, "running": 0, "errors": 0, "recovery_failed": 0})
    monkeypatch.setattr(dashboard_app, "append_snapshot", lambda snapshot, max_points=None: events.append("append_snapshot") or [snapshot])
    monkeypatch.setattr(dashboard_app, "get_history", lambda: [{"captured_at": "now"}])
    monkeypatch.setattr(dashboard_app.time, "sleep", lambda seconds: events.append(f"sleep:{seconds}"))
    monkeypatch.setattr(stub, "rerun", lambda: events.append("rerun"))

    render_calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        dashboard_app,
        "render_overview_tab",
        lambda *args, **kwargs: render_calls.append(("overview", args, kwargs)),
    )
    monkeypatch.setattr(
        dashboard_app,
        "render_models_tab",
        lambda *args, **kwargs: render_calls.append(("models", args, kwargs)),
    )
    monkeypatch.setattr(
        dashboard_app,
        "render_clients_tab",
        lambda *args, **kwargs: render_calls.append(("clients", args, kwargs)),
    )
    monkeypatch.setattr(
        dashboard_app,
        "render_results_tab",
        lambda *args, **kwargs: render_calls.append(("results", args, kwargs)),
    )
    monkeypatch.setattr(
        dashboard_app,
        "render_judge_tab",
        lambda *args, **kwargs: render_calls.append(("judge", args, kwargs)),
    )
    monkeypatch.setattr(dashboard_app, "render_waiting_state", lambda *args, **kwargs: events.append("waiting"))

    dashboard_app.main()

    assert f"feedback:{results_dir}" in events
    assert f"trace:{results_dir}" in events
    results_call = next(call for call in render_calls if call[0] == "results")
    assert results_call[2]["results_dir"] == results_dir
    assert results_call[2]["feedback_artifacts"] == feedback_rows
    judge_call = next(call for call in render_calls if call[0] == "judge")
    assert judge_call[2]["results_dir"] == results_dir
    assert judge_call[2]["trace_artifacts"] == trace_rows


def _dashboard_summary_row() -> dict[str, object]:
    return {
        "dashboard_model_key": "diagnostics.duckdb::1::train",
        "name": "model_a",
        "split": "train",
        "metric_name": "BIC",
        "metric_value": 10.5,
        "mean_r2": 0.91,
        "max_r2": 0.97,
        "status": "complete",
        "detail": {
            "code": "def model_a(): return 1",
            "validation_errors": [{"error_type": "shape", "error_message": "bad shape", "error_details": "x"}],
            "parameter_recovery": {"passed": True, "mean_r": 0.8, "n_successful": 1, "per_param_r": {"alpha": 1.0}, "simulation_error": None},
            "individual_differences": {"mean_r2": 0.91, "max_r2": 0.97, "best_param": "alpha", "per_param_r2": {"alpha": 0.97}, "per_param_detail": {"alpha": "ok"}},
            "ppc": [{"participant_id": 1, "statistic_name": "loss", "condition": "raw", "observed": 1.0, "simulated_mean": 0.9, "simulated_q025": 0.8, "simulated_q975": 1.1, "n_sims": 100}],
            "block_residuals": [{"participant_id": 1, "block_idx": 0, "block_start": 0, "block_end": 10, "mean_nll_per_trial": 0.2, "n_trials": 10}],
        },
    }


def _dashboard_feedback_artifacts() -> list[dict[str, object]]:
    return [
        {
            "kind": "feedback",
            "source": "feedback",
            "path": "feedback/iter0_run0.txt",
            "content": "Raw feedback text for the current run.",
        },
        {
            "kind": "model_code",
            "source": "models",
            "path": "models/iter0_run0.txt",
            "content": "def model_a(): return 1",
        },
        {
            "kind": "review",
            "source": "reviews",
            "path": "reviews/iter0.json",
            "payload": {"synthesized_feedback": "Use a simpler mechanism.", "summary": "ok"},
        },
    ]


def _dashboard_judge_trace_artifacts() -> list[dict[str, object]]:
    return [
        {
            "kind": "judge_trace",
            "source": "judge",
            "path": "judge/iter0_run0.json",
            "trace": [{"tool": "call"}],
            "full_trace": [{"stage": "analysis"}],
            "synthesized_feedback": {"default": "Improve recovery."},
            "verdict": {"accepted": True},
            "payload": {"trace": [{"tool": "call"}]},
        }
    ]


class _ViewRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def subheader(self, text: str) -> None:
        self.events.append(("subheader", text))

    def caption(self, text: str) -> None:
        self.events.append(("caption", text))

    def write(self, *args: object) -> None:
        self.events.append(("write", args))

    def metric(self, label: str, value: object) -> None:
        self.events.append(("metric", (label, value)))

    def dataframe(self, frame: pd.DataFrame, **kwargs: object) -> None:
        self.events.append(("dataframe", tuple(frame.columns)))

    def selectbox(self, label: str, options: list[str], index: int = 0, format_func=None) -> str:
        self.events.append(("selectbox", (label, tuple(options), index)))
        return options[index]

    def info(self, text: str) -> None:
        self.events.append(("info", text))

    def warning(self, text: str) -> None:
        self.events.append(("warning", text))

    def json(self, body: object) -> None:
        self.events.append(("json", body))

    def code(self, body: str, language: str = "text") -> None:
        self.events.append(("code", (language, body)))

    def expander(self, label: str, expanded: bool = False):
        recorder = self

        class _Ctx:
            def __enter__(self_inner):
                recorder.events.append(("expander", (label, expanded)))
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()


def test_phase_zero_workflows_are_reachable_from_renderers(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _ViewRecorder()
    monkeypatch.setattr(dashboard_views, "st", recorder)
    monkeypatch.setattr(components, "st", recorder)

    snapshot = {
        "global_best": {"name": "model_a", "metric_value": 10.5},
        "baseline": {"name": "baseline", "metric_value": 14.0},
        "client_entries": {
            "0": {"status": "running", "iteration": 1, "timestamp": "2025-01-01T00:00:00"},
            "1": {"status": "complete", "iteration": 2, "timestamp": "2025-01-01T01:00:00"},
        },
    }
    history = [
        {"captured_at": "2025-01-01T00:00:00", "task_name": "phase-one", "registry_available": True, "stats": {"complete": 1}, "results_dir": "/tmp/results"},
    ]
    summary = pd.DataFrame([_dashboard_summary_row()])
    feedback_artifacts = _dashboard_feedback_artifacts()
    judge_rows = [
        {
            "iteration": 0,
            "synthesized_feedback": {"default": "Improve recovery."},
            "verdict": {"accepted": True},
            "failed": False,
            "timestamp": "2025-01-01T00:00:00",
            "error": None,
        }
    ]
    trace_artifacts = _dashboard_judge_trace_artifacts()

    dashboard_views.render_overview_tab(snapshot, summary, history)
    dashboard_views.render_clients_tab(snapshot)
    dashboard_views.render_models_tab(summary, top_n=1)
    dashboard_views.render_results_tab(
        summary,
        results_dir=Path("/tmp/results"),
        feedback_artifacts=feedback_artifacts,
        trace_artifacts=trace_artifacts,
    )
    dashboard_views.render_judge_tab(judge_rows, results_dir=Path("/tmp/results"), trace_artifacts=trace_artifacts)

    labels = [value for kind, value in recorder.events if kind in {"subheader", "caption", "info", "warning"}]
    text_values = [value for kind, value in recorder.events if kind == "write"]
    code_values = [value for kind, value in recorder.events if kind == "code"]

    assert any("Trajectory / session history" in str(label) for label in labels)
    assert any("Clients" == str(label) for label in labels)
    assert any("Selected model detail" in str(label) for label in labels)
    assert any("Artifacts" == str(label) for label in labels)
    assert any("Verdict / feedback drilldown" in str(label) for label in labels)
    assert any("Raw feedback text for the current run." in str(entry) for entry in text_values)
    assert any("def model_a(): return 1" in str(entry) for entry in code_values)
    assert any(
        isinstance(payload, dict) and payload.get("synthesized_feedback") == "Use a simpler mechanism."
        for kind, payload in recorder.events
        if kind == "json"
    )
    assert any(
        isinstance(payload, dict) and payload.get("trace") == [{"tool": "call"}]
        for kind, payload in recorder.events
        if kind == "json"
    )
    assert any(kind == "dataframe" for kind, _ in recorder.events)


def test_raw_debug_sections_are_collapsed(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _ViewRecorder()
    monkeypatch.setattr(dashboard_views, "st", recorder)
    monkeypatch.setattr(components, "st", recorder)

    summary = pd.DataFrame([_dashboard_summary_row()])
    feedback_artifacts = _dashboard_feedback_artifacts()
    judge_rows = [
        {
            "iteration": 0,
            "synthesized_feedback": {"default": "Improve recovery."},
            "verdict": {"accepted": True},
            "failed": False,
            "timestamp": "2025-01-01T00:00:00",
            "error": "trace warning",
        }
    ]
    trace_artifacts = _dashboard_judge_trace_artifacts()

    dashboard_views.render_models_tab(summary, top_n=1)
    dashboard_views.render_results_tab(
        summary,
        feedback_artifacts=feedback_artifacts,
        snapshot={"client_entries": {}, "iteration_history": []},
        trace_artifacts=trace_artifacts,
    )
    dashboard_views.render_judge_tab(judge_rows, trace_artifacts=trace_artifacts)

    expander_labels = [value for kind, value in recorder.events if kind == "expander"]
    section_labels = [value for kind, value in recorder.events if kind == "subheader"]
    assert any("diagnostics.duckdb::1::train" in label for label, _ in expander_labels)
    assert any(label == "Model artifacts" and expanded is False for label, expanded in expander_labels)
    assert any(label == "Selected model details" and expanded is False for label, expanded in expander_labels)
    assert any(label == "Judge artifacts" and expanded is False for label, expanded in expander_labels)
    assert any(label == "Registry JSON" and expanded is False for label, expanded in expander_labels)
    assert any(label == "Raw LLM output / feedback" for label in section_labels)
    assert any(label == "Full judge traces" for label in section_labels)
    assert any(label == "Model summary payload" and expanded is False for label, expanded in expander_labels)
    assert any(label == "Raw result row payload" and expanded is False for label, expanded in expander_labels)
