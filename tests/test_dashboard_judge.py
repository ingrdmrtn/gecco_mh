"""Dashboard judge tests for the verdict-first workflow."""

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
import dashboard.views as dashboard_views  # noqa: E402


class _JudgeViewStub:
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
        self.events.append(("dataframe", frame.copy()))

    def json(self, body: object) -> None:
        self.events.append(("json", body))

    def code(self, body: str, language: str = "text") -> None:
        self.events.append(("code", (language, body)))

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


def _install_stub(monkeypatch):
    stub = _JudgeViewStub()
    monkeypatch.setattr(dashboard_views, "st", stub)
    monkeypatch.setattr(components, "st", stub)
    return stub


def _judge_rows() -> list[dict[str, object]]:
    return [
        {
            "iteration": 0,
            "synthesized_feedback": {"default": "Use a simpler mechanism."},
            "verdict": {
                "accepted": False,
                "per_angle": [
                    {"angle": "mechanism", "findings": "Too similar", "confidence": "low"},
                    {"angle": "trajectory", "summary": "Still improving", "confidence": "medium"},
                ],
                "key_recommendations": [
                    "Try a more mechanistically distinct update.",
                    "Reduce free parameters.",
                ],
            },
            "failed": True,
            "timestamp": "2025-01-01T00:00:00",
            "error": "Judge process crashed",
        },
        {
            "iteration": 1,
            "synthesized_feedback": None,
            "verdict": None,
            "failed": False,
            "timestamp": "2025-01-02T00:00:00",
            "error": None,
        },
    ]


def _judge_trace_artifacts() -> list[dict[str, object]]:
    return [
        {
            "kind": "judge_trace",
            "iteration": 0,
            "source": "judge",
            "path": "judge/iter0_run0.json",
            "trace": [{"tool": "call"}],
            "full_trace": [{"stage": "analysis"}],
            "synthesized_feedback": {"default": "Use a simpler mechanism."},
            "verdict": {
                "accepted": False,
                "per_angle": [
                    {"angle": "mechanism", "findings": "Too similar", "confidence": "low"},
                ],
                "key_recommendations": ["Try a more mechanistically distinct update."],
            },
            "payload": {"short_circuit": True, "trace": [{"tool": "call"}]},
        },
        {
            "kind": "judge_trace",
            "iteration": 10,
            "source": "judge",
            "path": "judge/iter10_run0.json",
            "trace": [{"tool": "call"}],
            "full_trace": [{"stage": "analysis"}],
            "synthesized_feedback": {"default": "Should not leak to iteration 1."},
            "verdict": {"accepted": True},
            "payload": {"trace": [{"tool": "call"}]},
        }
    ]


def test_render_judge_tab_prioritizes_registry_verdict_summary(monkeypatch) -> None:
    stub = _install_stub(monkeypatch)

    dashboard_views.render_judge_tab(_judge_rows(), trace_artifacts=_judge_trace_artifacts())

    section_labels = [value for kind, value in stub.events if kind == "subheader"]
    expander_labels = [value[0] for kind, value in stub.events if kind == "expander"]
    dataframes = [value for kind, value in stub.events if kind == "dataframe"]

    assert "Registry verdict summary" in section_labels
    assert "Judge iteration details" in section_labels
    assert section_labels.index("Registry verdict summary") < section_labels.index("Judge iteration details")
    assert "Judge artifacts" in expander_labels
    assert not any(label.startswith("Judge trace") for label in expander_labels)

    summary_frame = dataframes[0]
    assert list(summary_frame["iteration"]) == [0, 1]
    assert list(summary_frame.columns)[:6] == ["iteration", "status", "verdict", "recommendations", "confidence summary", "trace state"]
    assert "more mechanistically distinct update" in str(summary_frame.iloc[0]["recommendations"])
    assert "mechanism" in str(summary_frame.iloc[0]["confidence summary"])
    assert "trace" in str(summary_frame.iloc[0]["trace state"]).lower()
    assert "missing" in str(summary_frame.iloc[1]["trace state"]).lower() or "no trace" in str(summary_frame.iloc[1]["trace state"]).lower()


def test_render_judge_tab_handles_missing_trace_and_optional_fields(monkeypatch) -> None:
    stub = _install_stub(monkeypatch)

    dashboard_views.render_judge_tab(
        [
            {
                "iteration": 2,
                "synthesized_feedback": {"persona_a": "Stay focused.", "default": None},
                "verdict": {
                    "accepted": True,
                    "per_angle": "not-a-list",
                    "key_recommendations": ["Stay focused.", None, {"note": "bad shape"}],
                },
                "failed": False,
                "timestamp": "2025-01-03T00:00:00",
                "error": None,
            },
            {
                "iteration": 3,
                "synthesized_feedback": None,
                "verdict": "not-a-dict",
                "failed": True,
                "timestamp": "2025-01-04T00:00:00",
                "error": "judge crashed",
            },
            {
                "iteration": 4,
                "synthesized_feedback": {"default": "Keep going."},
                "verdict": {"accepted": False, "per_angle": [], "key_recommendations": ["Retry with more evidence."]},
                "failed": False,
                "timestamp": "2025-01-05T00:00:00",
                "error": None,
            },
        ],
        trace_artifacts=[
            {
                "kind": "judge_trace",
                "iteration": 4,
                "source": "judge",
                "path": "judge/iter4_run0.json",
                "trace": "not-a-list",
                "full_trace": {"stage": "analysis"},
                "synthesized_feedback": {"default": "Keep going."},
                "verdict": "not-a-dict",
                "payload": {"short_circuit": False},
            }
        ],
    )

    section_labels = [value for kind, value in stub.events if kind == "subheader"]
    info_messages = [value for kind, value in stub.events if kind == "info"]
    dataframes = [value for kind, value in stub.events if kind == "dataframe"]

    assert "Registry verdict summary" in section_labels
    assert any("Trace JSON" in str(message) or "trace" in str(message).lower() for message in info_messages)

    summary_frame = dataframes[0]
    assert summary_frame.iloc[0]["verdict"] == "Accepted"
    assert summary_frame.iloc[1]["status"] == "Failed"
    assert summary_frame.iloc[1]["verdict"] == "Failed"
    assert summary_frame.iloc[1]["recommendations"] == "—"
    assert "Stay focused." in str(summary_frame.iloc[0]["recommendations"])
    assert "Retry with more evidence" in str(summary_frame.iloc[2]["recommendations"])
    assert "0 tool calls" in str(summary_frame.iloc[2]["trace state"])
    assert "missing" in str(summary_frame.iloc[0]["trace state"]).lower() or "no trace" in str(summary_frame.iloc[0]["trace state"]).lower()
