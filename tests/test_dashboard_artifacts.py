"""Dashboard artifacts tests for the centralized raw/debug workflow."""

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
from dashboard.data_adapter import load_feedback_artifacts, load_judge_trace_artifacts  # noqa: E402
import dashboard.views as dashboard_views  # noqa: E402


class _ArtifactsViewStub:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []
        self.selectbox_value: object | None = None

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

    def selectbox(self, label: str, options: list[object], index: int = 0, format_func=None) -> object:
        self.events.append(("selectbox", {"label": label, "options": list(options), "index": index}))
        return self.selectbox_value if self.selectbox_value is not None else options[index]

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
    stub = _ArtifactsViewStub()
    monkeypatch.setattr(dashboard_views, "st", stub)
    monkeypatch.setattr(components, "st", stub)
    return stub


def _summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
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
        ]
    )


def _feedback_artifacts() -> list[dict[str, object]]:
    return [
        {"kind": "feedback", "source": "feedback", "path": "feedback/iter0_run0.txt", "content": "Raw feedback text."},
        {"kind": "model_code", "source": "models", "path": "models/iter0_run0.txt", "content": "def model_a(): return 1"},
        {"kind": "review", "source": "reviews", "path": "reviews/iter0.json", "payload": {"synthesized_feedback": "Use a simpler mechanism.", "summary": "ok"}},
        {"kind": "review", "source": "reviews", "path": "reviews/iter1.json", "payload": {"synthesized_feedback": "Different label."}},
    ]


def _judge_trace_artifacts() -> list[dict[str, object]]:
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


def test_load_feedback_artifacts_handles_text_json_and_missing_directories(tmp_path: Path) -> None:
    (tmp_path / "feedback").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "reviews").mkdir()

    (tmp_path / "feedback" / "iter0_run0.txt").write_text("raw feedback text", encoding="utf-8")
    (tmp_path / "models" / "iter0_run0.txt").write_text("def model_a(): return 1", encoding="utf-8")
    (tmp_path / "reviews" / "iter0.json").write_text('{"summary": "ok", "synthesized_feedback": "Use a simpler mechanism."}', encoding="utf-8")
    (tmp_path / "reviews" / "iter1.json").write_text('{"summary": "broken"', encoding="utf-8")

    rows = load_feedback_artifacts(tmp_path)

    assert {row["kind"] for row in rows} == {"feedback", "model_code", "review"}
    assert next(row for row in rows if row["kind"] == "feedback")["content"] == "raw feedback text"
    assert next(row for row in rows if row["kind"] == "model_code")["content"] == "def model_a(): return 1"
    good_review = next(row for row in rows if row["path"] == "reviews/iter0.json")
    assert good_review["payload"]["synthesized_feedback"] == "Use a simpler mechanism."
    bad_review = next(row for row in rows if row["path"] == "reviews/iter1.json")
    assert bad_review["error"]
    assert bad_review["raw_text"].startswith('{"summary": "broken"')
    assert load_feedback_artifacts(tmp_path / "missing") == []


def test_load_judge_trace_artifacts_handles_absent_and_malformed_json(tmp_path: Path) -> None:
    assert load_judge_trace_artifacts(tmp_path / "missing") == []

    (tmp_path / "judge").mkdir()
    (tmp_path / "judge" / "iter0_run0.json").write_text('{"verdict": true', encoding="utf-8")

    rows = load_judge_trace_artifacts(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "judge_trace"
    assert row["path"] == "judge/iter0_run0.json"
    assert row["error"]
    assert row["raw_text"].startswith('{"verdict": true')


def test_render_results_tab_centralizes_registry_and_trace_artifacts(monkeypatch) -> None:
    stub = _install_stub(monkeypatch)

    dashboard_views.render_results_tab(
        _summary_frame(),
        results_dir=Path("/tmp/results"),
        feedback_artifacts=_feedback_artifacts(),
        snapshot={"client_entries": {"0": {"status": "running"}}, "iteration_history": []},
        trace_artifacts=_judge_trace_artifacts(),
    )

    section_labels = [value for kind, value in stub.events if kind == "subheader"]
    expander_labels = [value[0] for kind, value in stub.events if kind == "expander"]
    assert "Artifacts" in section_labels
    assert "Artifacts browser" in section_labels
    assert "Registry JSON" in expander_labels
    assert any(label.startswith("Artifact: Feedback") for label in expander_labels)
    assert any(label.startswith("Artifact: Model Code") for label in expander_labels)
    assert any(label.startswith("Artifact: Review") for label in expander_labels)
    assert any(label.startswith("Judge trace") for label in expander_labels)

    artifact_labels = [label for label in expander_labels if label.startswith(("Artifact:", "Judge trace", "Registry JSON"))]
    assert len(artifact_labels) == len(set(artifact_labels))


def test_render_models_tab_uses_collapsed_handoff_instead_of_verbose_artifacts(monkeypatch) -> None:
    stub = _install_stub(monkeypatch)
    stub.selectbox_value = "diagnostics.duckdb::1::train"

    dashboard_views.render_models_tab(_summary_frame(), snapshot={"baseline": {"name": "baseline", "metric_value": 12.0}}, top_n=1)

    section_labels = [value for kind, value in stub.events if kind == "subheader"]
    expander_labels = [value[0] for kind, value in stub.events if kind == "expander"]
    assert "Model artifacts" in expander_labels
    assert "Selected model details" in expander_labels
    assert "Raw model detail payload" in expander_labels
    assert "Selected model detail" in section_labels
    assert "Model code" in section_labels
    assert "Validation errors" in section_labels
    assert "Parameter recovery" in section_labels
    assert "R² details" in section_labels
    assert any(kind == "code" for kind, _ in stub.events)


def test_render_judge_tab_uses_collapsed_handoff_instead_of_trace_dump(monkeypatch) -> None:
    stub = _install_stub(monkeypatch)

    dashboard_views.render_judge_tab(
        [
            {
                "iteration": 0,
                "synthesized_feedback": {"default": "Improve recovery."},
                "verdict": {"accepted": True},
                "failed": False,
                "timestamp": "2025-01-01T00:00:00",
                "error": None,
            }
        ],
        trace_artifacts=_judge_trace_artifacts(),
    )

    expander_labels = [value[0] for kind, value in stub.events if kind == "expander"]
    assert "Judge artifacts" in expander_labels
    assert not any(label.startswith("Judge trace") for label in expander_labels)
    assert "Raw judge trace payload" not in expander_labels
    assert not any(kind == "json" for kind, _ in stub.events)
