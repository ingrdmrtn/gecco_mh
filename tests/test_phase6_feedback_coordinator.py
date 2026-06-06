"""Direct tests for the Phase 6 feedback coordinator boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gecco.feedback_coordinator import FeedbackCoordinator


def test_feedback_coordinator_uses_injected_orchestrated_pipeline_only(tmp_path: Path):
    """The coordinator should not fall back to monolith-owned imports or helpers."""

    judge = MagicMock()
    artifact = MagicMock()
    artifact.feedback_for_persona.return_value = "use simpler models"
    original_import = __import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "gecco.run_gecco":
            raise AssertionError("FeedbackCoordinator should not import gecco.run_gecco")
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=guarded_import):
        feedback, verdict = FeedbackCoordinator(
            run_orchestrated_judge_pipeline=MagicMock(return_value=artifact)
        ).resolve_feedback(
            judge=judge,
            cfg=SimpleNamespace(),
            results_dir=tmp_path,
            iteration=4,
            run_idx=1,
            tag="",
            best_model=None,
            best_metric=None,
            recovery_failures=[{"name": "model_a"}],
            prev_had_success=True,
            persona_name="client-a",
            set_activity=MagicMock(),
        )

    assert feedback == "use simpler models"
    assert verdict.synthesized_feedback == "use simpler models"
    artifact.feedback_for_persona.assert_called_once_with("client-a")
