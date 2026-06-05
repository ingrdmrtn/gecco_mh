"""Feedback coordination helpers for the orchestrated judge pipeline."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


run_orchestrated_judge_pipeline = None


class FeedbackCoordinator:
    """Resolve feedback text for a search iteration."""

    def resolve_feedback(
        self,
        *,
        judge: Any,
        cfg: Any,
        results_dir: Any,
        iteration: int,
        run_idx: int,
        tag: str,
        best_model: str | None,
        best_metric: float | None,
        recovery_failures: list[dict[str, Any]] | None,
        prev_had_success: bool,
        persona_name: str = "default",
        set_activity: Any | None = None,
    ) -> tuple[str, Any]:
        """Return feedback text and a verdict-like payload."""

        if judge is None:
            return "", SimpleNamespace(synthesized_feedback="", key_recommendations=[])

        if set_activity is not None:
            set_activity(f"judge synthesis (iter {iteration})")
        from gecco import run_gecco as run_gecco_module

        runner = run_orchestrated_judge_pipeline or run_gecco_module.run_orchestrated_judge_pipeline
        artifact = runner(
            judge=judge,
            cfg=cfg,
            results_dir=results_dir,
            iteration=iteration,
            run_idx=run_idx,
            tag=tag,
            best_model=best_model,
            best_metric=best_metric,
            recovery_failures=recovery_failures if recovery_failures else None,
            prev_had_success=prev_had_success,
        )
        feedback = artifact.feedback_for_persona(persona_name)
        verdict = SimpleNamespace(
            synthesized_feedback=feedback,
            key_recommendations=artifact.key_recommendations,
        )
        return feedback, verdict
