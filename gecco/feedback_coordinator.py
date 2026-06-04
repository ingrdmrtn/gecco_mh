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
        search: Any,
        iteration: int,
        run_idx: int,
        tag: str,
        best_model: str | None,
        best_metric: float | None,
        recovery_failures: list[dict[str, Any]] | None,
        prev_had_success: bool,
    ) -> tuple[str, Any]:
        """Return feedback text and a verdict-like payload."""

        if search.tool_judge is None:
            return "", SimpleNamespace(synthesized_feedback="", key_recommendations=[])

        search._set_activity(f"judge synthesis (iter {iteration})")
        from gecco import run_gecco as run_gecco_module

        runner = run_orchestrated_judge_pipeline or run_gecco_module.run_orchestrated_judge_pipeline
        artifact = runner(
            judge=search.tool_judge,
            cfg=search.cfg,
            results_dir=search.results_dir,
            iteration=iteration,
            run_idx=run_idx,
            tag=tag,
            best_model=best_model,
            best_metric=best_metric,
            recovery_failures=recovery_failures if recovery_failures else None,
            prev_had_success=prev_had_success,
        )
        persona_name = search.client_id or "default"
        feedback = artifact.feedback_for_persona(persona_name)
        verdict = SimpleNamespace(
            synthesized_feedback=feedback,
            key_recommendations=artifact.key_recommendations,
        )
        return feedback, verdict
