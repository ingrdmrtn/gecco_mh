"""Helpers for the hard orchestrated judge pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from config.schema import judge_has_capability


class FeedbackArtifact(BaseModel):
    """Canonical persisted output from the orchestrated judge pipeline.

    Attributes:
        iteration: Iteration index being judged.
        run_idx: Run index for the current search.
        tag: Optional file tag for distributed or persona-specific runs.
        timestamp: UTC ISO-8601 creation timestamp.
        tool_call_count: Number of analysis tool calls made by the judge.
        wall_time_seconds: Total analysis wall time.
        best_bic: Best score seen when the judge ran.
        tool_call_trace: Compact tool trace for auditability.
        full_trace: Full tool loop trace.
        per_angle: Structured angle analyses returned by synthesis.
        key_recommendations: Deduplicated recommendations across personas.
        synthesized_feedback: Final feedback keyed by persona name.
        personas: Ordered persona keys included in ``synthesized_feedback``.
        stuck_search: Whether stuck-search handling was triggered.
        short_circuit: Whether synthesis was short-circuited.
        no_substantive_feedback: Whether judge capabilities produced no feedback.
        random_feedback_only: Whether the random-feedback-only mode was used.
        best_model_code_included: Whether the final feedback already embeds best-model code.
    """

    iteration: int
    run_idx: int
    tag: str
    timestamp: str
    tool_call_count: int
    wall_time_seconds: float
    best_bic: float | None = None
    tool_call_trace: list[dict[str, Any]] = Field(default_factory=list)
    full_trace: list[dict[str, Any]] = Field(default_factory=list)
    per_angle: list[dict[str, Any]] = Field(default_factory=list)
    key_recommendations: list[str] = Field(default_factory=list)
    synthesized_feedback: dict[str, str] = Field(default_factory=dict)
    personas: list[str] = Field(default_factory=list)
    stuck_search: bool = False
    short_circuit: bool = False
    no_substantive_feedback: bool = False
    random_feedback_only: bool = False
    best_model_code_included: bool = False

    def feedback_for_persona(self, persona_name: str = "default") -> str:
        """Return feedback text for one persona with sensible fallback.

        Args:
            persona_name: Preferred persona key.

        Returns:
            The matching feedback string, or the default entry, or an empty string.
        """
        return self.synthesized_feedback.get(
            persona_name,
            self.synthesized_feedback.get("default", ""),
        )


def build_feedback_artifact(
    *,
    iteration: int,
    run_idx: int,
    tag: str,
    analysis_data: dict[str, Any],
    synthesized_feedback: dict[str, str],
    verdict_payloads: list[dict[str, Any]],
    best_model: str | None,
    best_metric: float | None,
    include_best_model_code: bool,
) -> FeedbackArtifact:
    """Build the canonical feedback artifact for any orchestrated judge run.

    Args:
        iteration: Iteration index being judged.
        run_idx: Run index for the current search.
        tag: Optional file tag.
        analysis_data: Output from ``ToolUsingJudge.get_feedback_analysis``.
        synthesized_feedback: Persona-keyed feedback text before final packaging.
        verdict_payloads: Structured verdict payloads from synthesis.
        best_model: Best model code to append when the capability is enabled.
        best_metric: Best metric associated with ``best_model``.
        include_best_model_code: Whether to embed best-model code in the final feedback.

    Returns:
        The canonical ``FeedbackArtifact``.
    """
    final_feedback = dict(synthesized_feedback)
    if include_best_model_code and best_model is not None:
        best_metric_str = f"{best_metric:.2f}" if best_metric is not None else "N/A"
        appendix = (
            f"\n\n---\nBest model code so far (BIC={best_metric_str}):\n"
            f"```python\n{best_model}\n```"
        )
        final_feedback = {
            persona_name: f"{feedback_text}{appendix}"
            for persona_name, feedback_text in final_feedback.items()
        }

    key_recommendations: list[str] = []
    seen_recommendations: set[str] = set()
    per_angle: list[dict[str, Any]] = []
    for verdict_payload in verdict_payloads:
        raw_per_angle = verdict_payload.get("per_angle", []) or []
        if raw_per_angle:
            per_angle = [
                angle.model_dump() if hasattr(angle, "model_dump") else angle
                for angle in raw_per_angle
            ]
        for recommendation in verdict_payload.get("key_recommendations", []) or []:
            if recommendation not in seen_recommendations:
                seen_recommendations.add(recommendation)
                key_recommendations.append(recommendation)

    return FeedbackArtifact(
        iteration=iteration,
        run_idx=run_idx,
        tag=tag,
        timestamp=datetime.now(timezone.utc).isoformat(),
        tool_call_count=len(analysis_data.get("trace", [])),
        wall_time_seconds=float(analysis_data.get("wall_time", 0.0)),
        best_bic=analysis_data.get("best_bic"),
        tool_call_trace=analysis_data.get("trace", []),
        full_trace=analysis_data.get("full_trace", []),
        per_angle=per_angle,
        key_recommendations=key_recommendations,
        synthesized_feedback=final_feedback,
        personas=list(final_feedback.keys()),
        stuck_search=bool(analysis_data.get("is_stuck", False)),
        short_circuit=bool(analysis_data.get("short_circuit", False)),
        no_substantive_feedback=bool(analysis_data.get("no_capabilities", False)),
        random_feedback_only=bool(analysis_data.get("random_feedback_only", False)),
        best_model_code_included=include_best_model_code and best_model is not None,
    )


def persist_feedback_artifact(
    *, artifact: FeedbackArtifact, results_dir: Path | None
) -> Path | None:
    """Persist a feedback artifact to the canonical judge directory.

    Args:
        artifact: Artifact to serialise.
        results_dir: Root results directory for the run.

    Returns:
        Path to the written JSON artifact, or ``None`` if persistence is disabled.
    """
    if results_dir is None:
        return None

    judge_dir = results_dir / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    suffix = artifact.tag or ""
    artifact_path = judge_dir / f"iter{artifact.iteration}{suffix}_run{artifact.run_idx}.json"
    with artifact_path.open("w", encoding="utf-8") as file_obj:
        json.dump(artifact.model_dump(), file_obj, indent=2)
    return artifact_path


def run_orchestrated_judge_pipeline(
    *,
    judge: Any,
    cfg: Any,
    results_dir: Path | None,
    iteration: int,
    run_idx: int,
    tag: str,
    best_model: str | None,
    best_metric: float | None,
    recovery_failures: list[dict[str, Any]] | None,
    prev_had_success: bool,
) -> FeedbackArtifact:
    """Run the shared orchestrated judge pipeline for local or distributed use.

    Args:
        judge: Initialised ``ToolUsingJudge`` instance.
        cfg: Loaded runtime config.
        results_dir: Results directory for artifact persistence.
        iteration: Iteration index being judged.
        run_idx: Run index for the current search.
        tag: Optional file tag.
        best_model: Best model code available to the judge.
        best_metric: Best metric associated with ``best_model``.
        recovery_failures: Optional recovery-failure context.
        prev_had_success: Whether the previous iteration produced any runnable model.

    Returns:
        Persisted ``FeedbackArtifact`` for the completed judge pass.
    """
    analysis_data = judge.get_feedback_analysis(
        iteration=iteration,
        run_idx=run_idx,
        tag=tag,
        best_model=best_model,
        best_metric=best_metric,
        recovery_failures=recovery_failures,
        prev_had_success=prev_had_success,
    )

    verdict_payloads: list[dict[str, Any]] = []
    if analysis_data.get("short_circuit"):
        if getattr(getattr(cfg, "centralized_model_generation", None), "enabled", False):
            generator_name = getattr(
                getattr(cfg, "centralized_model_generation", None),
                "generator_client",
                "generator",
            )
            synthesized_feedback = {generator_name: analysis_data["analysis_text"]}
        else:
            synthesized_feedback = {"default": analysis_data["analysis_text"]}
    else:
        clients = getattr(cfg, "clients", {}) or {}
        if not isinstance(clients, dict):
            clients = {
                name: getattr(clients, name)
                for name in vars(clients).keys()
                if not name.startswith("_")
            }

        synthesized_feedback: dict[str, str] = {}
        if getattr(getattr(cfg, "centralized_model_generation", None), "enabled", False):
            generator_name = getattr(
                getattr(cfg, "centralized_model_generation", None),
                "generator_client",
                "generator",
            )
            persona_config = clients.get(generator_name) if clients else None
            persona_suffix = ""
            if persona_config and hasattr(persona_config, "llm"):
                persona_suffix = getattr(
                    persona_config.llm, "feedback_guidance", None
                ) or getattr(persona_config.llm, "system_prompt_suffix", "")
            feedback_text, verdict_payload = judge.synthesize_for_persona(
                analysis_data,
                persona_name=generator_name,
                persona_suffix=persona_suffix,
                persona_config=persona_config,
            )
            synthesized_feedback[generator_name] = feedback_text
            verdict_payloads.append(verdict_payload)
        elif clients:
            for persona_name, persona_config in clients.items():
                persona_suffix = ""
                if persona_config and hasattr(persona_config, "llm"):
                    persona_suffix = getattr(
                        persona_config.llm, "feedback_guidance", None
                    ) or getattr(persona_config.llm, "system_prompt_suffix", "")
                feedback_text, verdict_payload = judge.synthesize_for_persona(
                    analysis_data,
                    persona_name=persona_name,
                    persona_suffix=persona_suffix,
                    persona_config=persona_config,
                )
                synthesized_feedback[persona_name] = feedback_text
                verdict_payloads.append(verdict_payload)
        else:
            feedback_text, verdict_payload = judge.synthesize_for_persona(
                analysis_data,
                persona_name="default",
                persona_suffix="",
            )
            synthesized_feedback["default"] = feedback_text
            verdict_payloads.append(verdict_payload)

    artifact = build_feedback_artifact(
        iteration=iteration,
        run_idx=run_idx,
        tag=tag,
        analysis_data=analysis_data,
        synthesized_feedback=synthesized_feedback,
        verdict_payloads=verdict_payloads,
        best_model=best_model,
        best_metric=best_metric,
        include_best_model_code=judge_has_capability(cfg, "best_model_code"),
    )
    persist_feedback_artifact(artifact=artifact, results_dir=results_dir)
    return artifact
