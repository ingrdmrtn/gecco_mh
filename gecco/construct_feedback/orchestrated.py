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
        metadata: Structured provenance and shortcut metadata.
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
    metadata: dict[str, Any] = Field(default_factory=dict)
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
            The matching feedback string, or the default entry, or an empty string
            when the artifact has no feedback at all.

        Raises:
            ValueError: If feedback exists, but neither ``persona_name`` nor
                ``default`` is available.
        """
        if persona_name in self.synthesized_feedback:
            return self.synthesized_feedback[persona_name]
        if "default" in self.synthesized_feedback:
            return self.synthesized_feedback["default"]
        if self.synthesized_feedback:
            available = ", ".join(sorted(self.synthesized_feedback))
            raise ValueError(
                f"No feedback available for persona '{persona_name}'. "
                f"Available personas: {available}."
            )
        return ""


def _normalise_clients(cfg: Any) -> dict[str, Any]:
    """Return client configs as a plain mapping."""
    clients = getattr(cfg, "clients", {}) or {}
    if isinstance(clients, dict):
        return dict(clients)
    return {
        name: getattr(clients, name)
        for name in vars(clients).keys()
        if not name.startswith("_")
    }


def _mapping_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a key from either a mapping or an attribute container."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _resolve_synthesis_personas(cfg: Any) -> dict[str, Any | None]:
    """Resolve the personas that should receive synthesis output."""
    clients = _normalise_clients(cfg)
    cmg_cfg = getattr(cfg, "centralized_model_generation", None)
    if getattr(cmg_cfg, "enabled", False):
        generator_name = getattr(cmg_cfg, "generator_client", "generator")
        return {generator_name: clients.get(generator_name)}
    if judge_has_capability(cfg, "persona_synthesis") and clients:
        return clients
    return {"default": None}


def _persona_suffix(persona_config: Any | None) -> str:
    """Extract persona-specific synthesis guidance."""
    llm_config = _mapping_get(persona_config, "llm")
    if llm_config is None:
        return ""
    return _mapping_get(llm_config, "feedback_guidance") or _mapping_get(
        llm_config, "system_prompt_suffix", ""
    )


def _coerce_feedback_map(
    synthesized_feedback: dict[str, str] | str | None,
    *,
    default_persona: str = "default",
) -> dict[str, str]:
    """Normalise feedback into the canonical persona-keyed mapping."""
    if isinstance(synthesized_feedback, dict):
        return dict(synthesized_feedback)
    if synthesized_feedback is None:
        return {default_persona: ""}
    return {default_persona: synthesized_feedback}


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
        metadata=dict(analysis_data.get("metadata", {}) or {}),
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
        verdict_payloads.extend(analysis_data.get("verdict_payloads", []) or [])
        shortcut_verdict_payload = analysis_data.get("shortcut_verdict_payload")
        if shortcut_verdict_payload:
            verdict_payloads.append(shortcut_verdict_payload)
        synthesized_feedback = _coerce_feedback_map(
            analysis_data.get("synthesized_feedback", analysis_data.get("analysis_text"))
        )
    else:
        synthesized_feedback: dict[str, str] = {}
        for persona_name, persona_config in _resolve_synthesis_personas(cfg).items():
            feedback_text, verdict_payload = judge.synthesize_for_persona(
                analysis_data,
                persona_name=persona_name,
                persona_suffix=_persona_suffix(persona_config),
                persona_config=persona_config,
            )
            synthesized_feedback[persona_name] = feedback_text
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
