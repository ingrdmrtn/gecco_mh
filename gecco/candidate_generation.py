"""Candidate generation service extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from gecco.artifacts import ArtifactStore


@dataclass(slots=True)
class CandidateGenerationResult:
    """Structured result from a generation pass."""

    code_text: str
    parsed_models: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    model_file: Path


class CandidateGenerator:
    """Generate and persist candidate models for one iteration."""

    def __init__(self, artifact_store: ArtifactStore):
        self.artifact_store = artifact_store

    def generate_iteration(
        self,
        *,
        iteration: int,
        run_idx: int,
        feedback: str,
        cmg_cfg: Any,
        tag: str,
        client_id: Any,
        naive_enabled: bool,
        build_prompt: Callable[..., str],
        generate_models: Callable[..., tuple[str, list[dict[str, Any]]]],
        generate_models_naive: Callable[..., tuple[str, list[dict[str, Any]]]],
        shared_registry: Any,
        participant: str | None = None,
        set_activity: Callable[[str], None] | None = None,
    ) -> CandidateGenerationResult:
        """Run one candidate-generation iteration."""

        n_models = cmg_cfg.n_models
        if set_activity is not None:
            set_activity(f"generating centralized candidates (iter {iteration})")

        try:
            if naive_enabled:
                code_text, parsed_models = generate_models_naive(feedback, n_models=n_models)
            else:
                prompt = build_prompt(feedback_text=feedback, n_models=n_models)
                code_text, parsed_models = generate_models(prompt, n_models=n_models)

            model_file = self.artifact_store.write_candidate_artifacts(
                iteration=iteration,
                run_idx=run_idx,
                tag=tag,
                code_text=code_text,
                parsed_models=parsed_models,
                participant=participant,
            )

            candidates: list[dict[str, Any]] = []
            for index, model in enumerate(parsed_models):
                func_name = f"cognitive_model{index + 1}"
                candidates.append(
                    {
                        "index": index,
                        "func_name": func_name,
                        "name": model.get("name", func_name),
                        "code": model.get("code", ""),
                        "rationale": model.get("rationale", ""),
                        "analysis": model.get("analysis", ""),
                        "parameters": model.get("parameters", []),
                        "validation_failed": model.get("validation_failed", False),
                        "validation_errors": model.get("validation_errors", []),
                    }
                )

            if len(candidates) != n_models:
                raise ValueError(
                    f"CMG generator produced {len(candidates)} candidates, expected {n_models}"
                )

            shared_registry.set_candidate_models(iteration, candidates, client_id)
            shared_registry.set_generator_status(
                iteration=iteration,
                client_id=client_id,
                status="complete",
                n_candidates=len(candidates),
            )
            return CandidateGenerationResult(
                code_text=code_text,
                parsed_models=parsed_models,
                candidates=candidates,
                model_file=model_file,
            )
        except Exception as exc:
            if shared_registry is not None:
                shared_registry.set_generator_status(
                    iteration=iteration,
                    client_id=client_id,
                    status="failed",
                    n_candidates=0,
                    error=str(exc),
                )
            raise
