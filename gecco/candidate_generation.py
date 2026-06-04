"""Candidate generation service extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        search: Any,
        iteration: int,
        run_idx: int,
        feedback: str,
        cmg_cfg: Any,
    ) -> CandidateGenerationResult:
        """Run one candidate-generation iteration."""

        n_models = cmg_cfg.n_models
        tag = search._file_tag()
        search._set_activity(f"generating centralized candidates (iter {iteration})")

        try:
            client_config = (
                getattr(search.cfg.clients, search.client_id, None) if search.client_id else None
            )
            naive_enabled = bool(
                client_config
                and getattr(getattr(client_config, "naive_ideation", None), "enabled", False)
            )

            if naive_enabled:
                code_text, parsed_models = search.generate_models_naive(feedback, n_models=n_models)
            else:
                prompt = search.prompt_builder.build_input_prompt(
                    feedback_text=feedback, n_models=n_models
                )
                code_text, parsed_models = search.generate_models(prompt, n_models=n_models)

            participant = (
                getattr(search.df, "participant", [None])[0]
                if getattr(search, "df", None) is not None
                else None
            )
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

            search.shared_registry.set_candidate_models(iteration, candidates, search.client_id)
            search.shared_registry.set_generator_status(
                iteration=iteration,
                client_id=search.client_id,
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
            if search.shared_registry is not None:
                search.shared_registry.set_generator_status(
                    iteration=iteration,
                    client_id=search.client_id,
                    status="failed",
                    n_candidates=0,
                    error=str(exc),
                )
            raise
