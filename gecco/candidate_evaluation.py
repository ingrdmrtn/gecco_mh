"""Candidate evaluation service extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gecco.artifacts import ArtifactStore


@dataclass(slots=True)
class CandidateEvaluationResult:
    """Structured result from an evaluation pass."""

    iteration_results: list[dict[str, Any]]
    model_file: Path


class CandidateEvaluator:
    """Evaluate and, when needed, repair a generated candidate."""

    def __init__(self, artifact_store: ArtifactStore):
        self.artifact_store = artifact_store

    def evaluate_iteration(
        self,
        *,
        search: Any,
        iteration: int,
        run_idx: int,
        feedback: str,
        cmg_cfg: Any,
        baseline_bic: float | None,
    ) -> CandidateEvaluationResult:
        """Evaluate the candidate assigned to this client."""

        idx = search._cmg_evaluator_index(cmg_cfg)
        if idx is None:
            raise ValueError(
                f"CMG evaluator client_id must be numeric in range 0..{cmg_cfg.n_models - 1}; got {search.client_id!r}"
            )

        barrier_timeout = getattr(getattr(search.cfg.judge, "barrier", None), "client_wait_seconds", 1800)
        gen_data = search.shared_registry.wait_for_candidate_models(
            iteration,
            timeout_seconds=barrier_timeout,
        )
        if gen_data is None:
            raise TimeoutError(f"Timed out waiting for CMG candidates for iteration {iteration}")

        candidates = gen_data.get("candidates", [])
        candidate = next((c for c in candidates if c.get("index") == idx), None)
        if candidate is None:
            raise ValueError(f"No CMG candidate {idx} for iteration {iteration}")

        func_name = candidate.get("func_name", f"cognitive_model{idx + 1}")
        display_name = candidate.get("name", func_name)
        model_dict = {
            "func_name": func_name,
            "name": display_name,
            "code": candidate.get("code", ""),
            "parameters": candidate.get("parameters", []),
        }

        tag = search._file_tag()
        participant = getattr(search.df, "participant", [None])[0] if getattr(search, "df", None) is not None else None
        model_file = self.artifact_store.candidate_model_path(
            iteration=iteration,
            run_idx=run_idx,
            tag=tag,
            participant=participant,
        )
        model_file.write_text(model_dict.get("code", ""), encoding="utf-8")

        iteration_results: list[dict[str, Any]] = []
        syntax_retry_count = 0
        max_syntax_retries = getattr(getattr(search.cfg, "validation", None), "max_syntax_retries", 2)
        current_model_dict = model_dict

        while syntax_retry_count <= max_syntax_retries:
            result, should_stop = search._fit_candidate_model(
                model_dict=current_model_dict,
                model_idx=idx,
                n_models=cmg_cfg.n_models,
                it=iteration,
                run_idx=run_idx,
                tag=tag,
                model_file=model_file,
                baseline_bic=baseline_bic,
            )

            is_repairable_error = search._is_cmg_repairable_error(result)

            if not is_repairable_error or syntax_retry_count >= max_syntax_retries:
                if result is not None:
                    iteration_results = [result]
                break

            syntax_retry_count += 1
            search._update_registry(iteration, [], status="retrying")
            current_model_dict = search._repair_cmg_candidate(
                candidate=candidate,
                current_model_dict=current_model_dict,
                error_result=result,
                expected_func_name=func_name,
                iteration=iteration,
                candidate_index=idx,
            )
            model_file.write_text(current_model_dict.get("code", ""), encoding="utf-8")

        if result is not None:
            result.setdefault("candidate_index", idx)
            result.setdefault("expected_func_name", func_name)
            result.setdefault("display_name", display_name)

        search._finalize_iteration_results(
            it=iteration,
            run_idx=run_idx,
            tag=tag,
            iteration_results=iteration_results,
        )
        return CandidateEvaluationResult(iteration_results=iteration_results, model_file=model_file)
