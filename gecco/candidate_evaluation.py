"""Candidate evaluation service extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
        iteration: int,
        run_idx: int,
        cmg_cfg: Any,
        tag: str,
        client_id: Any,
        evaluator_index: int | None,
        baseline_bic: float | None,
        shared_registry: Any,
        fit_candidate_model: Callable[..., tuple[dict | None, bool]],
        is_repairable_error: Callable[[dict | None], bool],
        repair_candidate: Callable[..., dict],
        update_registry: Callable[..., None],
        finalize_iteration_results: Callable[..., bool],
        max_syntax_retries: int,
        barrier_timeout_seconds: int,
        participant: str | None = None,
    ) -> CandidateEvaluationResult:
        """Evaluate the candidate assigned to this client."""

        idx = evaluator_index
        if idx is None:
            raise ValueError(
                f"CMG evaluator client_id must be numeric in range 0..{cmg_cfg.n_models - 1}; got {client_id!r}"
            )

        gen_data = shared_registry.wait_for_candidate_models(
            iteration,
            timeout_seconds=barrier_timeout_seconds,
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

        model_file = self.artifact_store.candidate_model_path(
            iteration=iteration,
            run_idx=run_idx,
            tag=tag,
            participant=participant,
        )
        model_file.write_text(model_dict.get("code", ""), encoding="utf-8")

        iteration_results: list[dict[str, Any]] = []
        syntax_retry_count = 0
        current_model_dict = model_dict

        while syntax_retry_count <= max_syntax_retries:
            result, should_stop = fit_candidate_model(
                model_dict=current_model_dict,
                model_idx=idx,
                n_models=cmg_cfg.n_models,
                it=iteration,
                run_idx=run_idx,
                tag=tag,
                model_file=model_file,
                baseline_bic=baseline_bic,
            )

            repairable_error = is_repairable_error(result)

            if not repairable_error or syntax_retry_count >= max_syntax_retries:
                if result is not None:
                    iteration_results = [result]
                break

            syntax_retry_count += 1
            update_registry(iteration, [], status="retrying")
            current_model_dict = repair_candidate(
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

        finalize_iteration_results(
            it=iteration,
            run_idx=run_idx,
            tag=tag,
            iteration_results=iteration_results,
        )
        return CandidateEvaluationResult(iteration_results=iteration_results, model_file=model_file)
