"""Focused persistence helpers for search artefacts.

These helpers centralise file and DuckDB writes so that the search monolith can
delegate persistence to a small interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gecco.diagnostic_store.store import DiagnosticStore
from gecco.run_context import RunContext
from gecco.utils import TimestampedConsole


console = TimestampedConsole()


class ArtifactStore:
    """Persist model, feedback, and iteration artefacts for a run."""

    def __init__(
        self,
        run_context: RunContext | str | Path,
        diagnostic_store: DiagnosticStore | None = None,
    ):
        self.run_context = run_context if isinstance(run_context, RunContext) else None
        self.results_dir = run_context.results_dir if isinstance(run_context, RunContext) else Path(run_context)
        self.diagnostic_store = diagnostic_store

    def _model_dir(self) -> Path:
        path = self.results_dir / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _bics_dir(self) -> Path:
        path = self.results_dir / "bics"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _feedback_dir(self) -> Path:
        path = self.results_dir / "feedback"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def candidate_model_path(
        self,
        *,
        iteration: int,
        run_idx: int,
        tag: str = "",
        participant: str | None = None,
    ) -> Path:
        """Return the canonical candidate model file path."""

        if self.run_context is not None:
            return self.run_context.candidate_model_path(
                iteration=iteration,
                run_idx=run_idx,
                tag=tag,
                participant=participant,
            )
        suffix = f"_participant{participant}" if participant else ""
        return self._model_dir() / f"iter{iteration}{tag}_run{run_idx}{suffix}.txt"

    def write_candidate_artifacts(
        self,
        *,
        iteration: int,
        run_idx: int,
        tag: str,
        code_text: str,
        parsed_models: list[dict[str, Any]],
        participant: str | None = None,
    ) -> Path:
        """Persist raw and structured candidate-generation artefacts."""

        model_file = self.candidate_model_path(
            iteration=iteration,
            run_idx=run_idx,
            tag=tag,
            participant=participant,
        )
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_text(code_text, encoding="utf-8")

        if parsed_models:
            structured_file = model_file.with_suffix(".json")
            with structured_file.open("w", encoding="utf-8") as file_obj:
                json.dump(
                    [
                        {
                            "name": model["name"],
                            "rationale": model.get("rationale", ""),
                            "analysis": model.get("analysis", ""),
                            "parameters": model.get("parameters", []),
                        }
                        for model in parsed_models
                    ],
                    file_obj,
                    indent=2,
                )

        return model_file

    def write_feedback_text(
        self,
        *,
        iteration: int,
        run_idx: int,
        tag: str,
        feedback: str,
        participant: str | None = None,
    ) -> Path:
        """Persist the plain-text feedback artefact."""

        if self.run_context is not None:
            feedback_file = self.run_context.feedback_path(
                iteration=iteration,
                run_idx=run_idx,
                tag=tag,
                participant=participant,
            )
        else:
            suffix = f"_participant{participant}" if participant else ""
            feedback_file = self._feedback_dir() / f"iter{iteration}{tag}_run{run_idx}{suffix}.txt"
        feedback_file.write_text(feedback, encoding="utf-8")
        return feedback_file

    def write_review(self, review: dict[str, Any], *, iteration: int, tag: str = "") -> Path:
        """Persist a review payload for inspection."""

        review_dir = self.results_dir / "reviews"
        review_dir.mkdir(parents=True, exist_ok=True)
        existing = list(review_dir.glob("iter*.json"))
        review_file = review_dir / f"iter{len(existing) + 1}{tag}.json"
        review_file.write_text(json.dumps(review, indent=2), encoding="utf-8")
        return review_file

    def write_iteration_results(
        self,
        *,
        iteration: int,
        run_idx: int,
        tag: str,
        iteration_results: list[dict[str, Any]],
        client_id: Any = None,
        results_source: Any = None,
    ) -> bool:
        """Persist iteration results to JSON and DuckDB.

        Returns:
            ``True`` when at least one runnable model was observed.
        """

        participant = None
        is_individual = bool(self.run_context.is_individual) if self.run_context is not None else False
        if is_individual and results_source is not None and hasattr(results_source, "participant"):
            participant = results_source.participant[0]

        ppc_results_map = {
            row["function_name"]: row["ppc"]
            for row in iteration_results
            if "ppc" in row
        }

        if self.diagnostic_store is not None:
            try:
                self.diagnostic_store.write_iteration(
                    iteration=iteration,
                    run_idx=run_idx,
                    iteration_results=iteration_results,
                    ppc_results=ppc_results_map if ppc_results_map else None,
                    tag=tag,
                    client_id=client_id,
                )
            except Exception as exc:  # pragma: no cover - surfaced via search logs
                console.print(f"[yellow]Diagnostic store write failed:[/] {exc}")

        if self.run_context is not None:
            bic_file = self.run_context.iteration_results_path(
                iteration=iteration,
                run_idx=run_idx,
                tag=tag,
                participant=participant,
            )
        else:
            suffix = f"_participant{participant}" if participant else ""
            bic_file = self._bics_dir() / f"iter{iteration}{tag}_run{run_idx}{suffix}.json"
        bic_file.write_text(
            json.dumps(iteration_results, indent=2, default=str), encoding="utf-8"
        )

        had_runnable_model = (
            any(
                row.get("metric_name")
                not in ("VALIDATION_ERROR", "FIT_ERROR", "RECOVERY_FAILED", None)
                for row in iteration_results
            )
            if iteration_results
            else False
        )
        return had_runnable_model
