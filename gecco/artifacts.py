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


def _json_default(value: Any) -> Any:
    """Convert common non-serialisable values for inspection JSON output."""

    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


class ArtifactStore:
    """Persist model, feedback, and iteration artefacts for a run."""

    def __init__(
        self,
        run_context: RunContext | str | Path,
        diagnostic_store: DiagnosticStore | None = None,
        inspection_output_enabled: bool = False,
    ):
        self.run_context = run_context if isinstance(run_context, RunContext) else None
        self.results_dir = run_context.results_dir if isinstance(run_context, RunContext) else Path(run_context)
        self.diagnostic_store = diagnostic_store
        self.inspection_output_enabled = inspection_output_enabled

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
        """Persist candidate artefacts and optional inspection JSON."""

        model_file = self.candidate_model_path(
            iteration=iteration,
            run_idx=run_idx,
            tag=tag,
            participant=participant,
        )
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_text(code_text, encoding="utf-8")

        if self.inspection_output_enabled and parsed_models:
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

    def write_best_metric_inspection(
        self,
        *,
        run_idx: int,
        tag: str,
        metric_value: float,
        val_metric_value: float | None = None,
        val_mean_nll: float | None = None,
        val_eval_metrics: list[float] | None = None,
        val_per_participant_nll: list[float] | None = None,
        participant: str | None = None,
    ) -> tuple[Path | None, Path | None]:
        """Persist best-metric inspection JSON when inspection output is enabled."""

        if not self.inspection_output_enabled:
            return None, None

        suffix = f"_participant{participant}" if participant else ""
        best_bic_file = self._bics_dir() / f"best_bic{tag}_{run_idx}{suffix}.json"
        best_bic_file.write_text(
            json.dumps({"bic": metric_value}, default=_json_default),
            encoding="utf-8",
        )

        best_bic_val_file: Path | None = None
        if val_metric_value is not None:
            best_bic_val_file = (
                self._bics_dir() / f"best_bic_val{tag}_{run_idx}{suffix}.json"
            )
            best_bic_val_file.write_text(
                json.dumps(
                    {
                        "mean_BIC": val_metric_value,
                        "mean_NLL": val_mean_nll,
                        "individual_BIC": val_eval_metrics or [],
                        "individual_NLL": val_per_participant_nll or [],
                    },
                    default=_json_default,
                ),
                encoding="utf-8",
            )

        return best_bic_file, best_bic_val_file

    def write_best_model_code(
        self,
        *,
        run_idx: int,
        tag: str,
        code_text: str,
        participant: str | None = None,
    ) -> Path:
        """Persist the current best model code as a plain text artefact."""

        suffix = f"_participant{participant}" if participant else ""
        best_model_file = self._model_dir() / f"best_model{tag}_{run_idx}{suffix}.txt"
        best_model_file.write_text(code_text, encoding="utf-8")
        return best_model_file

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

    def write_review(self, review: dict[str, Any], *, iteration: int, tag: str = "") -> Path | None:
        """Persist a review payload for inspection."""

        if not self.inspection_output_enabled:
            return None

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
        """Persist iteration results to DuckDB and optional inspection JSON.

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

        if self.inspection_output_enabled:
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
