"""Candidate evaluation service extraction."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Callable

from gecco.artifacts import ArtifactStore
from gecco.candidate_generation import CandidateGenerationResult
from gecco.utils import TimestampedConsole


console = TimestampedConsole()


@dataclass(slots=True)
class CandidateEvaluationResult:
    """Structured result from an evaluation pass."""

    iteration_results: list[dict[str, Any]]
    model_file: Path


@dataclass(slots=True)
class BestModelState:
    """Mutable best-model state shared between the orchestrator and evaluator."""

    best_metric: float = float("inf")
    best_model: str | None = None
    best_params: list[Any] = field(default_factory=list)
    best_iter: int = -1
    best_param_names: list[str] = field(default_factory=list)
    best_param_values: Any | None = None
    best_id_results: Any | None = None


@dataclass(slots=True)
class NonCMGEvaluationResult:
    """Structured result from a non-CMG iteration attempt."""

    iteration_results: list[dict[str, Any]]
    had_runnable_model: bool
    should_retry: bool
    retry_feedback: str | None = None


class CandidateEvaluator:
    """Evaluate and, when needed, repair a generated candidate."""

    def __init__(self, artifact_store: ArtifactStore):
        self.artifact_store = artifact_store

    def _is_repairable_error(self, result: dict | None) -> bool:
        """Return True if a CMG candidate result should trigger the repair loop."""
        if result is None:
            return False
        metric_name = result.get("metric_name")
        if metric_name in ("VALIDATION_ERROR", "FIT_ERROR"):
            return True
        if metric_name == "RECOVERY_FAILED":
            return bool(result.get("simulation_error")) and result.get("recovery_n_successful", 0) == 0
        return False

    def _update_best_state(
        self,
        *,
        best_state: BestModelState | None,
        result: dict[str, Any],
        iteration: int,
        run_idx: int,
        tag: str,
        participant: str | None = None,
    ) -> bool:
        """Update the best-model state and persist inspection artefacts when improved."""

        if best_state is None:
            return False

        mean_metric = float(result.get("metric_value", float("inf")))
        if mean_metric >= best_state.best_metric:
            return False

        best_state.best_metric = mean_metric
        best_state.best_model = result.get("code")
        best_state.best_iter = iteration
        best_state.best_params = list(result.get("param_names", []))
        best_state.best_param_names = list(result.get("param_names", []))
        best_state.best_param_values = result.get("parameter_values")
        best_state.best_id_results = result.get("individual_differences")

        self.artifact_store.write_best_model_code(
            run_idx=run_idx,
            tag=tag,
            code_text=result.get("code", ""),
            participant=participant,
        )
        self.artifact_store.write_best_metric_inspection(
            run_idx=run_idx,
            tag=tag,
            metric_value=mean_metric,
            val_metric_value=result.get("val_metric_value"),
            val_mean_nll=result.get("val_mean_nll"),
            val_eval_metrics=result.get("val_eval_metrics"),
            val_per_participant_nll=result.get("val_per_participant_nll"),
            participant=participant,
        )
        return True

    def _publish_registry_status(
        self,
        *,
        shared_registry: Any,
        client_id: Any,
        iteration: int,
        results: list[dict[str, Any]],
        status: str,
        had_runnable_model: bool | None,
        best_state: BestModelState | None,
        tried_param_sets: list[list[Any]] | None,
    ) -> None:
        """Publish evaluator-owned registry state after persistence succeeds."""

        if shared_registry is None:
            return

        shared_registry.update(
            client_id=client_id,
            iteration=iteration,
            results=results,
            best_model=best_state.best_model if best_state is not None else None,
            best_metric=best_state.best_metric if best_state is not None else None,
            param_names=list(best_state.best_params) if best_state is not None else None,
            tried_param_sets=tried_param_sets or [],
            status=status,
            had_runnable_model=had_runnable_model,
        )

    def _repair_candidate(
        self,
        *,
        candidate: dict,
        current_model_dict: dict,
        error_result: dict,
        expected_func_name: str,
        iteration: int,
        candidate_index: int,
        cfg: Any,
        model: Any,
        tokenizer: Any,
        generate_text: Callable[..., str],
        prompt_builder: Any,
        shared_registry: Any,
    ) -> dict:
        """Repair one assigned candidate using the generation backend.

        Builds a repair prompt with the failing code, error details, and
        full task context, then generates exactly one repaired model.
        """
        from gecco.structured_output import parse_model_response, get_model_schema

        error_feedback = self._build_syntax_error_feedback([error_result])
        current_code = current_model_dict.get("code", "")
        candidate_name = current_model_dict.get("name", expected_func_name)
        candidate_params = current_model_dict.get("parameters", [])
        candidate_rationale = candidate.get("rationale", "")

        repair_section = (
            f"The assigned candidate model failed validation or fitting.\n\n"
            f"You must repair this exact candidate. Do not propose a new model idea.\n\n"
            f"Assigned function name: `{expected_func_name}`\n"
            f"Candidate name: {candidate_name}\n"
            f"Candidate rationale: {candidate_rationale}\n"
            f"Candidate parameters: {candidate_params}\n\n"
            f"Current code:\n"
            f"```python\n{current_code}\n```\n\n"
            f"Error:\n{error_feedback}\n\n"
            f"Requirements:\n"
            f"- Return exactly one repaired model.\n"
            f"- The repaired code must define `{expected_func_name}` exactly.\n"
            f"- Keep the same conceptual mechanism unless a small change is necessary "
            f"to make it runnable.\n"
            f"- Keep parameter declarations consistent with the repaired code.\n"
        )

        new_models = None
        try:
            prompt = prompt_builder.build_input_prompt(
                feedback_text=repair_section,
                n_models=1,
                force_include_feedback=True,
            )

            correction_schema = get_model_schema(1, include_analysis=False)
            structured = getattr(cfg.llm, "structured_output", True)
            correction_text = generate_text(
                model,
                tokenizer,
                prompt,
                response_schema=correction_schema if structured else None,
            )
            corrected, _ = parse_model_response(correction_text, 1, structured_output=structured)
            if corrected:
                new_models = corrected
        except Exception as exc:
            console.print(f"  [yellow]CMG repair generation failed: {exc}[/]")
            new_models = None

        if not new_models:
            console.print(
                "[yellow]CMG repair produced no models — keeping original code[/]"
            )
            return current_model_dict

        repaired = new_models[0]
        repaired["func_name"] = expected_func_name
        repaired["name"] = repaired.get(
            "name", current_model_dict.get("name", expected_func_name)
        )
        repaired_code = repaired.get("code", "")

        # Structural validation — ensure repaired code actually defines the function
        if not self._validate_repaired_func_name(
            repaired_code,
            expected_func_name,
            cfg=cfg,
            structured_params=repaired.get(
                "parameters", current_model_dict.get("parameters", [])
            ),
        ):
            return current_model_dict

        updated_candidate = dict(candidate)
        updated_candidate.update({
            "code": repaired_code,
            "name": repaired.get("name", updated_candidate.get("name", expected_func_name)),
            "parameters": repaired.get("parameters", updated_candidate.get("parameters", [])),
            "func_name": expected_func_name,
        })
        shared_registry.update_candidate_model(
            iteration, candidate_index, updated_candidate
        )

        console.print(f"[green]CMG evaluator {candidate_index}: repaired candidate code[/]")

        return {
            "func_name": expected_func_name,
            "name": updated_candidate["name"],
            "code": repaired_code,
            "parameters": updated_candidate.get("parameters", []),
        }

    def _validate_repaired_func_name(
        self,
        repaired_code: str,
        expected_func_name: str,
        *,
        cfg: Any | None = None,
        structured_params: list[dict[str, Any]] | None = None,
        base_class_code: str | None = None,
    ) -> bool:
        """Check that repaired code structurally defines the expected function."""
        import ast

        try:
            tree = ast.parse(repaired_code)
            found = any(
                isinstance(node, ast.FunctionDef) and node.name == expected_func_name
                for node in ast.walk(tree)
            ) or any(
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == expected_func_name
                    for target in node.targets
                )
                for node in ast.walk(tree)
            )
            if not found:
                console.print(
                    f"[yellow]Repaired code does not define function "
                    f"`{expected_func_name}`[/]"
                )
                return False
        except SyntaxError as exc:
            console.print(
                f"[yellow]Repaired code has a syntax error: {exc}[/]"
            )
            return False

        try:
            from gecco.offline_evaluation.utils import build_model_spec
            build_model_spec(
                repaired_code,
                expected_func_name=expected_func_name,
                cfg=cfg,
                base_class_code=base_class_code,
                structured_params=structured_params or [],
            )
            return True
        except Exception as exc:
            console.print(
                f"[yellow]Repaired code failed structural validation: {exc}[/]"
            )
            return False

    def _build_syntax_error_feedback(self, iteration_results: list[dict]) -> str:
        """Build feedback text from syntax/validation errors for regeneration."""
        error_messages = []
        for i, result in enumerate(iteration_results):
            model_name = result.get("function_name", f"model_{i}")
            error_type = result.get("metric_name", "ERROR")

            if error_type == "VALIDATION_ERROR":
                msg = result.get("error_message", "Unknown validation error")
                error_messages.append(f"- {model_name}: {msg}")
            elif error_type == "FIT_ERROR":
                error_msg = result.get("error", "Unknown fit error")
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                error_messages.append(f"- {model_name}: {error_msg}")
            elif error_type == "RECOVERY_FAILED":
                sim_err = result.get("simulation_error")
                if sim_err:
                    error_messages.append(
                        f"- {model_name}: parameter recovery simulation failed: {sim_err}. "
                        f"The model must always return a finite numeric negative log-likelihood, "
                        f"including when called on short prefix trial arrays during simulation."
                    )
                else:
                    error_messages.append(
                        f"- {model_name}: parameter recovery failed."
                    )

        if not error_messages:
            return "All models failed validation. Please review and fix syntax errors."

        feedback = "The following models failed syntax/validation:\n"
        feedback += "\n".join(error_messages)
        feedback += (
            "\n\nPlease regenerate the models with correct Python syntax. "
            "Ensure all functions are properly defined, parentheses match, "
            "and all required imports are handled."
        )
        return feedback

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
        df: Any,
        cfg: Any,
        recovery_checker: Any | None = None,
        id_eval_data: Any | None = None,
        ppc_enabled: bool = False,
        ppc_simulator: Any | None = None,
        ppc_n_sims: int = 100,
        block_residuals_enabled: bool = False,
        block_residuals_n_blocks: int = 10,
        df_val: Any | None = None,
        set_activity: Callable[[str], None] | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        generate_text: Callable[..., str] | None = None,
        prompt_builder: Any | None = None,
        max_syntax_retries: int,
        barrier_timeout_seconds: int,
        participant: str | None = None,
        best_state: BestModelState | None = None,
        tried_param_sets: list[list[Any]] | None = None,
    ) -> CandidateEvaluationResult:
        """Evaluate the candidate assigned to this client.

        Args:
            iteration: Iteration index.
            run_idx: Run index.
            cmg_cfg: CMG configuration.
            tag: File-tag suffix for persisted artefacts.
            client_id: Active client identifier.
            evaluator_index: Evaluator-owned candidate index.
            baseline_bic: Optional early-stop threshold.
            shared_registry: Shared registry collaborator.
            df: Training data frame.
            cfg: Runtime configuration.
            recovery_checker: Optional recovery helper.
            id_eval_data: Optional individual-differences data.
            ppc_enabled: Whether PPC diagnostics are enabled.
            ppc_simulator: Optional PPC simulator.
            ppc_n_sims: Number of PPC simulations.
            block_residuals_enabled: Whether block residuals are enabled.
            block_residuals_n_blocks: Number of residual blocks.
            df_val: Optional validation data frame.
            set_activity: Optional activity callback.
            model: LLM model handle.
            tokenizer: LLM tokenizer handle.
            generate_text: Low-level text-generation backend.
            prompt_builder: Prompt builder collaborator.
            max_syntax_retries: Maximum repair attempts.
            barrier_timeout_seconds: Candidate wait timeout in seconds.
            participant: Optional participant identifier.
            best_state: Shared best-model state.
            tried_param_sets: Mutable list tracking successful parameter sets.

        Returns:
            Structured evaluation results for the assigned candidate.
        """

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
            result, should_stop = self.fit_candidate_model(
                model_dict=current_model_dict,
                model_idx=idx,
                n_models=cmg_cfg.n_models,
                it=iteration,
                run_idx=run_idx,
                tag=tag,
                model_file=model_file,
                baseline_bic=baseline_bic,
                df=df,
                cfg=cfg,
                recovery_checker=recovery_checker,
                id_eval_data=id_eval_data,
                ppc_enabled=ppc_enabled,
                ppc_simulator=ppc_simulator,
                ppc_n_sims=ppc_n_sims,
                block_residuals_enabled=block_residuals_enabled,
                block_residuals_n_blocks=block_residuals_n_blocks,
                df_val=df_val,
                set_activity=set_activity,
                participant=participant,
                tried_param_sets=tried_param_sets,
            )

            repairable_error = self._is_repairable_error(result)

            if not repairable_error or syntax_retry_count >= max_syntax_retries:
                if result is not None:
                    iteration_results = [result]
                break

            syntax_retry_count += 1
            self._publish_registry_status(
                shared_registry=shared_registry,
                client_id=client_id,
                iteration=iteration,
                results=[],
                status="retrying",
                had_runnable_model=False,
                best_state=best_state,
                tried_param_sets=tried_param_sets,
            )

            current_model_dict = self._repair_candidate(
                candidate=candidate,
                current_model_dict=current_model_dict,
                error_result=result,
                expected_func_name=func_name,
                iteration=iteration,
                candidate_index=idx,
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                generate_text=generate_text,
                prompt_builder=prompt_builder,
                shared_registry=shared_registry,
            )
            model_file.write_text(current_model_dict.get("code", ""), encoding="utf-8")

        if result is not None:
            result.setdefault("candidate_index", idx)
            result.setdefault("expected_func_name", func_name)
            result.setdefault("display_name", display_name)

        if result is not None:
            self._update_best_state(
                best_state=best_state,
                result=result,
                iteration=iteration,
                run_idx=run_idx,
                tag=tag,
                participant=participant,
            )

        self.finalize_iteration_results(
            iteration=iteration,
            run_idx=run_idx,
            tag=tag,
            iteration_results=iteration_results,
            client_id=client_id,
            results_source=df,
            shared_registry=shared_registry,
            best_state=best_state,
            tried_param_sets=tried_param_sets,
            feedback_record=None,
        )
        return CandidateEvaluationResult(iteration_results=iteration_results, model_file=model_file)

    def run_non_cmg_iteration(
        self,
        *,
        iteration: int,
        run_idx: int,
        tag: str,
        generation_result: CandidateGenerationResult,
        baseline_bic: float | None,
        df: Any,
        cfg: Any,
        shared_registry: Any,
        client_id: Any,
        results_source: Any,
        best_state: BestModelState | None = None,
        recovery_checker: Any | None = None,
        id_eval_data: Any | None = None,
        ppc_enabled: bool = False,
        ppc_simulator: Any | None = None,
        ppc_n_sims: int = 100,
        block_residuals_enabled: bool = False,
        block_residuals_n_blocks: int = 10,
        df_val: Any | None = None,
        set_activity: Callable[[str], None] | None = None,
        max_syntax_retries: int = 0,
        syntax_retry_count: int = 0,
        participant: str | None = None,
        feedback_record: Callable[[int, list[dict[str, Any]]], None] | None = None,
        tried_param_sets: list[list[Any]] | None = None,
    ) -> NonCMGEvaluationResult:
        """Evaluate a non-CMG batch and finalise it when no retry is needed.

        Args:
            iteration: Iteration index.
            run_idx: Run index.
            tag: File-tag suffix for persisted artefacts.
            generation_result: Generated candidate batch.
            baseline_bic: Optional early-stop threshold.
            df: Training data frame.
            cfg: Runtime configuration.
            shared_registry: Shared registry collaborator.
            client_id: Active client identifier.
            results_source: Runtime results source for participant metadata.
            best_state: Shared best-model state.
            recovery_checker: Optional recovery helper.
            id_eval_data: Optional individual-differences data.
            ppc_enabled: Whether PPC diagnostics are enabled.
            ppc_simulator: Optional PPC simulator.
            ppc_n_sims: Number of PPC simulations.
            block_residuals_enabled: Whether block residuals are enabled.
            block_residuals_n_blocks: Number of residual blocks.
            df_val: Optional validation data frame.
            set_activity: Optional activity callback.
            max_syntax_retries: Maximum retry attempts.
            syntax_retry_count: Current retry count.
            participant: Optional participant identifier.
            feedback_record: Optional feedback-history recorder.
            tried_param_sets: Mutable list tracking successful parameter sets.

        Returns:
            Structured outcome for the non-CMG iteration.
        """

        iteration_results: list[dict[str, Any]] = []

        for model_idx, model_dict in enumerate(generation_result.parsed_models):
            result, should_stop = self.fit_candidate_model(
                model_dict=model_dict,
                model_idx=model_idx,
                n_models=len(generation_result.parsed_models),
                it=iteration,
                run_idx=run_idx,
                tag=tag,
                model_file=generation_result.model_file,
                baseline_bic=baseline_bic,
                df=df,
                cfg=cfg,
                recovery_checker=recovery_checker,
                id_eval_data=id_eval_data,
                ppc_enabled=ppc_enabled,
                ppc_simulator=ppc_simulator,
                ppc_n_sims=ppc_n_sims,
                block_residuals_enabled=block_residuals_enabled,
                block_residuals_n_blocks=block_residuals_n_blocks,
                df_val=df_val,
                set_activity=set_activity,
                participant=participant,
                tried_param_sets=tried_param_sets,
            )

            if result is not None:
                iteration_results.append(result)
                self._update_best_state(
                    best_state=best_state,
                    result=result,
                    iteration=iteration,
                    run_idx=run_idx,
                    tag=tag,
                    participant=participant,
                )

            if should_stop:
                break

        all_syntax_errors = (
            all(
                row.get("metric_name") in ("VALIDATION_ERROR", "FIT_ERROR")
                for row in iteration_results
            )
            and iteration_results
        )

        if all_syntax_errors and syntax_retry_count < max_syntax_retries:
            retry_feedback = self._build_syntax_error_feedback(iteration_results)
            self._publish_registry_status(
                shared_registry=shared_registry,
                client_id=client_id,
                iteration=iteration,
                results=[],
                status="retrying",
                had_runnable_model=False,
                best_state=best_state,
                tried_param_sets=tried_param_sets,
            )
            return NonCMGEvaluationResult(
                iteration_results=iteration_results,
                had_runnable_model=False,
                should_retry=True,
                retry_feedback=retry_feedback,
            )

        had_runnable_model = self.finalize_iteration_results(
            iteration=iteration,
            run_idx=run_idx,
            tag=tag,
            iteration_results=iteration_results,
            client_id=client_id,
            results_source=results_source,
            shared_registry=shared_registry,
            best_state=best_state,
            tried_param_sets=tried_param_sets,
            feedback_record=feedback_record,
        )

        return NonCMGEvaluationResult(
            iteration_results=iteration_results,
            had_runnable_model=had_runnable_model,
            should_retry=False,
        )

    def _smoke_test_model_return_value(self, spec: Any, cfg: Any) -> str | None:
        """Check that a model returns a finite numeric value on dummy inputs."""

        import numpy as np

        n = 3
        dummy_arrays = []
        for _ in getattr(cfg.data, "input_columns", []):
            dummy_arrays.append(np.zeros(n, dtype=np.int64))

        params = []
        for p in spec.param_names:
            lb, ub = spec.bounds[p]
            params.append((float(lb) + float(ub)) / 2.0)
        params = np.asarray(params, dtype=float)

        try:
            value = spec.func(*dummy_arrays, params)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

        if value is None:
            return "Model returned None instead of a numeric negative log-likelihood."

        try:
            value = float(value)
        except Exception:
            return f"Model returned non-numeric value of type {type(value).__name__}."

        if not np.isfinite(value):
            return f"Model returned non-finite value: {value}."

        return None

    def fit_candidate_model(
        self,
        *,
        model_dict: dict,
        model_idx: int,
        n_models: int,
        it: int,
        run_idx: int,
        tag: str,
        model_file: Path,
        baseline_bic: float | None,
        df: Any,
        cfg: Any,
        recovery_checker: Any | None = None,
        id_eval_data: Any | None = None,
        ppc_enabled: bool = False,
        ppc_simulator: Any | None = None,
        ppc_n_sims: int = 100,
        block_residuals_enabled: bool = False,
        block_residuals_n_blocks: int = 10,
        df_val: Any | None = None,
        set_activity: Callable[[str], None] | None = None,
        participant: str | None = None,
        tried_param_sets: list[list[Any]] | None = None,
    ) -> tuple[dict | None, bool]:
        """Fit one candidate model with explicit collaborators.

        Args:
            model_dict: Candidate model payload.
            model_idx: Zero-based index of the candidate.
            n_models: Number of candidates in the iteration.
            it: Iteration index.
            run_idx: Run index.
            tag: File tag used for output naming.
            model_file: Path to the persisted model source.
            baseline_bic: Optional baseline threshold for early stopping.
            df: Training data frame.
            cfg: Runtime configuration.
            recovery_checker: Optional parameter-recovery helper.
            id_eval_data: Optional individual-differences data.
            ppc_enabled: Whether PPC should run.
            ppc_simulator: Optional PPC simulator.
            ppc_n_sims: Number of PPC simulations.
            block_residuals_enabled: Whether block residuals should run.
            block_residuals_n_blocks: Number of residual blocks.
            df_val: Optional validation frame.
            set_activity: Optional status callback.
            participant: Optional participant identifier.
            tried_param_sets: Mutable list tracking successful parameter sets.

        Returns:
            A ``(result_dict, should_stop)`` pair.
        """

        from gecco.offline_evaluation.exceptions import ModelValidationError
        from gecco.offline_evaluation.fit_generated_models import (
            run_fit as run_fit_model,
        )

        func_name = model_dict.get("func_name", f"cognitive_model{model_idx + 1}")
        display_name = model_dict.get("name", func_name)
        func_code = model_dict.get("code", "")
        structured_params = model_dict.get("parameters")
        recovery = None

        if not func_code:
            return {
                "function_name": display_name,
                "metric_name": "VALIDATION_ERROR",
                "metric_value": float("inf"),
                "param_names": [],
                "code": func_code,
                "error_type": "empty_code",
                "error_message": f"No code provided for {func_name}",
                "error_details": {"expected_func_name": func_name},
            }, False

        try:
            if recovery_checker is not None:
                if set_activity is not None:
                    set_activity(
                        f"parameter recovery {model_idx + 1}/{n_models}: {display_name} (iter {it})"
                    )

                from gecco.offline_evaluation.utils import build_model_spec

                try:
                    spec = build_model_spec(
                        func_code,
                        expected_func_name=func_name,
                        cfg=cfg,
                        structured_params=structured_params,
                    )
                    smoke_error = self._smoke_test_model_return_value(spec, cfg)
                    if smoke_error:
                        return {
                            "function_name": display_name,
                            "metric_name": "FIT_ERROR",
                            "metric_value": float("inf"),
                            "param_names": spec.param_names,
                            "code": func_code,
                            "error": smoke_error,
                        }, False
                    console.print(
                        f"  [dim]Running parameter recovery check for {display_name} "
                        f"({recovery_checker.n_subjects} subjects, "
                        f"{recovery_checker.n_trials} trials)...[/]"
                    )
                    recovery = recovery_checker.check(spec)
                    if not recovery["passed"]:
                        sim_err = recovery.get("simulation_error")
                        if sim_err and recovery["n_successful"] == 0:
                            console.print(
                                f"  [yellow]{display_name} failed parameter recovery "
                                f"— simulation error: {sim_err}[/]"
                            )
                        else:
                            console.print(
                                f"  [yellow]{display_name} failed parameter recovery "
                                f"(mean r={recovery['mean_r']:.2f}, "
                                f"threshold={recovery_checker.threshold})[/]"
                            )
                        result = {
                            "function_name": display_name,
                            "metric_name": "RECOVERY_FAILED",
                            "metric_value": float("inf"),
                            "param_names": spec.param_names,
                            "code": func_code,
                            "recovery_r": recovery["mean_r"],
                            "recovery_per_param": recovery["per_param_r"],
                            "recovery_n_successful": recovery["n_successful"],
                            "simulation_error": sim_err,
                        }
                        from gecco.sentry_init import capture_recovery_failed

                        capture_recovery_failed(
                            iteration=it,
                            model_name=display_name,
                            error=Exception(
                                f"Parameter recovery failed (mean r={recovery['mean_r']:.2f})"
                            ),
                        )
                        return result, False
                except ModelValidationError as e:
                    console.print(
                        f"  [yellow]{display_name} validation error ({e.error_type}): {e.message}[/]"
                    )
                    if e.details:
                        for key, value in e.details.items():
                            console.print(f"    [dim]{key}: {value}[/]")
                    safe_details: dict[str, Any] = {}
                    for key, value in e.details.items():
                        try:
                            import json

                            json.dumps(value)
                            safe_details[key] = value
                        except (TypeError, ValueError):
                            safe_details[key] = str(value)
                    return {
                        "function_name": display_name,
                        "metric_name": "VALIDATION_ERROR",
                        "metric_value": float("inf"),
                        "param_names": [],
                        "code": func_code,
                        "error_type": e.error_type,
                        "error_message": e.message,
                        "error_details": safe_details,
                    }, False
                except Exception as exc:
                    console.print(
                        f"  [yellow]{display_name} recovery check error: {exc}[/]"
                    )
                    from gecco.sentry_init import capture_fit_error

                    capture_fit_error(
                        iteration=it,
                        model_name=display_name,
                        error=exc,
                        run=run_idx,
                    )
                    return {
                        "function_name": display_name,
                        "metric_name": "FIT_ERROR",
                        "metric_value": float("inf"),
                        "param_names": [],
                        "code": func_code,
                        "error": str(exc),
                    }, False

            if set_activity is not None:
                set_activity(
                    f"fitting model {model_idx + 1}/{n_models}: {display_name} (iter {it})"
                )

            fit_res = run_fit_model(
                df,
                func_code,
                cfg=cfg,
                expected_func_name=func_name,
                structured_params=structured_params,
            )

            mean_metric = float(fit_res["metric_value"])
            metric_name = fit_res["metric_name"]
            params = fit_res["param_names"]
            if tried_param_sets is not None:
                tried_param_sets.append(list(params))

            console.print(
                f"  [bold]{display_name}[/]: mean {metric_name} = [cyan]{mean_metric:.2f}[/]"
            )

            id_results = None
            if id_eval_data is not None:
                try:
                    from gecco.offline_evaluation.individual_differences import (
                        evaluate_individual_differences,
                    )

                    id_results = evaluate_individual_differences(
                        fit_res,
                        df,
                        cfg,
                        id_data=id_eval_data,
                    )
                except Exception as exc:
                    console.print(
                        f"  [yellow]Individual differences eval failed for {display_name}:[/] {exc}"
                    )

            val_fit_res = None
            val_id_results = None
            if df_val is not None:
                try:
                    val_fit_res = run_fit_model(
                        df_val,
                        func_code,
                        cfg=cfg,
                        expected_func_name=func_name,
                        structured_params=structured_params,
                    )
                    console.print(
                        f"    [dim]val {metric_name} = [cyan]{val_fit_res['metric_value']:.2f}[/]"
                    )
                    if id_eval_data is not None:
                        try:
                            from gecco.offline_evaluation.individual_differences import (
                                evaluate_individual_differences,
                            )

                            val_id_results = evaluate_individual_differences(
                                val_fit_res,
                                df_val,
                                cfg,
                                id_data=id_eval_data,
                            )
                        except Exception as exc:
                            console.print(
                                f"  [yellow]Individual differences eval failed for {display_name} on val:[/] {exc}"
                            )
                except Exception as exc:
                    console.print(
                        f"  [yellow]Val fitting failed for {display_name}:[/] {exc}"
                    )

            ppc_result = None
            block_residuals_result = None
            needs_diagnostic_spec = bool(
                fit_res.get("parameter_values")
                and ((ppc_enabled and ppc_simulator is not None) or block_residuals_enabled)
            )
            diagnostic_spec = None
            if needs_diagnostic_spec:
                try:
                    from gecco.offline_evaluation.utils import build_model_spec

                    diagnostic_spec = build_model_spec(
                        func_code,
                        expected_func_name=func_name,
                        cfg=cfg,
                        structured_params=structured_params,
                    )
                except Exception as exc:
                    console.print(
                        f"  [yellow]Diagnostic spec build failed for {display_name}:[/] {exc}"
                    )

            if ppc_enabled and ppc_simulator is not None and fit_res.get("parameter_values") and diagnostic_spec is not None:
                try:
                    from gecco.offline_evaluation.ppc import compute_ppc, _get_participants

                    console.print(
                        f"  [dim]Computing PPC for {display_name} (n_sims={ppc_n_sims})...[/]"
                    )
                    _, participants = _get_participants(df)
                    n_participants = len(participants)

                    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

                    ppc_progress = Progress(
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        MofNCompleteColumn(),
                        TimeElapsedColumn(),
                    )
                    with ppc_progress:
                        task_id = ppc_progress.add_task(
                            f"  [dim]PPC {display_name}[/]",
                            total=n_participants,
                        )
                        ppc_result = compute_ppc(
                            spec=diagnostic_spec,
                            df=df,
                            fitted_params_list=fit_res["parameter_values"],
                            simulator=ppc_simulator,
                            n_sims=ppc_n_sims,
                            input_columns=list(cfg.data.input_columns),
                            n_jobs=-1,
                            progress_callback=lambda: ppc_progress.advance(task_id),
                        )
                except Exception as exc:
                    console.print(f"  [yellow]PPC failed for {display_name}:[/] {exc}")

            if block_residuals_enabled and fit_res.get("parameter_values") and diagnostic_spec is not None:
                try:
                    from gecco.offline_evaluation.ppc import compute_block_residuals

                    console.print(
                        f"  [dim]Computing block residuals for {display_name} (n_blocks={block_residuals_n_blocks})...[/]"
                    )
                    block_residuals_result = compute_block_residuals(
                        spec=diagnostic_spec,
                        df=df,
                        fitted_params_list=fit_res["parameter_values"],
                        n_blocks=block_residuals_n_blocks,
                        input_columns=list(cfg.data.input_columns),
                    )
                except Exception as exc:
                    console.print(f"  [yellow]Block residuals failed for {display_name}:[/] {exc}")

            result_dict = {
                "function_name": display_name,
                "metric_name": metric_name,
                "metric_value": mean_metric,
                "param_names": params,
                "code_file": str(model_file),
                "recovery": recovery if recovery_checker is not None else None,
                "individual_differences": id_results,
                "code": func_code,
                "eval_metrics": fit_res.get("eval_metrics", []),
                "participant_n_trials": fit_res.get("participant_n_trials", []),
                "parameter_values": fit_res.get("parameter_values", []),
                "mean_nll": fit_res.get("mean_nll"),
                "per_participant_nll": fit_res.get("per_participant_nll"),
            }
            if val_fit_res is not None:
                result_dict["val_metric_value"] = val_fit_res["metric_value"]
                result_dict["val_mean_nll"] = val_fit_res["mean_nll"]
                result_dict["val_eval_metrics"] = val_fit_res["eval_metrics"]
                result_dict["val_per_participant_nll"] = val_fit_res["per_participant_nll"]
                result_dict["val_individual_differences"] = val_id_results
            if ppc_result is not None:
                result_dict["ppc"] = ppc_result
            if block_residuals_result is not None:
                result_dict["block_residuals"] = block_residuals_result

            should_stop = baseline_bic is not None and mean_metric < baseline_bic
            return result_dict, should_stop

        except ModelValidationError as exc:
            console.print(
                f"  [bold red]Validation error in {display_name}:[/] {exc.message}"
            )
            safe_details = {}
            for key, value in exc.details.items():
                try:
                    import json

                    json.dumps(value)
                    safe_details[key] = value
                except (TypeError, ValueError):
                    safe_details[key] = str(value)
            return {
                "function_name": display_name,
                "metric_name": "VALIDATION_ERROR",
                "metric_value": float("inf"),
                "param_names": [],
                "code": func_code,
                "error_type": exc.error_type,
                "error_message": exc.message,
                "error_details": safe_details,
            }, False
        except Exception as exc:
            console.print(f"  [bold red]Error fitting {display_name}:[/] {exc}")
            from gecco.sentry_init import capture_fit_error

            capture_fit_error(iteration=it, model_name=display_name, error=exc, run=run_idx)
            return {
                "function_name": display_name,
                "metric_name": "FIT_ERROR",
                "metric_value": float("inf"),
                "param_names": [],
                "code": func_code,
                "error": str(exc),
            }, False

    def finalize_iteration_results(
        self,
        *,
        iteration: int,
        run_idx: int,
        tag: str,
        iteration_results: list[dict[str, Any]],
        client_id: Any,
        results_source: Any,
        shared_registry: Any,
        best_state: BestModelState | None = None,
        tried_param_sets: list[list[Any]] | None = None,
        feedback_record: Callable[[int, list[dict[str, Any]]], None] | None = None,
    ) -> bool:
        """Persist and publish results for one completed iteration."""

        had_runnable_model = self.artifact_store.write_iteration_results(
            iteration=iteration,
            run_idx=run_idx,
            tag=tag,
            iteration_results=iteration_results,
            client_id=client_id,
            results_source=results_source,
        )

        if feedback_record is not None:
            feedback_record(iteration, iteration_results)

        completion_status = "complete" if had_runnable_model else "complete_no_success"
        self._publish_registry_status(
            shared_registry=shared_registry,
            client_id=client_id,
            iteration=iteration,
            results=iteration_results,
            status=completion_status,
            had_runnable_model=had_runnable_model,
            best_state=best_state,
            tried_param_sets=tried_param_sets,
        )

        return had_runnable_model
