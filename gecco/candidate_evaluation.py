"""Candidate evaluation service extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from gecco.artifacts import ArtifactStore
from gecco.utils import TimestampedConsole


console = TimestampedConsole()


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
        update_registry: Callable[..., None],
        feedback_record: Callable[[int, list[dict[str, Any]]], None],
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

        feedback_record(iteration, iteration_results)

        completion_status = "complete" if had_runnable_model else "complete_no_success"
        update_registry(
            iteration,
            iteration_results,
            status=completion_status,
            had_runnable_model=had_runnable_model,
        )

        return had_runnable_model
