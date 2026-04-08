# engine/model_search.py
import os
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.table import Table

from gecco.offline_evaluation.fit_generated_models import (
    run_fit_hierarchical as run_fit,
)
from gecco.construct_feedback.feedback import FeedbackGenerator, LLMFeedbackGenerator
from pathlib import Path

console = Console()


class GeCCoModelSearch:
    def __init__(
        self,
        model,
        tokenizer,
        cfg,
        df,
        prompt_builder,
        client_id=None,
        shared_registry=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.df = df
        self.prompt_builder = prompt_builder
        self.client_id = client_id
        self.shared_registry = shared_registry

        # --- Choose feedback generator based on config ---
        if (
            hasattr(cfg, "feedback")
            and getattr(cfg.feedback, "type", "manual") == "llm"
        ):
            self.feedback = LLMFeedbackGenerator(cfg, model, tokenizer)
        else:
            self.feedback = FeedbackGenerator(cfg)

        # --- Set project root ---
        self.project_root = Path(__file__).resolve().parents[1]

        # --- Results directory (absolute path) ---
        fit_type = getattr(self.cfg.evaluation, "fit_type", "group")
        self.results_dir = (
            self.project_root / "results" / self.cfg.task.name
            if fit_type != "individual"
            else self.project_root / "results" / f"{self.cfg.task.name}_individual"
        )

        (self.results_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "bics").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "feedback").mkdir(parents=True, exist_ok=True)

        # --- Individual differences evaluation (optional) ---
        self.id_eval_data = None
        if hasattr(cfg, "individual_differences_eval"):
            from gecco.offline_evaluation.individual_differences import load_id_data

            self.id_eval_data = load_id_data(cfg)

        # --- Tracking ---
        self.best_model = None
        self.best_metric = np.inf
        self.best_params = []
        self.best_iter = -1
        self.best_param_names = []
        self.best_param_values = None
        self.tried_param_sets = []
        self.best_id_results = None

        # --- Track which registry entries we've already merged ---
        self._merged_history_count = 0

        # --- Parameter recovery checker (optional) ---
        self.recovery_checker = None
        if hasattr(cfg, "parameter_recovery") and getattr(
            cfg.parameter_recovery, "enabled", False
        ):
            from gecco.parameter_recovery import ParameterRecoveryChecker, get_simulator

            simulator = get_simulator(cfg.parameter_recovery)
            self.recovery_checker = ParameterRecoveryChecker(
                simulator=simulator,
                n_subjects=getattr(cfg.parameter_recovery, "n_subjects", 50),
                n_trials=getattr(cfg.parameter_recovery, "n_trials", 100),
                threshold=getattr(cfg.parameter_recovery, "threshold", 0.5),
                n_fitting_starts=getattr(cfg.parameter_recovery, "n_fitting_starts", 3),
                n_jobs=getattr(cfg.parameter_recovery, "n_jobs", -1),
            )

    def generate(self, model, tokenizer=None, prompt=None, response_schema=None):
        """
        Unified text generation function for any supported backend.
        Handles both OpenAI GPT and Hugging Face-style models cleanly.

        Parameters
        ----------
        model : object
            The model object (OpenAI client, HF model, etc.)
        tokenizer : object, optional
            Tokenizer for HuggingFace models.
        prompt : str, optional
            The prompt text to send to the model.
        response_schema : dict, optional
            JSON schema for structured output. If None, no schema enforcement
            is applied (free-form text response).

        Returns
        -------
        str
            The generated text response.
        """
        if model is None:
            raise ValueError("Model not initialized correctly.")
        provider = self.cfg.llm.provider.lower()

        # -----------------------------
        # OpenAI / GPT-style generation
        # -----------------------------
        if "openai" in provider or "gpt" in provider:
            max_out = self.cfg.llm.max_output_tokens
            reasoning_effort = getattr(self.cfg.llm, "reasoning_effort", "medium")
            text_verbosity = getattr(self.cfg.llm, "text_verbosity", "low")

            console.print(
                f"[dim]Generating with GPT [cyan]{self.cfg.llm.base_model}[/] "
                f"(reasoning={reasoning_effort}, max_tokens={max_out})[/]"
            )

            create_kwargs = {
                "model": self.cfg.llm.base_model,
                "input": [
                    {"role": "developer", "content": self.cfg.llm.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }

            if reasoning_effort:
                create_kwargs["reasoning"] = {"effort": reasoning_effort}

            # Structured output via JSON schema (only if schema provided)
            if response_schema is not None:
                from gecco.structured_output import get_openai_response_format

                create_kwargs["text"] = {
                    "format": get_openai_response_format(response_schema)
                }

            resp = model.responses.create(**create_kwargs)
            decoded = resp.output_text.strip()

            return decoded

        elif "gemini" in provider:
            from google.genai import types

            reasoning_effort = getattr(self.cfg.llm, "reasoning_effort", "low")

            console.print(
                f"[dim]Generating with Gemini [cyan]{self.cfg.llm.base_model}[/] "
                f"(reasoning={reasoning_effort})[/]"
            )

            config_args = {
                "temperature": self.cfg.llm.temperature,
                "system_instruction": self.cfg.llm.system_prompt,
            }

            use_thinking = False
            if reasoning_effort:
                valid_levels = ["minimal", "low", "medium", "high"]
                assert reasoning_effort in valid_levels, (
                    f"Invalid reasoning_effort: {reasoning_effort}. Choose from {valid_levels}."
                )
                if self.cfg.llm.base_model.lower().startswith("gemini-3"):
                    config_args["thinking_config"] = types.ThinkingConfig(
                        thinking_level=reasoning_effort
                    )
                    use_thinking = True
                elif self.cfg.llm.base_model.lower().startswith("gemini-2"):
                    budget_map = {
                        "minimal": 0,
                        "low": 4096,
                        "medium": 12288,
                        "high": 24576,
                    }
                    config_args["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=budget_map[reasoning_effort]
                    )
                    use_thinking = True

            # Structured output via response schema.
            # Gemini may not support response_schema + thinking_config together,
            # so when thinking is enabled we rely on the prompt-level JSON
            # instructions instead.
            # Skip review schema for Gemini — use prompt-only JSON
            if response_schema is not None and not use_thinking:
                from gecco.structured_output import get_gemini_schema

                is_review = "reviews" in response_schema.get("properties", {})
                if not is_review:
                    config_args["response_mime_type"] = "application/json"
                    config_args["response_schema"] = get_gemini_schema(response_schema)

            resp = model.models.generate_content(
                model=self.cfg.llm.base_model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_args),
            )
            decoded = resp.text.strip()

            return decoded

        # -----------------------------
        # vLLM / KCL / OpenCode / OpenRouter (OpenAI-compatible API)
        # -----------------------------
        elif (
            "vllm" in provider
            or "kcl" in provider
            or "opencode" in provider
            or "openrouter" in provider
        ):
            max_out = getattr(
                self.cfg.llm,
                "max_output_tokens",
                getattr(self.cfg.llm, "max_tokens", 4096),
            )

            if "kcl" in provider:
                provider_label = "KCL"
            elif "opencode" in provider:
                provider_label = "OpenCode Zen"
            elif "openrouter" in provider:
                provider_label = "OpenRouter"
            else:
                provider_label = "vLLM"
            console.print(
                f"[dim]Generating with {provider_label} [cyan]{self.cfg.llm.base_model}[/] "
                f"(max_tokens={max_out}, temp={self.cfg.llm.temperature})[/]"
            )

            create_kwargs = {
                "model": self.cfg.llm.base_model,
                "messages": [
                    {"role": "system", "content": self.cfg.llm.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.cfg.llm.temperature,
                "max_tokens": max_out,
            }

            # Structured output via json_object mode (OpenAI-compatible APIs)
            # Applied when response_schema is provided (any schema triggers JSON mode)
            if response_schema is not None:
                from gecco.structured_output import (
                    get_openai_compatible_response_format,
                )

                create_kwargs["response_format"] = (
                    get_openai_compatible_response_format()
                )

            resp = model.chat.completions.create(**create_kwargs)
            if not hasattr(resp, "choices"):
                raise TypeError(
                    f"Expected a ChatCompletion response but got {type(resp).__name__!r}. "
                    f"Response: {resp!r:.200}"
                )
            content = resp.choices[0].message.content
            if content is None:
                finish = getattr(resp.choices[0], "finish_reason", "unknown")
                console.print(
                    f"[yellow]API returned empty response (finish_reason={finish})[/]"
                )
                return ""
            return content.strip()

        # -----------------------------
        # Hugging Face-style generation
        # -----------------------------
        else:
            from transformers import TextStreamer

            max_new = getattr(
                self.cfg.llm,
                "max_output_tokens",
                getattr(self.cfg.llm, "max_tokens", 4096),
            )
            n_input = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])

            console.print(
                f"[dim]Generating with [cyan]{self.cfg.llm.base_model}[/] "
                f"(input={n_input}, max_new={max_new}, temp={self.cfg.llm.temperature})[/]"
            )

            # Progress bar streamer
            gen_progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total} tokens"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            )
            gen_task = gen_progress.add_task("[green]Generating", total=max_new)

            class _ProgressStreamer(TextStreamer):
                def __init__(self, tokenizer, progress, task_id):
                    super().__init__(
                        tokenizer, skip_prompt=True, skip_special_tokens=True
                    )
                    self.token_count = 0
                    self._progress = progress
                    self._task_id = task_id

                def on_finalized_text(self, text, stream_end=False):
                    n = len(text.split()) if text.strip() else 1
                    self.token_count += n
                    self._progress.update(self._task_id, completed=self.token_count)

            streamer = _ProgressStreamer(tokenizer, gen_progress, gen_task)

            t0 = time.time()
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with gen_progress:
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    temperature=self.cfg.llm.temperature,
                    do_sample=True,
                    streamer=streamer,
                )
            elapsed = time.time() - t0
            n_tokens = output.shape[1] - inputs["input_ids"].shape[1]
            console.print(
                f"[dim]Generated [cyan]{n_tokens}[/] tokens in {elapsed:.1f}s "
                f"({n_tokens / elapsed:.1f} tok/s)[/]"
            )
            return tokenizer.decode(output[0], skip_special_tokens=True)

    def generate_models(self, prompt):
        """
        Generate cognitive models with structured output, parsing, and
        optional review-and-fix cycle.

        Returns
        -------
        tuple of (raw_text, list of model dicts)
            Each model dict has keys: name, rationale, code, analysis.
        """
        from gecco.structured_output import (
            parse_model_response,
            build_review_prompt,
            build_fix_prompt,
            parse_review_response,
            validate_single_model,
            build_correction_prompt,
            get_schema_instructions,
            get_model_schema,
            get_review_schema,
        )

        structured = getattr(self.cfg.llm, "structured_output", True)
        n_models = self.cfg.llm.models_per_iteration
        validation_cfg = getattr(self.cfg, "validation", None)
        max_retries = (
            getattr(validation_cfg, "retry_limit", 3)
            if validation_cfg is not None
            else 3
        )

        # --- Initial generation ---
        include_analysis = getattr(self.cfg.llm, "analysis_scratchpad", True)
        model_schema = get_model_schema(n_models, include_analysis=include_analysis)

        raw_text = self.generate(
            self.model,
            self.tokenizer,
            prompt,
            response_schema=model_schema if structured else None,
        )
        models, json_ok = parse_model_response(
            raw_text, n_models, structured_output=structured
        )

        if not models:
            console.print("[yellow]No models extracted from LLM response[/]")
            return raw_text, []

        # Log analysis scratchpads
        for m in models:
            if m.get("analysis"):
                console.print(
                    f"  [dim]{m['name']} analysis:[/] "
                    f"{m['analysis'][:200]}{'...' if len(m['analysis']) > 200 else ''}"
                )

        # --- Validation correction loop ---
        # Validate each model with Pydantic and retry on failures,
        # BEFORE the review-and-fix cycle to avoid wasting compute on invalid models.
        validated_models = []
        for i, model in enumerate(models):
            validated_model = model
            model_name = model.get("name", f"cognitive_model{i + 1}")
            validation_result = None

            for retry_attempt in range(max_retries):
                validation_result = validate_single_model(validated_model)

                if validation_result.is_valid:
                    console.print(
                        f"  [dim]Model {i + 1} ({model_name}) passed validation[/]"
                    )
                    break

                error_trace = "\n".join(f"  {err}" for err in validation_result.errors)
                console.print(
                    f"  [yellow]Model {i + 1} ({model_name}) failed validation "
                    f"(attempt {retry_attempt + 1}/{max_retries}):[/]\n{error_trace}"
                )

                schema_instructions = get_schema_instructions(1, include_analysis=False)
                correction_prompt = build_correction_prompt(
                    model=validated_model,
                    model_index=i + 1,
                    validation_errors=validation_result.errors,
                    schema_instructions=schema_instructions,
                )

                correction_schema = get_model_schema(1, include_analysis=False)
                correction_text = self.generate(
                    self.model,
                    self.tokenizer,
                    correction_prompt,
                    response_schema=correction_schema if structured else None,
                )
                corrected, _ = parse_model_response(
                    correction_text, 1, structured_output=structured
                )

                if corrected:
                    validated_model = corrected[0]
                else:
                    console.print("  [yellow]Failed to parse correction attempt[/]")
                    break

            if validation_result is not None and validation_result.is_valid:
                validated_models.append(validated_model)
            else:
                console.print(
                    f"  [bold red]Model {i + 1} ({model_name}) failed validation "
                    f"after {max_retries} retries — skipping[/]"
                )
                validated_models.append(
                    {
                        "name": model_name,
                        "rationale": model.get("rationale", ""),
                        "code": model.get("code", ""),
                        "analysis": model.get("analysis", ""),
                        "validation_failed": True,
                        "validation_errors": validation_result.errors
                        if validation_result
                        else [],
                    }
                )

        models = validated_models
        if not models:
            console.print(
                "[yellow]All models failed validation — no models to process[/]"
            )
            return raw_text, []

        # --- Review-and-Fix cycle (optional) ---
        reviewer_config = getattr(self.cfg.llm, "reviewer", None)
        if reviewer_config and getattr(reviewer_config, "enabled", False) and models:
            guardrails = getattr(self.cfg.llm, "guardrails", [])
            persona = getattr(reviewer_config, "persona", None)
            focus_areas = getattr(reviewer_config, "focus_areas", None)

            # --- Review phase ---
            console.print("[dim]Running code review...[/]")
            review_prompt = build_review_prompt(
                models,
                guardrails=guardrails,
                persona=persona,
                focus_areas=focus_areas,
            )
            review_schema = get_review_schema()
            review_text = self.generate(
                self.model,
                self.tokenizer,
                review_prompt,
                response_schema=review_schema if structured else None,
            )
            review = parse_review_response(review_text)

            # --- Save review to disk ---
            self._save_review(review)

            # --- Check if any issues found ---
            total_issues = sum(
                len(r.get("issues", [])) for r in review.get("reviews", [])
            )

            if total_issues > 0:
                console.print(
                    f"[dim]Review found {total_issues} issue(s) across "
                    f"{sum(1 for r in review.get('reviews', []) if r.get('issues'))} model(s)[/]"
                )

                # --- Fix phase ---
                fix_prompt = build_fix_prompt(models, review, guardrails=guardrails)
                if fix_prompt:
                    console.print("[dim]Requesting fixes...[/]")
                    fix_text = self.generate(
                        self.model,
                        self.tokenizer,
                        fix_prompt,
                        response_schema=model_schema if structured else None,
                    )
                    fixed_models, fixed_json_ok = parse_model_response(
                        fix_text, n_models, structured_output=structured
                    )

                    if fixed_models and len(fixed_models) == len(models):
                        # Preserve names/analysis from originals, use fixed code
                        for orig, fixed in zip(models, fixed_models):
                            orig["code"] = fixed["code"]
                            if fixed.get("rationale"):
                                orig["rationale"] = fixed["rationale"]
                        console.print(
                            f"[dim]Applied fixes to {len(models)} model(s)[/]"
                        )
                    else:
                        console.print(
                            "[yellow]Fix parsing failed — using original models[/]"
                        )
            else:
                # Check if any models have non-passing assessments
                non_passing = sum(
                    1
                    for r in review.get("reviews", [])
                    if r.get("overall_assessment", "passes") != "passes"
                )
                if non_passing == 0:
                    console.print("[dim]Review passed — no issues found[/]")
                else:
                    console.print(
                        f"[dim]Review found {non_passing} model(s) with issues (no fix attempted)[/]"
                    )

        return raw_text, models

    def _save_review(self, review: dict):
        """
        Save review comments to disk for debugging.

        Parameters
        ----------
        review : dict
            Structured review from parse_review_response().
        """
        review_dir = self.results_dir / "reviews"
        review_dir.mkdir(parents=True, exist_ok=True)

        # Use iteration count for filename
        existing = list(review_dir.glob("iter*.json"))
        iteration = len(existing) + 1
        review_file = review_dir / f"iter{iteration}{self._file_tag()}.json"

        with open(review_file, "w") as f:
            json.dump(review, f, indent=2)

    def _file_tag(self):
        """Return a client tag for filenames, or empty string if not distributed."""
        if self.client_id is not None:
            return f"_client{self.client_id}"
        return ""

    def _sync_from_registry(self):
        """
        Pull cross-client data from the shared registry into local state.
        Updates best model, tried param sets, and merges history into
        self.feedback.history so all feedback analysis methods see
        cross-client data.
        """
        if self.shared_registry is None:
            return

        data = self.shared_registry.read()

        # Update best model if global best is better
        global_best = data.get("global_best")
        if global_best and global_best["metric_value"] < self.best_metric:
            self.best_metric = global_best["metric_value"]
            self.best_model = global_best["model_code"]
            self.best_params = global_best["param_names"]
            console.print(
                f"  [bold magenta]Synced global best from client {global_best['client_id']}:[/] "
                f"BIC = {global_best['metric_value']:.2f}"
            )

        # Merge tried param sets (deduplicate)
        existing = {tuple(s) for s in self.tried_param_sets}
        for ps in data.get("tried_param_sets", []):
            key = tuple(ps)
            if key not in existing:
                self.tried_param_sets.append(ps)
                existing.add(key)

        # Merge cross-client iteration history into feedback.history
        all_history = data.get("iteration_history", [])
        new_entries = all_history[self._merged_history_count :]

        for entry in new_entries:
            # Skip our own entries (already in feedback.history)
            if entry.get("client_id") == self.client_id:
                continue

            self.feedback.history.append(
                {
                    "iteration": entry["iteration"],
                    "results": entry["results"],
                    "client_id": entry.get("client_id"),
                }
            )

        self._merged_history_count = len(all_history)

    def _set_activity(self, activity):
        """Update current activity in the shared registry."""
        if self.shared_registry is None:
            return
        self.shared_registry.set_activity(self.client_id, activity)

    def _update_registry(self, iteration, results):
        """Push this iteration's results to the shared registry."""
        if self.shared_registry is None:
            return

        self.shared_registry.update(
            client_id=self.client_id,
            iteration=iteration,
            results=results,
            best_model=self.best_model,
            best_metric=self.best_metric,
            param_names=self.best_params,
            tried_param_sets=self.tried_param_sets,
        )

    def run_n_shots(self, run_idx, baseline_bic):
        # Resume from the next iteration after what's already in the registry
        start_iter = 0
        if self.shared_registry is not None:
            max_existing = self.shared_registry.get_max_iteration()
            if max_existing >= 0:
                start_iter = max_existing + 1
                console.print(
                    f"[dim]Resuming from iteration {start_iter} (registry has up to {max_existing})[/]"
                )

        end_iter = start_iter + self.cfg.loop.max_iterations
        for it in range(start_iter, end_iter):
            console.rule(f"[bold]Iteration {it}")

            stop_iterations = False  # ✅ reset each iteration

            # --- Sync from shared registry (distributed mode) ---
            self._sync_from_registry()

            feedback = ""
            if self.best_model is not None:
                feedback = self.feedback.get_feedback(
                    self.best_model,
                    self.tried_param_sets,
                    id_results=self.best_id_results,
                )

                # Save feedback for inspection
                tag = self._file_tag()
                feedback_file = (
                    self.results_dir / "feedback" / f"iter{it}{tag}_run{run_idx}.txt"
                    if getattr(self.cfg.evaluation, "fit_type", "group") != "individual"
                    else self.results_dir
                    / "feedback"
                    / f"iter{it}{tag}_run{run_idx}_participant{self.df.participant[0]}.txt"
                )
                with open(feedback_file, "w") as f:
                    f.write(feedback)

            self._set_activity(f"generating models (iter {it})")
            prompt = self.prompt_builder.build_input_prompt(feedback_text=feedback)
            code_text, parsed_models = self.generate_models(prompt)

            tag = self._file_tag()
            model_file = (
                self.results_dir / "models" / f"iter{it}{tag}_run{run_idx}.txt"
                if getattr(self.cfg.evaluation, "fit_type", "group") != "individual"
                else self.results_dir
                / "models"
                / f"iter{it}{tag}_run{run_idx}_participant{self.df.participant[0]}.txt"
            )

            with open(model_file, "w") as f:
                f.write(code_text)

            # Save structured output (analysis, names, rationale) for inspection
            if parsed_models:
                structured_file = model_file.with_suffix(".json")
                with open(structured_file, "w") as f:
                    json.dump(
                        [
                            {
                                "name": m["name"],
                                "rationale": m.get("rationale", ""),
                                "analysis": m.get("analysis", ""),
                                "parameters": m.get("parameters", []),
                            }
                            for m in parsed_models
                        ],
                        f,
                        indent=2,
                    )

            iteration_results = []

            n_models = len(parsed_models)
            for i, model_dict in enumerate(parsed_models):
                func_name = f"cognitive_model{i + 1}"
                display_name = model_dict.get("name", func_name)
                func_code = model_dict["code"]
                structured_params = model_dict.get("parameters")

                if not func_code:
                    continue

                try:
                    from gecco.offline_evaluation.exceptions import ModelValidationError  # noqa: F811

                    # --- Parameter recovery check (optional) ---
                    if self.recovery_checker is not None:
                        self._set_activity(
                            f"parameter recovery {i + 1}/{n_models}: {display_name} (iter {it})"
                        )
                        from gecco.offline_evaluation.utils import build_model_spec
                        from gecco.offline_evaluation.exceptions import (
                            ModelValidationError,
                        )

                        try:
                            spec = build_model_spec(
                                func_code,
                                expected_func_name=func_name,
                                cfg=self.cfg,
                                structured_params=structured_params,
                            )
                            console.print(
                                f"  [dim]Running parameter recovery check for {display_name} "
                                f"({self.recovery_checker.n_subjects} subjects, "
                                f"{self.recovery_checker.n_trials} trials)...[/]"
                            )
                            recovery = self.recovery_checker.check(spec)
                            if not recovery["passed"]:
                                console.print(
                                    f"  [yellow]{display_name} failed parameter recovery "
                                    f"(mean r={recovery['mean_r']:.2f}, "
                                    f"threshold={self.recovery_checker.threshold})[/]"
                                )
                                iteration_results.append(
                                    {
                                        "function_name": display_name,
                                        "metric_name": "RECOVERY_FAILED",
                                        "metric_value": float("inf"),
                                        "param_names": spec.param_names,
                                        "code": func_code,
                                        "recovery_r": recovery["mean_r"],
                                        "recovery_per_param": recovery["per_param_r"],
                                    }
                                )
                                continue
                        except ModelValidationError as e:
                            console.print(
                                f"  [yellow]{display_name} validation error ({e.error_type}): {e.message}[/]"
                            )
                            iteration_results.append(
                                {
                                    "function_name": display_name,
                                    "metric_name": "VALIDATION_ERROR",
                                    "metric_value": float("inf"),
                                    "param_names": [],
                                    "code": func_code,
                                    "error_type": e.error_type,
                                    "error_message": e.message,
                                    "error_details": e.details,
                                }
                            )
                            continue
                        except Exception as e:
                            console.print(
                                f"  [yellow]{display_name} recovery check error: {e}[/]"
                            )
                            iteration_results.append(
                                {
                                    "function_name": display_name,
                                    "metric_name": "FIT_ERROR",
                                    "metric_value": float("inf"),
                                    "param_names": [],
                                    "code": func_code,
                                    "error": str(e),
                                }
                            )
                            continue

                    self._set_activity(
                        f"fitting model {i + 1}/{n_models}: {display_name} (iter {it})"
                    )
                    fit_res = run_fit(
                        self.df,
                        func_code,
                        cfg=self.cfg,
                        expected_func_name=func_name,
                        structured_params=structured_params,
                    )

                    mean_metric = float(fit_res["metric_value"])
                    metric_name = fit_res["metric_name"]
                    params = fit_res["param_names"]
                    self.tried_param_sets.append(params)

                    console.print(
                        f"  [bold]{display_name}[/]: mean {metric_name} = [cyan]{mean_metric:.2f}[/]"
                    )

                    # --- Individual differences evaluation (optional) ---
                    id_results = None
                    if self.id_eval_data is not None:
                        try:
                            from gecco.offline_evaluation.individual_differences import (
                                evaluate_individual_differences,
                            )

                            id_results = evaluate_individual_differences(
                                fit_res, self.df, self.cfg, id_data=self.id_eval_data
                            )
                        except Exception as e:
                            console.print(
                                f"  [yellow]Individual differences eval failed for {display_name}:[/] {e}"
                            )

                    iteration_results.append(
                        {
                            "function_name": display_name,
                            "metric_name": metric_name,
                            "metric_value": mean_metric,
                            "param_names": params,
                            "code_file": str(model_file),
                            "individual_differences": id_results,
                            "code": func_code,
                            "eval_metrics": fit_res.get("eval_metrics", []),
                            "participant_n_trials": fit_res.get(
                                "participant_n_trials", []
                            ),
                        }
                    )

                    if mean_metric < self.best_metric:
                        self.best_metric = mean_metric
                        self.best_model = func_code
                        self.best_iter = it
                        self.best_params = params
                        self.best_param_names = fit_res["param_names"]
                        self.best_param_values = fit_res["parameter_values"]
                        self.best_id_results = id_results
                        console.print(
                            f"  [bold green]New best model:[/] {display_name} ({metric_name}={mean_metric:.2f})"
                        )

                        best_model_file = (
                            self.results_dir
                            / "models"
                            / f"best_model{tag}_{run_idx}.txt"
                            if getattr(self.cfg.evaluation, "fit_type", "group")
                            != "individual"
                            else self.results_dir
                            / "models"
                            / f"best_model{tag}_{run_idx}_participant{self.df.participant[0]}.txt"
                        )
                        with open(best_model_file, "w") as f:
                            f.write(func_code)

                        # save best model bic
                        best_bic_file = (
                            self.results_dir / "bics" / f"best_bic{tag}_{run_idx}.json"
                            if getattr(self.cfg.evaluation, "fit_type", "group")
                            != "individual"
                            else self.results_dir
                            / "bics"
                            / f"best_bic{tag}_{run_idx}_participant{self.df.participant[0]}.json"
                        )
                        with open(best_bic_file, "w") as f:
                            json.dump({"bic": mean_metric}, f)

                    # ✅ stop if ANY model beats baseline
                    if baseline_bic is not None and mean_metric < baseline_bic:
                        stop_iterations = True
                        # optional: break here to save compute evaluating remaining models
                        break

                except ModelValidationError as e:
                    console.print(
                        f"  [bold red]Validation error in {display_name}:[/] {e.message}"
                    )
                    iteration_results.append(
                        {
                            "function_name": display_name,
                            "metric_name": "VALIDATION_ERROR",
                            "metric_value": float("inf"),
                            "param_names": [],
                            "code": func_code,
                            "error_type": e.error_type,
                            "error_message": e.message,
                            "error_details": e.details,
                        }
                    )
                except Exception as e:
                    console.print(f"  [bold red]Error fitting {display_name}:[/] {e}")
                    iteration_results.append(
                        {
                            "function_name": display_name,
                            "metric_name": "FIT_ERROR",
                            "metric_value": float("inf"),
                            "param_names": [],
                            "code": func_code,
                            "error": str(e),
                        }
                    )

            self._set_activity(f"saving results (iter {it})")

            # ✅ always save what happened this iteration (even if stopping)
            bic_file = (
                self.results_dir / "bics" / f"iter{it}{tag}_run{run_idx}.json"
                if getattr(self.cfg.evaluation, "fit_type", "group") != "individual"
                else self.results_dir
                / "bics"
                / f"iter{it}{tag}_run{run_idx}_participant{self.df.participant[0]}.json"
            )
            with open(bic_file, "w") as f:
                json.dump(iteration_results, f, indent=2)

            self.feedback.record_iteration(it, iteration_results)

            # --- Push results to shared registry (distributed mode) ---
            self._update_registry(it, iteration_results)

        console.print(
            f"\n[bold]Search complete.[/] "
            f"Best model (iteration {self.best_iter}): "
            f"{self.cfg.evaluation.metric.upper()} = [bold cyan]{self.best_metric:.2f}[/]"
        )

        # --- save best parameters ---
        if (
            self.best_model is not None
            and self.best_params
            and self.best_param_values is not None
        ):
            param_df = pd.DataFrame(
                self.best_param_values, columns=self.best_param_names
            )

            param_dir = self.results_dir / "parameters"
            param_dir.mkdir(parents=True, exist_ok=True)

            tag = self._file_tag()
            param_file = (
                param_dir / f"best_params{tag}_run{run_idx}.csv"
                if getattr(self.cfg.evaluation, "fit_type", "group") != "individual"
                else param_dir
                / f"best_params{tag}_run{run_idx}_participant{self.df.participant[0]}.csv"
            )

            param_df.to_csv(param_file, index=False)

        # --- Mark client complete in shared registry ---
        if self.shared_registry is not None and self.client_id is not None:
            self.shared_registry.mark_complete(self.client_id)

        return self.best_model, self.best_metric, self.best_params
