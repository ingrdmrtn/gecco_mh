# engine/model_search.py
import os
import json
import math
import time
from typing import Optional
from types import SimpleNamespace

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
from gecco.utils import log as _log, TimestampedConsole
from gecco.sentry_init import capture_fit_error, capture_recovery_failed
from pathlib import Path

console = TimestampedConsole()


class _NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        return super().default(o)

    def encode(self, o):
        return super().encode(self._sanitize(o))

    def _sanitize(self, o):
        if isinstance(o, float):
            if math.isinf(o) or math.isnan(o):
                return None
        elif isinstance(o, dict):
            return {k: self._sanitize(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self._sanitize(v) for v in o]
        return o


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
        df_val=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.df = df
        self.df_val = df_val
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
        self._ppc_simulator = None
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
            # Reuse the same simulator for PPC
            self._ppc_simulator = simulator

        # --- Diagnostic store (optional) ---
        self.diagnostic_store = None
        judge_cfg = getattr(cfg, "judge", None)
        if judge_cfg and getattr(
            getattr(judge_cfg, "diagnostic_store", None), "enabled", False
        ):
            try:
                from gecco.diagnostic_store import DiagnosticStore

                shard = f"_{self.client_id}" if self.client_id else ""
                db_path = self.results_dir / f"diagnostics{shard}.duckdb"
                self.diagnostic_store = DiagnosticStore(db_path)
                console.print(f"[dim]Diagnostic store: {db_path}[/]")
            except ImportError:
                console.print(
                    "[yellow]duckdb not installed — diagnostic store disabled.[/]"
                )

        # --- Tool-using judge (optional) ---
        self.tool_judge = None
        _orchestrated = (
            shared_registry is not None
            and judge_cfg
            and getattr(judge_cfg, "orchestrated", False)
        )
        if judge_cfg and getattr(judge_cfg, "mode", "manual") == "tool_using":
            if _orchestrated:
                console.print(
                    "[dim]Orchestrated mode: per-client tool judge skipped.[/]"
                )
            elif self.diagnostic_store is None:
                console.print(
                    "[yellow]tool_using judge requires diagnostic_store.enabled=true — "
                    "falling back to standard feedback.[/]"
                )
            else:
                from gecco.construct_feedback.tool_judge import ToolUsingJudge

                self.tool_judge = ToolUsingJudge(
                    cfg=cfg,
                    diagnostic_store=self.diagnostic_store,
                    model=model,
                    tokenizer=tokenizer,
                    results_dir=self.results_dir,
                )
                console.print("[dim]Tool-using judge initialised.[/]")

        # --- PPC config ---
        ppc_cfg = getattr(judge_cfg, "ppc", None) if judge_cfg else None
        self.ppc_enabled = bool(ppc_cfg and getattr(ppc_cfg, "enabled", False))
        self.ppc_n_sims: int = getattr(ppc_cfg, "n_sims", 100) if ppc_cfg else 100
        block_residual_cfg = (
            getattr(judge_cfg, "block_residuals", None) if judge_cfg else None
        )
        self.block_residuals_enabled = bool(
            getattr(
                block_residual_cfg,
                "enabled",
                self.ppc_enabled,
            )
            if judge_cfg
            else False
        )
        self.block_residuals_n_blocks: int = (
            getattr(
                block_residual_cfg,
                "n_blocks",
                10,
            )
            if block_residual_cfg
            else 10
        )

    def generate(
        self,
        model,
        tokenizer=None,
        prompt=None,
        response_schema=None,
        system_prompt: Optional[str] = None,
    ):
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
                system_prompt : str, optional
                    Explicit system prompt to use. If None, falls back to cfg.llm.system_prompt.

                Returns
        -------
                str
                    The generated text response.
        """
        if model is None:
            raise ValueError("Model not initialized correctly.")
        provider = self.cfg.llm.provider.lower()
        active_system_prompt = (
            system_prompt if system_prompt is not None else self.cfg.llm.system_prompt
        )

        # -----------------------------
        # OpenAI / GPT-style generation
        # -----------------------------
        if "openai" in provider or "gpt" in provider:
            console.print(
                f"[yellow]Using OpenAI-compatible API provider: {self.cfg.llm.base_model}[/]"
            )
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
                    {"role": "developer", "content": active_system_prompt},
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
            console.print(
                f"[yellow]Using Gemini-compatible API provider: {self.cfg.llm.base_model}[/]"
            )
            from google.genai import types

            reasoning_effort = getattr(self.cfg.llm, "reasoning_effort", "low")

            console.print(
                f"[dim]Generating with Gemini [cyan]{self.cfg.llm.base_model}[/] "
                f"(reasoning={reasoning_effort})[/]"
            )

            config_args = {
                "temperature": self.cfg.llm.temperature,
                "system_instruction": active_system_prompt,
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
            console.print(
                f"[yellow]Using vLLM-compatible API provider: {self.cfg.llm.base_model}[/]"
            )
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
            temperature = getattr(self.cfg.llm, "temperature", None)
            console.print(
                f"[dim]Generating with {provider_label} [cyan]{self.cfg.llm.base_model}[/] "
                f"(max_tokens={max_out}, temp={temperature})[/]"
            )

            create_kwargs = {
                "model": self.cfg.llm.base_model,
                "messages": [
                    {"role": "system", "content": active_system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_out,
            }
            if temperature is not None:
                create_kwargs["temperature"] = temperature

            # Structured output for OpenAI-compatible APIs.
            # OpenRouter supports full json_schema enforcement when the model supports it
            # (opt-in via supports_json_schema: true in config).
            # All other providers (vLLM, KCL, OpenCode) fall back to json_object mode.
            if response_schema is not None:
                use_json_schema = "openrouter" in provider and getattr(
                    self.cfg.llm, "supports_json_schema", False
                )
                if use_json_schema:
                    from gecco.structured_output import get_chat_json_schema_format

                    create_kwargs["response_format"] = get_chat_json_schema_format(
                        response_schema
                    )
                    # Route only to providers that honour the response_format parameter.
                    # Without this, OpenRouter may silently forward to a provider that
                    # ignores json_schema and returns free-form text.
                    create_kwargs["extra_body"] = {
                        "provider": {"require_parameters": True}
                    }
                else:
                    from gecco.structured_output import (
                        get_openai_compatible_response_format,
                    )

                    create_kwargs["response_format"] = (
                        get_openai_compatible_response_format()
                    )

            # debug_kwargs = {k: v for k, v in create_kwargs.items() if k != "messages"}
            # console.print(f"[dim]Request kwargs (excl. messages): {debug_kwargs!r}[/]")
            try:
                resp = model.chat.completions.create(**create_kwargs)
                # Log the raw response for debugging, especially to inspect reasoning_details and any API error messages
                # console.print(f"RAW response object:")
                # console.print(f"[dim]Raw response: {resp!r}[/]")
            except Exception as api_exc:
                # Catch 404 from OpenRouter when no endpoint supports the
                # requested parameters (e.g. json_schema mode not available
                # for this model).  Surface a clear actionable message rather
                # than a raw stack trace.
                exc_str = str(api_exc)
                # Log the raw exception for debugging
                console.print(f"[dim]API exception: {exc_str}[/]")
                if "404" in exc_str and "No endpoints found" in exc_str:
                    hint = ""
                    if "extra_body" in create_kwargs and create_kwargs.get(
                        "extra_body", {}
                    ).get("provider", {}).get("require_parameters"):
                        hint = (
                            f"\n  [bold]Cause:[/] [cyan]{self.cfg.llm.base_model}[/] has no "
                            f"OpenRouter endpoint that supports [bold]json_schema[/] structured output. "
                            f"Remove [bold]supports_json_schema: true[/] from the config to fall back "
                            f"to json_object mode."
                        )
                    console.print(
                        f"[red]OpenRouter 404 — no matching endpoint.{hint}[/]"
                    )
                    return ""
                raise
            if not hasattr(resp, "choices"):
                raise TypeError(
                    f"Expected a ChatCompletion response but got {type(resp).__name__!r}. "
                    f"Response: {resp!r:.200}"
                )
            if not resp.choices:
                api_error = getattr(resp, "error", None)
                if api_error:
                    code = api_error.get("code", "?")
                    message = api_error.get("message", "unknown error")
                    console.print(f"[yellow]API error {code}: {message}[/]")
                else:
                    msg = "[yellow]API returned empty choices list"
                    if "response_format" in create_kwargs:
                        fmt = create_kwargs["response_format"]
                        fmt_type = (
                            fmt.get("type", "unknown")
                            if isinstance(fmt, dict)
                            else getattr(fmt, "type", "unknown")
                        )
                        msg += (
                            f"\n  [bold]Likely cause:[/] structured output was requested "
                            f"(response_format={fmt_type!r}) but [cyan]{self.cfg.llm.base_model}[/] "
                            f"may not support it. Try setting [bold]structured_output: false[/] in the config."
                        )
                    msg += "[/]"
                    console.print(msg)
                    console.print(f"[dim]Full response: {resp!r}[/]")
                return ""
            message = resp.choices[0].message
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning:
                console.print(
                    f"[dim](reasoning tokens present, {len(reasoning)} chars)[/]"
                )
            content = message.content
            if content is None:
                finish = getattr(resp.choices[0], "finish_reason", "unknown")
                msg = f"[yellow]API returned empty response (finish_reason={finish})"
                if finish == "length":
                    msg += (
                        f"\n  [bold]Likely cause:[/] response was truncated at max_tokens={max_out}. "
                        f"Try increasing [bold]max_tokens[/] in the config."
                    )
                elif "response_format" in create_kwargs:
                    fmt = create_kwargs["response_format"]
                    fmt_type = (
                        fmt.get("type", "unknown")
                        if isinstance(fmt, dict)
                        else getattr(fmt, "type", "unknown")
                    )
                    msg += (
                        f"\n  [bold]Likely cause:[/] structured output was requested "
                        f"(response_format={fmt_type!r}) but [cyan]{self.cfg.llm.base_model}[/] "
                        f"may not support it. Try setting [bold]structured_output: false[/] in the config."
                    )
                msg += "[/]"
                console.print(msg)
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

    def generate_models_naive(self, feedback_text):
        """
        Two-phase generation:
        1. Phase 1: Naive ideation (psychologist persona)
        2. Phase 2: Computational translation (neuroscientist persona)
        """
        # 1. Read naive ideation config
        client_config = (
            getattr(self.cfg.clients, self.client_id, None) if self.client_id else None
        )
        naive_cfg = (
            getattr(client_config, "naive_ideation", None) if client_config else None
        )

        if not naive_cfg or not getattr(naive_cfg, "enabled", False):
            # Fallback to standard generation if misconfigured
            prompt = self.prompt_builder.build_input_prompt(feedback_text=feedback_text)
            return self.generate_models(prompt)

        persona = naive_cfg.persona
        translation_preamble = getattr(naive_cfg, "translation_preamble", None)

        if "hf" in self.cfg.llm.provider or "huggingface" in self.cfg.llm.provider:
            console.print(
                "[yellow]Warning: HuggingFace backend does not support system prompts. "
                "Phase 1 persona will have no effect.[/]"
            )

        # 2. Phase 1 — ideation call
        console.print("  [dim]Phase 1: Naive psychological ideation...[/]")
        naive_prompt = self.prompt_builder.build_naive_prompt(feedback_text)

        naive_idea = self.generate(
            self.model,
            self.tokenizer,
            naive_prompt,
            response_schema=None,  # Plain text
            system_prompt=persona,
        )

        if not naive_idea:
            console.print(
                "  [yellow]Phase 1 ideation failed to return an idea — using empty idea.[/]"
            )
            naive_idea = ""
        else:
            console.print(f"  [dim]Naive hypothesis:[/] {naive_idea[:200]}...")

        # 3. Phase 2 — translation call
        console.print("  [dim]Phase 2: Computational translation...[/]")
        prompt = self.prompt_builder.build_input_prompt(
            feedback_text=feedback_text,
            naive_idea=naive_idea,
            translation_preamble=translation_preamble,
        )

        return self.generate_models(prompt)

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

    def _update_registry(
        self, iteration, results, status="running", had_runnable_model=None
    ):
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
            status=status,
            had_runnable_model=had_runnable_model,
        )

    def _build_syntax_error_feedback(self, iteration_results):
        """
        Build feedback text from syntax/validation errors for regeneration.

        This is used when all models fail syntax validation to help the LLM
        understand what went wrong and how to fix it.
        """
        error_messages = []
        for i, result in enumerate(iteration_results):
            model_name = result.get("function_name", f"model_{i}")
            error_type = result.get("metric_name", "ERROR")

            if error_type == "VALIDATION_ERROR":
                msg = result.get("error_message", "Unknown validation error")
                error_messages.append(f"- {model_name}: {msg}")
            elif error_type == "FIT_ERROR":
                error_msg = result.get("error", "Unknown fit error")
                # Truncate very long error messages
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                error_messages.append(f"- {model_name}: {error_msg}")

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

            tag = self._file_tag()
            feedback = ""

            orchestrated_judge_enabled = (
                self.shared_registry is not None
                and getattr(self.cfg, "judge", None) is not None
                and getattr(self.cfg.judge, "orchestrated", False)
            )

            if orchestrated_judge_enabled and it > 0:
                barrier_timeout = getattr(
                    getattr(self.cfg.judge, "barrier", None),
                    "client_wait_seconds",
                    1800,
                )
                self._set_activity(f"waiting for centralized judge (iter {it})")
                shared_feedback_dict = self.shared_registry.wait_for_judge_feedback(
                    iteration=it - 1,
                    timeout_seconds=barrier_timeout,
                    poll_seconds=2.0,
                )

                verdict = SimpleNamespace(
                    synthesized_feedback="", key_recommendations=[]
                )
                if shared_feedback_dict is not None and shared_feedback_dict.get(
                    "failed"
                ):
                    error_msg = shared_feedback_dict.get("error", "unknown")
                    console.print(
                        f"  [yellow]Orchestrated judge failed for iteration {it}: "
                        f"{error_msg}. Using failure as feedback for regeneration.[/]"
                    )
                    feedback = (
                        f"The judge failed to analyze the previous iteration: {error_msg}. "
                        f"Please try a different approach or simplify your models. "
                        f"Consider: 1) Checking model syntax, 2) Ensuring models are "
                        f"identifiable, 3) Using simpler parameterizations."
                    )
                elif shared_feedback_dict is not None:
                    synthesized_feedback = shared_feedback_dict.get(
                        "synthesized_feedback", ""
                    )
                    if isinstance(synthesized_feedback, dict):
                        persona_name = self.client_id or "default"
                        feedback = synthesized_feedback.get(
                            persona_name, synthesized_feedback.get("default", "")
                        )
                    else:
                        feedback = synthesized_feedback
                    console.print(
                        f"  [green]Using centralized judge feedback for iteration {it}[/]"
                    )
                    verdict = SimpleNamespace(
                        synthesized_feedback=feedback,
                        key_recommendations=shared_feedback_dict.get(
                            "key_recommendations", []
                        ),
                    )
                else:
                    console.print(
                        f"  [yellow]Orchestrated judge timed out for iteration {it} "
                        f"(waited {barrier_timeout}s). Using timeout as feedback.[/]"
                    )
                    feedback = (
                        f"The judge timed out while analyzing the previous iteration. "
                        f"This may indicate the models were too complex to evaluate. "
                        f"Please try simpler models or ensure they can be evaluated efficiently."
                    )

            if self.best_model is not None:
                # --- Detect recovery failures from previous iteration ---
                recovery_failures = []
                prev_had_success = True
                if self.feedback.history:
                    last = self.feedback.history[-1]
                    if last["iteration"] == it - 1:
                        last_results = last["results"]
                        prev_had_success = any(
                            r.get("metric_name")
                            not in ("RECOVERY_FAILED", "FIT_ERROR", None)
                            for r in last_results
                        )
                        recovery_failures = [
                            {
                                "name": r.get("name")
                                or r.get("model_name")
                                or "unknown",
                                "mean_r": r.get("recovery_r"),
                                "per_param_r": r.get("recovery_per_param") or {},
                                "iteration": last["iteration"],
                            }
                            for r in last_results
                            if r.get("metric_name") == "RECOVERY_FAILED"
                        ]

                # --- Dispatch judge (non-orchestrated path only) ---
                if not orchestrated_judge_enabled:
                    if self.tool_judge is not None:
                        try:
                            self._set_activity(f"tool judge (iter {it})")

                            no_tools_lesion_active = (
                                getattr(self.cfg.judge, "lesion", None) is not None
                                and getattr(self.cfg.judge.lesion, "enabled", False)
                                and getattr(self.cfg.judge.lesion, "lesion_type", None)
                                == "no_tools"
                            )
                            if no_tools_lesion_active:
                                self.tool_judge._tool_loop = None

                            verdict = self.tool_judge.get_feedback(
                                iteration=it,
                                run_idx=run_idx,
                                tag=tag,
                                best_model=self.best_model,
                                best_metric=self.best_metric,
                                recovery_failures=recovery_failures
                                if recovery_failures
                                else None,
                                prev_had_success=prev_had_success,
                            )
                            feedback = verdict.synthesized_feedback
                        except Exception as e:
                            console.print(
                                f"  [yellow]Tool judge failed, falling back to standard feedback:[/] {e}"
                            )
                            feedback = self.feedback.get_feedback(
                                self.best_model,
                                self.tried_param_sets,
                                id_results=self.best_id_results,
                            )
                            verdict = SimpleNamespace(
                                synthesized_feedback=feedback,
                                key_recommendations=[],
                            )
                    else:
                        feedback = self.feedback.get_feedback(
                            self.best_model,
                            self.tried_param_sets,
                            id_results=self.best_id_results,
                        )
                        verdict = SimpleNamespace(
                            synthesized_feedback=feedback,
                            key_recommendations=[],
                        )

                # Apply lesion if configured
                lesion_cfg = getattr(self.cfg.judge, "lesion", None)
                if lesion_cfg and getattr(lesion_cfg, "enabled", False):
                    from gecco.construct_feedback.judge_lesion import JudgeLesion

                    if not hasattr(self, "_lesion"):
                        self._lesion = JudgeLesion(self.cfg.judge)
                    feedback = self._lesion.apply(verdict, it, feedback)

                # R1: Conditionally append best model code based on show_best_model_code flag
                show_best_model_code = getattr(
                    self.cfg.llm, "show_best_model_code", True
                )
                if show_best_model_code and self.best_model is not None:
                    best_metric_str = (
                        f"{self.best_metric:.2f}"
                        if self.best_metric is not None
                        else "N/A"
                    )
                    feedback += (
                        f"\n\n---\nBest model code so far (BIC={best_metric_str}):\n"
                        f"```python\n{self.best_model}\n```"
                    )

            # Save feedback for inspection (runs whenever feedback was populated)
            if feedback:
                feedback_file = (
                    self.results_dir / "feedback" / f"iter{it}{tag}_run{run_idx}.txt"
                    if getattr(self.cfg.evaluation, "fit_type", "group") != "individual"
                    else self.results_dir
                    / "feedback"
                    / f"iter{it}{tag}_run{run_idx}_participant{self.df.participant[0]}.txt"
                )
                with open(feedback_file, "w") as f:
                    f.write(feedback)

            # --- Syntax retry loop ---
            # Track retries for syntax/validation failures
            syntax_retry_count = 0
            max_syntax_retries = getattr(
                getattr(self.cfg, "validation", None), "max_syntax_retries", 2
            )

            while syntax_retry_count <= max_syntax_retries:
                self._set_activity(
                    f"generating models (iter {it}, attempt {syntax_retry_count + 1})"
                )

                # Update registry with retrying status if not first attempt
                if syntax_retry_count > 0 and self.shared_registry is not None:
                    self._update_registry(it, [], status="retrying")

                # Check for naive ideation
                client_config = (
                    getattr(self.cfg.clients, self.client_id, None)
                    if self.client_id
                    else None
                )
                naive_enabled = False
                if client_config and hasattr(client_config, "naive_ideation"):
                    naive_enabled = getattr(
                        client_config.naive_ideation, "enabled", False
                    )

                if naive_enabled:
                    code_text, parsed_models = self.generate_models_naive(feedback)
                else:
                    prompt = self.prompt_builder.build_input_prompt(
                        feedback_text=feedback
                    )
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
                        from gecco.offline_evaluation.exceptions import (
                            ModelValidationError,
                        )  # noqa: F811

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
                                            "recovery_per_param": recovery[
                                                "per_param_r"
                                            ],
                                            "recovery_n_successful": recovery[
                                                "n_successful"
                                            ],
                                            "simulation_error": sim_err,
                                        }
                                    )
                                    capture_recovery_failed(
                                        iteration=it,
                                        model_name=display_name,
                                        error=Exception(
                                            f"Parameter recovery failed (mean r={recovery['mean_r']:.2f})"
                                        ),
                                    )
                                    continue
                            except ModelValidationError as e:
                                console.print(
                                    f"  [yellow]{display_name} validation error ({e.error_type}): {e.message}[/]"
                                )
                                if e.details:
                                    for k, v in e.details.items():
                                        console.print(f"    [dim]{k}: {v}[/]")
                                safe_details = {}
                                for k, v in e.details.items():
                                    try:
                                        json.dumps(v)
                                        safe_details[k] = v
                                    except (TypeError, ValueError):
                                        safe_details[k] = str(v)
                                iteration_results.append(
                                    {
                                        "function_name": display_name,
                                        "metric_name": "VALIDATION_ERROR",
                                        "metric_value": float("inf"),
                                        "param_names": [],
                                        "code": func_code,
                                        "error_type": e.error_type,
                                        "error_message": e.message,
                                        "error_details": safe_details,
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
                                capture_fit_error(
                                    iteration=it,
                                    model_name=display_name,
                                    error=e,
                                    run=run_idx,
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
                                    fit_res,
                                    self.df,
                                    self.cfg,
                                    id_data=self.id_eval_data,
                                )
                            except Exception as e:
                                console.print(
                                    f"  [yellow]Individual differences eval failed for {display_name}:[/] {e}"
                                )

                        # --- Validation on val split (if provided) ---
                        val_fit_res = None
                        val_id_results = None
                        if self.df_val is not None:
                            try:
                                val_fit_res = run_fit(
                                    self.df_val,
                                    func_code,
                                    cfg=self.cfg,
                                    expected_func_name=func_name,
                                    structured_params=structured_params,
                                )
                                console.print(
                                    f"    [dim]val {metric_name} = [cyan]{val_fit_res['metric_value']:.2f}[/]"
                                )
                                if self.id_eval_data is not None:
                                    try:
                                        val_id_results = (
                                            evaluate_individual_differences(
                                                val_fit_res,
                                                self.df_val,
                                                self.cfg,
                                                id_data=self.id_eval_data,
                                            )
                                        )
                                    except Exception as e:
                                        console.print(
                                            f"  [yellow]Individual differences eval failed for {display_name} on val:[/] {e}"
                                        )
                            except Exception as e:
                                console.print(
                                    f"  [yellow]Val fitting failed for {display_name}:[/] {e}"
                                )

                        # --- Posterior predictive checks (optional) ---
                        ppc_result = None
                        block_residuals_result = None
                        needs_diagnostic_spec = bool(
                            fit_res.get("parameter_values")
                            and (
                                (self.ppc_enabled and self._ppc_simulator is not None)
                                or self.block_residuals_enabled
                            )
                        )
                        diagnostic_spec = None
                        if needs_diagnostic_spec:
                            try:
                                from gecco.offline_evaluation.utils import (
                                    build_model_spec,
                                )

                                diagnostic_spec = build_model_spec(
                                    func_code,
                                    expected_func_name=func_name,
                                    cfg=self.cfg,
                                    structured_params=structured_params,
                                )
                            except Exception as e:
                                console.print(
                                    f"  [yellow]Diagnostic spec build failed for {display_name}:[/] {e}"
                                )

                        if (
                            self.ppc_enabled
                            and self._ppc_simulator is not None
                            and fit_res.get("parameter_values")
                            and diagnostic_spec is not None
                        ):
                            try:
                                from gecco.offline_evaluation.ppc import compute_ppc

                                console.print(
                                    f"  [dim]Computing PPC for {display_name} "
                                    f"(n_sims={self.ppc_n_sims})...[/]"
                                )

                                # Count participants for progress bar
                                from gecco.offline_evaluation.ppc import (
                                    _get_participants,
                                )

                                _, participants = _get_participants(self.df)
                                n_participants = len(participants)

                                # Create progress bar for PPC
                                ppc_progress = Progress(
                                    TextColumn(
                                        "[progress.description]{task.description}"
                                    ),
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
                                        df=self.df,
                                        fitted_params_list=fit_res["parameter_values"],
                                        simulator=self._ppc_simulator,
                                        n_sims=self.ppc_n_sims,
                                        input_columns=list(self.cfg.data.input_columns),
                                        n_jobs=-1,
                                        progress_callback=lambda: ppc_progress.advance(
                                            task_id
                                        ),
                                    )
                            except Exception as e:
                                console.print(
                                    f"  [yellow]PPC failed for {display_name}:[/] {e}"
                                )

                        if (
                            self.block_residuals_enabled
                            and fit_res.get("parameter_values")
                            and diagnostic_spec is not None
                        ):
                            try:
                                from gecco.offline_evaluation.ppc import (
                                    compute_block_residuals,
                                )

                                console.print(
                                    f"  [dim]Computing block residuals for {display_name} "
                                    f"(n_blocks={self.block_residuals_n_blocks})...[/]"
                                )
                                block_residuals_result = compute_block_residuals(
                                    spec=diagnostic_spec,
                                    df=self.df,
                                    fitted_params_list=fit_res["parameter_values"],
                                    n_blocks=self.block_residuals_n_blocks,
                                    input_columns=list(self.cfg.data.input_columns),
                                )
                            except Exception as e:
                                console.print(
                                    f"  [yellow]Block residuals failed for {display_name}:[/] {e}"
                                )

                        result_dict = {
                            "function_name": display_name,
                            "metric_name": metric_name,
                            "metric_value": mean_metric,
                            "param_names": params,
                            "code_file": str(model_file),
                            "recovery": recovery
                            if self.recovery_checker is not None
                            else None,
                            "individual_differences": id_results,
                            "code": func_code,
                            "eval_metrics": fit_res.get("eval_metrics", []),
                            "participant_n_trials": fit_res.get(
                                "participant_n_trials", []
                            ),
                            "parameter_values": fit_res.get("parameter_values", []),
                            "mean_nll": fit_res.get("mean_nll"),
                            "per_participant_nll": fit_res.get("per_participant_nll"),
                        }
                        if val_fit_res is not None:
                            result_dict["val_metric_value"] = val_fit_res[
                                "metric_value"
                            ]
                            result_dict["val_mean_nll"] = val_fit_res["mean_nll"]
                            result_dict["val_eval_metrics"] = val_fit_res[
                                "eval_metrics"
                            ]
                            result_dict["val_per_participant_nll"] = val_fit_res[
                                "per_participant_nll"
                            ]
                            result_dict["val_individual_differences"] = val_id_results
                        if ppc_result is not None:
                            result_dict["ppc"] = ppc_result
                        if block_residuals_result is not None:
                            result_dict["block_residuals"] = block_residuals_result
                        iteration_results.append(result_dict)

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
                                self.results_dir
                                / "bics"
                                / f"best_bic{tag}_{run_idx}.json"
                                if getattr(self.cfg.evaluation, "fit_type", "group")
                                != "individual"
                                else self.results_dir
                                / "bics"
                                / f"best_bic{tag}_{run_idx}_participant{self.df.participant[0]}.json"
                            )
                            with open(best_bic_file, "w") as f:
                                json.dump(
                                    {"bic": mean_metric}, f, cls=_NumpyJSONEncoder
                                )

                            # save best model val metrics
                            if val_fit_res is not None:
                                best_bic_val_file = (
                                    self.results_dir
                                    / "bics"
                                    / f"best_bic_val{tag}_{run_idx}.json"
                                    if getattr(self.cfg.evaluation, "fit_type", "group")
                                    != "individual"
                                    else self.results_dir
                                    / "bics"
                                    / f"best_bic_val{tag}_{run_idx}_participant{self.df.participant[0]}.json"
                                )
                                with open(best_bic_val_file, "w") as f:
                                    json.dump(
                                        {
                                            "mean_BIC": val_fit_res["metric_value"],
                                            "mean_NLL": val_fit_res["mean_nll"],
                                            "individual_BIC": val_fit_res[
                                                "eval_metrics"
                                            ],
                                            "individual_NLL": val_fit_res[
                                                "per_participant_nll"
                                            ],
                                        },
                                        f,
                                        cls=_NumpyJSONEncoder,
                                    )

                        # ✅ stop if ANY model beats baseline
                        if baseline_bic is not None and mean_metric < baseline_bic:
                            stop_iterations = True
                            # optional: break here to save compute evaluating remaining models
                            break

                    except ModelValidationError as e:
                        console.print(
                            f"  [bold red]Validation error in {display_name}:[/] {e.message}"
                        )
                        safe_details = {}
                        for k, v in e.details.items():
                            try:
                                json.dumps(v)
                                safe_details[k] = v
                            except (TypeError, ValueError):
                                safe_details[k] = str(v)
                        iteration_results.append(
                            {
                                "function_name": display_name,
                                "metric_name": "VALIDATION_ERROR",
                                "metric_value": float("inf"),
                                "param_names": [],
                                "code": func_code,
                                "error_type": e.error_type,
                                "error_message": e.message,
                                "error_details": safe_details,
                            }
                        )
                    except Exception as e:
                        console.print(
                            f"  [bold red]Error fitting {display_name}:[/] {e}"
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
                        capture_fit_error(
                            iteration=it,
                            model_name=display_name,
                            error=e,
                            run=run_idx,
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
                    json.dump(iteration_results, f, indent=2, cls=_NumpyJSONEncoder)

                self.feedback.record_iteration(it, iteration_results)

                # --- Write to diagnostic store (optional) ---
                if self.diagnostic_store is not None:
                    try:
                        ppc_results_map = {}
                        for r in iteration_results:
                            if "ppc" in r:
                                ppc_results_map[r["function_name"]] = r["ppc"]
                        self.diagnostic_store.write_iteration(
                            iteration=it,
                            run_idx=run_idx,
                            iteration_results=iteration_results,
                            ppc_results=ppc_results_map if ppc_results_map else None,
                            tag=tag,
                            client_id=self.client_id,
                        )
                    except Exception as e:
                        console.print(
                            f"  [yellow]Diagnostic store write failed:[/] {e}"
                        )

                # --- Check for syntax retry ---
                # Determine if we had any runnable models
                had_runnable_model = (
                    any(
                        r.get("metric_name")
                        not in ("VALIDATION_ERROR", "FIT_ERROR", "RECOVERY_FAILED", None)
                        for r in iteration_results
                    )
                    if iteration_results
                    else False
                )

                # Check if ALL models failed with syntax/validation errors
                all_syntax_errors = (
                    all(
                        r.get("metric_name") in ("VALIDATION_ERROR", "FIT_ERROR")
                        for r in iteration_results
                    )
                    and iteration_results
                )

                if all_syntax_errors and syntax_retry_count < max_syntax_retries:
                    # Build error feedback for regeneration
                    error_feedback = self._build_syntax_error_feedback(
                        iteration_results
                    )
                    feedback = error_feedback
                    syntax_retry_count += 1
                    console.print(
                        f"[yellow]All models failed syntax validation, retrying "
                        f"({syntax_retry_count}/{max_syntax_retries})[/]"
                    )
                    # Continue the while loop to regenerate
                    continue

                # --- Push results to shared registry (distributed mode) ---
                # Determine completion status
                if had_runnable_model:
                    completion_status = "complete"
                else:
                    completion_status = "complete_no_success"
                self._update_registry(
                    it,
                    iteration_results,
                    status=completion_status,
                    had_runnable_model=had_runnable_model,
                )

                # Break out of retry loop - we're done with this iteration
                break

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
