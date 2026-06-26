# engine/model_search.py
import os
import json
import math
import time
from typing import Any, Optional
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

from gecco.artifacts import ArtifactStore
from gecco.candidate_evaluation import BestModelState, CandidateEvaluator
from gecco.candidate_generation import CandidateGenerator
from gecco.distributed_coordinator import DistributedCoordinator
from gecco.feedback_coordinator import FeedbackCoordinator
from gecco.load_llms.provider_registry import get_provider_spec
from gecco.run_context import RunContext
from gecco.utils import log as _log, TimestampedConsole
from config.schema import get_judge_capabilities, get_judge_mode
from gecco.construct_feedback.orchestrated import run_orchestrated_judge_pipeline
from pathlib import Path

console = TimestampedConsole()


def _mapping_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a key from either a mapping or an attribute container."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


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


class _IterationFeedbackState:
    """Track per-iteration search history for judge context building."""

    def __init__(self):
        self.history: list[dict] = []

    def record_iteration(self, iteration: int, results: list[dict]) -> None:
        """Append one iteration's results to the local history."""
        self.history.append({"iteration": iteration, "results": results})


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

        # --- Local iteration history for judge/runtime state ---
        self.feedback = _IterationFeedbackState()

        # --- Run context / resolved paths ---
        self.run_context = RunContext.from_cfg(cfg, client_id=client_id)
        self.project_root = self.run_context.project_root
        self.results_dir = self.run_context.results_dir

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
        self.best_state = BestModelState()

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

                db_path = self.run_context.default_diagnostics_path()
                self.diagnostic_store = DiagnosticStore(db_path)
                console.print(f"[dim]Diagnostic store: {db_path}[/]")
            except ImportError:
                console.print(
                    "[yellow]duckdb not installed — diagnostic store disabled.[/]"
                )

        # --- Unified judge pipeline ---
        self.tool_judge = None
        judge_mode = get_judge_mode(cfg)
        self.judge_enabled = bool(judge_cfg is not None and judge_mode != "off")
        judge_needs_store = judge_mode not in ("off", "random")
        if self.judge_enabled:
            if shared_registry is not None:
                console.print(
                    "[dim]Orchestrated mode: per-client tool judge skipped.[/]"
                )
            elif self.diagnostic_store is None and judge_needs_store:
                raise ValueError(
                    "Judge requires judge.diagnostic_store.enabled=true"
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
                console.print("[dim]Unified judge pipeline initialised.[/]")

        self.artifact_store = ArtifactStore(self.run_context, self.diagnostic_store)
        self.candidate_generator = CandidateGenerator(self.artifact_store)
        self.candidate_evaluator = CandidateEvaluator(self.artifact_store)
        self.feedback_coordinator = FeedbackCoordinator(run_orchestrated_judge_pipeline)
        self.distributed_coordinator = DistributedCoordinator()

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

        self._sync_best_state_from_attrs()

    def close(self) -> None:
        """Release runtime-owned resources."""

        if getattr(self, "diagnostic_store", None) is not None:
            self.diagnostic_store.close()
            self.diagnostic_store = None
        if getattr(self, "run_context", None) is not None:
            self.run_context.close()

    # --- Explicit collaborator accessors (no fallback construction) ---

    def _require_candidate_generator(self):
        if not hasattr(self, "candidate_generator") or self.candidate_generator is None:
            raise RuntimeError("CandidateGenerator collaborator is required")
        return self.candidate_generator

    def _require_candidate_evaluator(self):
        if not hasattr(self, "candidate_evaluator") or self.candidate_evaluator is None:
            raise RuntimeError("CandidateEvaluator collaborator is required")
        return self.candidate_evaluator

    def _require_artifact_store(self):
        if not hasattr(self, "artifact_store") or self.artifact_store is None:
            raise RuntimeError("ArtifactStore collaborator is required")
        return self.artifact_store

    def _require_distributed_coordinator(self):
        if not hasattr(self, "distributed_coordinator") or self.distributed_coordinator is None:
            raise RuntimeError("DistributedCoordinator collaborator is required")
        return self.distributed_coordinator

    def _require_feedback_coordinator(self):
        if not hasattr(self, "feedback_coordinator") or self.feedback_coordinator is None:
            raise RuntimeError("FeedbackCoordinator collaborator is required")
        return self.feedback_coordinator

    def _sync_best_state_from_attrs(self) -> None:
        """Mirror the public best-model attributes into the shared state object."""

        self.best_state.best_metric = self.best_metric
        self.best_state.best_model = self.best_model
        self.best_state.best_params = list(self.best_params)
        self.best_state.best_iter = self.best_iter
        self.best_state.best_param_names = list(self.best_param_names)
        self.best_state.best_param_values = self.best_param_values
        self.best_state.best_id_results = self.best_id_results

    def _sync_best_attrs_from_state(self) -> None:
        """Mirror the shared best-model state back to public attributes."""

        self.best_metric = self.best_state.best_metric
        self.best_model = self.best_state.best_model
        self.best_params = list(self.best_state.best_params)
        self.best_iter = self.best_state.best_iter
        self.best_param_names = list(self.best_state.best_param_names)
        self.best_param_values = self.best_state.best_param_values
        self.best_id_results = self.best_state.best_id_results

    def _cmg_config(self):
        """Return CMG config object if enabled, else None."""
        cmg_cfg = getattr(self.cfg, "centralized_model_generation", None)
        if cmg_cfg is None or not getattr(cmg_cfg, "enabled", False):
            return None
        return cmg_cfg

    def _cmg_is_generator(self, cmg_cfg):
        """Return True if this client is the CMG generator."""
        return str(self.client_id) == str(getattr(cmg_cfg, "generator_client", ""))

    def _cmg_evaluator_index(self, cmg_cfg):
        """Return numeric evaluator index for this client, or None."""
        try:
            idx = int(self.client_id)
        except (TypeError, ValueError):
            return None
        n_models = getattr(cmg_cfg, "n_models", None)
        if n_models is None or idx < 0 or idx >= n_models:
            return None
        return idx

    def _validate_cmg_runtime(self, cmg_cfg):
        """Validate CMG config at runtime. Raises ValueError if invalid."""
        if self.shared_registry is None:
            raise ValueError("centralized_model_generation requires a shared registry")
        if getattr(self.cfg, "judge", None) is None:
            raise ValueError("centralized_model_generation requires judge configuration")
        generator_client = str(getattr(cmg_cfg, "generator_client", ""))
        if not generator_client:
            raise ValueError("centralized_model_generation.generator_client is required")
        if generator_client.isdigit() or generator_client.lstrip("-").isdigit():
            raise ValueError(
                "centralized_model_generation.generator_client must be a named profile, "
                "not a numeric evaluator ID"
            )
        n_models = getattr(cmg_cfg, "n_models", None)
        if not isinstance(n_models, int) or n_models <= 0:
            raise ValueError("centralized_model_generation.n_models must be a positive integer")

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
        provider_spec = get_provider_spec(self.cfg.llm.provider)
        active_system_prompt = (
            system_prompt if system_prompt is not None else self.cfg.llm.system_prompt
        )

        # -----------------------------
        # OpenAI / GPT-style generation
        # -----------------------------
        if provider_spec.api_family == "openai":
            console.print(
                f"[yellow]Using {provider_spec.label} API provider: {self.cfg.llm.base_model}[/]"
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

        elif provider_spec.api_family == "gemini":
            console.print(
                f"[yellow]Using {provider_spec.label} API provider: {self.cfg.llm.base_model}[/]"
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
        elif provider_spec.api_family == "openai_compatible":
            console.print(
                f"[yellow]Using {provider_spec.label} API provider: {self.cfg.llm.base_model}[/]"
            )
            max_out = getattr(
                self.cfg.llm,
                "max_output_tokens",
                getattr(self.cfg.llm, "max_tokens", 4096),
            )

            provider_label = provider_spec.label
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
                use_json_schema = (
                    provider_spec.structured_output_mode == "chat_json_schema_optional"
                    and getattr(self.cfg.llm, "supports_json_schema", False)
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
        coordinator = self._require_distributed_coordinator()
        sync_result = coordinator.sync_from_registry(
            shared_registry=self.shared_registry,
            best_metric=self.best_metric,
            best_model=self.best_model,
            best_params=self.best_params,
            tried_param_sets=self.tried_param_sets,
            feedback_history=self.feedback.history,
            merged_history_count=self._merged_history_count,
            client_id=self.client_id,
        )
        self.best_metric = sync_result.best_metric
        self.best_model = sync_result.best_model
        self.best_params = sync_result.best_params
        self.tried_param_sets = sync_result.tried_param_sets
        self.feedback.history = sync_result.feedback_history
        self._merged_history_count = sync_result.merged_history_count
        self._sync_best_state_from_attrs()

    def _set_activity(self, activity):
        """Update current activity in the shared registry."""
        coordinator = self._require_distributed_coordinator()
        coordinator.set_activity(
            shared_registry=self.shared_registry,
            client_id=self.client_id,
            activity=activity,
        )

    def _run_cmg_generator_iteration(self, it, run_idx, feedback, cmg_cfg):
        """Generator path: generate candidates and publish to registry."""
        clients = _mapping_get(self.cfg, "clients")
        client_config = _mapping_get(clients, self.client_id) if self.client_id else None
        naive_enabled = bool(
            client_config
            and _mapping_get(_mapping_get(client_config, "naive_ideation"), "enabled", False)
        )
        participant = (
            getattr(self.df, "participant", [None])[0]
            if getattr(self, "df", None) is not None
            else None
        )
        generator = self._require_candidate_generator()
        tag = self._file_tag()
        generator.generate_iteration(
            iteration=it,
            run_idx=run_idx,
            feedback=feedback,
            cmg_cfg=cmg_cfg,
            tag=tag,
            client_id=self.client_id,
            naive_enabled=naive_enabled,
            prompt_builder=self.prompt_builder,
            generate_text=self.generate,
            model=self.model,
            tokenizer=self.tokenizer,
            cfg=self.cfg,
            shared_registry=self.shared_registry,
            participant=participant,
            set_activity=self._set_activity,
        )

    def _run_cmg_evaluator_iteration(self, it, run_idx, feedback, cmg_cfg, baseline_bic):
        """Evaluator path: fit assigned candidate with repair loop."""
        participant = (
            getattr(self.df, "participant", [None])[0]
            if getattr(self, "df", None) is not None
            else None
        )
        evaluator = self._require_candidate_evaluator()
        evaluator.evaluate_iteration(
            iteration=it,
            run_idx=run_idx,
            cmg_cfg=cmg_cfg,
            tag=self._file_tag(),
            client_id=self.client_id,
            evaluator_index=self._cmg_evaluator_index(cmg_cfg),
            baseline_bic=baseline_bic,
            shared_registry=self.shared_registry,
            df=self.df,
            cfg=self.cfg,
            recovery_checker=self.recovery_checker,
            id_eval_data=self.id_eval_data,
            ppc_enabled=self.ppc_enabled,
            ppc_simulator=self._ppc_simulator,
            ppc_n_sims=self.ppc_n_sims,
            block_residuals_enabled=self.block_residuals_enabled,
            block_residuals_n_blocks=self.block_residuals_n_blocks,
            set_activity=self._set_activity,
            model=self.model,
            tokenizer=self.tokenizer,
            generate_text=self.generate,
            prompt_builder=self.prompt_builder,
            max_syntax_retries=getattr(
                getattr(self.cfg, "validation", None), "max_syntax_retries", 2
            ),
            barrier_timeout_seconds=getattr(
                getattr(self.cfg.judge, "barrier", None),
                "client_wait_seconds",
                1800,
            ),
            participant=participant,
            best_state=self.best_state,
            tried_param_sets=self.tried_param_sets,
        )
        self._sync_best_attrs_from_state()

    def run_n_shots(self, run_idx, baseline_bic):
        # Resume from the next iteration after what's already in the registry
        cmg_cfg = self._cmg_config()
        distributed_coordinator = self._require_distributed_coordinator()
        start_iter = distributed_coordinator.start_iteration(
            shared_registry=self.shared_registry,
            client_id=self.client_id,
            cmg_cfg=cmg_cfg,
            is_generator=self._cmg_is_generator(cmg_cfg) if cmg_cfg is not None else False,
        )
        max_existing = start_iter - 1
        if max_existing >= 0:
            console.print(
                f"[dim]Resuming from iteration {start_iter} (registry has up to {max_existing})[/]"
            )

        end_iter = self.cfg.loop.max_iterations
        for it in range(start_iter, end_iter):
            console.rule(f"[bold]Iteration {it}")

            if self.shared_registry is not None:
                self.shared_registry.raise_if_aborted()

            # --- Sync from shared registry (distributed mode) ---
            self._sync_from_registry()

            tag = self._file_tag()
            feedback = ""

            if self.shared_registry is not None and bool(getattr(self, "judge_enabled", False)) and it > 0:
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

                if self.tool_judge is not None:
                    feedback_coordinator = self._require_feedback_coordinator()
                    feedback, verdict = feedback_coordinator.resolve_feedback(
                        judge=self.tool_judge,
                        cfg=self.cfg,
                        results_dir=self.results_dir,
                        iteration=it,
                        run_idx=run_idx,
                        tag=tag,
                        best_model=self.best_model,
                        best_metric=self.best_metric,
                        recovery_failures=recovery_failures,
                        prev_had_success=prev_had_success,
                        persona_name=self.client_id or "default",
                        set_activity=self._set_activity,
                    )
                elif not feedback:
                    feedback = ""
                    verdict = SimpleNamespace(
                        synthesized_feedback="",
                        key_recommendations=[],
                    )

            # Save feedback for inspection (runs whenever feedback was populated)
            if feedback:
                participant = self.df.participant[0] if hasattr(self.df, "participant") else None
                artifact_store = self._require_artifact_store()
                artifact_store.write_feedback_text(
                    iteration=it,
                    run_idx=run_idx,
                    tag=tag,
                    feedback=feedback,
                    participant=participant,
                )

            # --- Centralized Model Generation (CMG) branch ---
            cmg_cfg = self._cmg_config()
            if cmg_cfg is not None:
                self._validate_cmg_runtime(cmg_cfg)
                if self._cmg_is_generator(cmg_cfg):
                    self._run_cmg_generator_iteration(it, run_idx, feedback, cmg_cfg)
                    continue
                else:
                    self._run_cmg_evaluator_iteration(it, run_idx, feedback, cmg_cfg, baseline_bic)
                    continue

            # --- Syntax retry loop ---
            # Track retries for syntax/validation failures
            syntax_retry_count = 0
            max_syntax_retries = getattr(
                getattr(self.cfg, "validation", None), "max_syntax_retries", 2
            )
            artifact_store = self._require_artifact_store()
            generator = self._require_candidate_generator()
            evaluator = self._require_candidate_evaluator()
            participant = (
                self.df.participant[0]
                if getattr(self.cfg.evaluation, "fit_type", "group") == "individual"
                and hasattr(self.df, "participant")
                else None
            )

            while syntax_retry_count <= max_syntax_retries:
                self._set_activity(
                    f"generating models (iter {it}, attempt {syntax_retry_count + 1})"
                )

                # Update registry with retrying status if not first attempt
                generation_result = generator.generate_non_cmg_iteration(
                    iteration=it,
                    run_idx=run_idx,
                    feedback=feedback,
                    n_models=getattr(self.cfg.llm, "models_per_iteration", 1),
                    cfg=self.cfg,
                    tag=tag,
                    prompt_builder=self.prompt_builder,
                    generate_text=self.generate,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    participant=participant,
                    client_id=self.client_id,
                )

                evaluation_result = evaluator.run_non_cmg_iteration(
                    iteration=it,
                    run_idx=run_idx,
                    tag=tag,
                    generation_result=generation_result,
                    baseline_bic=baseline_bic,
                    df=self.df,
                    cfg=self.cfg,
                    shared_registry=self.shared_registry,
                    client_id=self.client_id,
                    results_source=self.df,
                    best_state=self.best_state,
                    recovery_checker=self.recovery_checker,
                    id_eval_data=self.id_eval_data,
                    ppc_enabled=self.ppc_enabled,
                    ppc_simulator=self._ppc_simulator,
                    ppc_n_sims=self.ppc_n_sims,
                    block_residuals_enabled=self.block_residuals_enabled,
                    block_residuals_n_blocks=self.block_residuals_n_blocks,
                    set_activity=self._set_activity,
                    max_syntax_retries=max_syntax_retries,
                    syntax_retry_count=syntax_retry_count,
                    participant=participant,
                    feedback_record=self.feedback.record_iteration,
                    tried_param_sets=self.tried_param_sets,
                    )

                self._sync_best_attrs_from_state()

                if evaluation_result.should_retry:
                    feedback = evaluation_result.retry_feedback or feedback
                    syntax_retry_count += 1
                    console.print(
                        f"[yellow]All models failed syntax validation, retrying "
                        f"({syntax_retry_count}/{max_syntax_retries})[/]"
                    )
                    continue

                if evaluation_result.had_runnable_model:
                    self._sync_best_attrs_from_state()

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

        return self.best_model, self.best_metric, self.best_params
