# engine/model_search.py
import os
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table

from gecco.offline_evaluation.fit_generated_models import run_fit_hierarchical as run_fit
from gecco.utils import extract_model_code
from gecco.construct_feedback.feedback import FeedbackGenerator, LLMFeedbackGenerator
from pathlib import Path

console = Console()


class GeCCoModelSearch:
    def __init__(self, model, tokenizer, cfg, df, prompt_builder,
                 client_id=None, shared_registry=None):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.df = df
        self.prompt_builder = prompt_builder
        self.client_id = client_id
        self.shared_registry = shared_registry

        # --- Choose feedback generator based on config ---
        if hasattr(cfg, "feedback") and getattr(cfg.feedback, "type", "manual") == "llm":
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
        if hasattr(cfg, 'individual_differences_eval'):
            from gecco.offline_evaluation.individual_differences import load_id_data
            self.id_eval_data = load_id_data(cfg)

        # --- Tracking ---
        self.best_model = None
        self.best_metric = np.inf
        self.best_params = []
        self.best_iter = -1
        self.tried_param_sets = []
        self.best_id_results = None

        # --- Track which registry entries we've already merged ---
        self._merged_history_count = 0

    def generate(self, model, tokenizer=None, prompt=None):
        """
        Unified text generation function for any supported backend.
        Handles both OpenAI GPT and Hugging Face-style models cleanly.
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

            resp = model.responses.create(
                model=self.cfg.llm.base_model,
                reasoning={"effort": "low"},
                input=[
                    {"role": "developer", "content": self.cfg.llm.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
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

            if reasoning_effort:
                assert reasoning_effort in ["low", "high"], \
                    f"Invalid reasoning_effort: {reasoning_effort}. Choose from 'low', 'high'."
                if self.cfg.llm.base_model.lower().startswith("gemini-3"):
                    #note: flash models allow minimal and medium thinking levels but only low and high are exposed to be compatible with pro and flash
                    config_args["thinking_config"] = types.ThinkingConfig(
                        thinking_level=reasoning_effort
                    )
                elif self.cfg.llm.base_model.lower().startswith("gemini-2"):
                    #note: for Gemini 2, we set thinking_budget to approximate low/high levels (max thinking tokens for 2.5 pro is 32768; 2.5 flash is 24576)
                    # max set to 24576 to make it compatible with both 2.5 pro and flash
                    config_args["thinking_config"] = types.ThinkingConfig(
                        thinking_budget= 4096 if reasoning_effort == 'low' else 24576
                        )

            resp = model.models.generate_content(
                model=self.cfg.llm.base_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    **config_args
                )
            )
            decoded = resp.text.strip()

            return decoded

        # -----------------------------
        # vLLM (OpenAI-compatible API)
        # -----------------------------
        elif "vllm" in provider:
            max_out = getattr(self.cfg.llm, "max_output_tokens",
                              getattr(self.cfg.llm, "max_tokens", 2048))

            console.print(
                f"[dim]Generating with vLLM [cyan]{self.cfg.llm.base_model}[/] "
                f"(max_tokens={max_out}, temp={self.cfg.llm.temperature})[/]"
            )

            resp = model.chat.completions.create(
                model=self.cfg.llm.base_model,
                messages=[
                    {"role": "system", "content": self.cfg.llm.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.cfg.llm.temperature,
                max_tokens=max_out,
            )
            return resp.choices[0].message.content.strip()

        # -----------------------------
        # Hugging Face-style generation
        # -----------------------------
        else:
            from transformers import TextStreamer

            max_new = getattr(self.cfg.llm, "max_output_tokens", getattr(self.cfg.llm, "max_tokens", 2048))
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
                    super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
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
                f"({n_tokens/elapsed:.1f} tok/s)[/]"
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
        new_entries = all_history[self._merged_history_count:]

        for entry in new_entries:
            # Skip our own entries (already in feedback.history)
            if entry.get("client_id") == self.client_id:
                continue

            self.feedback.history.append({
                "iteration": entry["iteration"],
                "results": entry["results"],
                "client_id": entry.get("client_id"),
            })

        self._merged_history_count = len(all_history)

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
        for it in range(self.cfg.loop.max_iterations):
            console.rule(f"[bold]Iteration {it}")

            stop_iterations = False  # ✅ reset each iteration

            # --- Sync from shared registry (distributed mode) ---
            self._sync_from_registry()

            feedback = ""
            if self.best_model is not None:
                feedback = self.feedback.get_feedback(
                    self.best_model, self.tried_param_sets,
                    id_results=self.best_id_results
                )

                # Save feedback for inspection
                tag = self._file_tag()
                feedback_file = (
                    self.results_dir / "feedback" / f"iter{it}{tag}_run{run_idx}.txt"
                    if getattr(self.cfg.evaluation, "fit_type", "group") != "individual"
                    else self.results_dir / "feedback" / f"iter{it}{tag}_run{run_idx}_participant{self.df.participant[0]}.txt"
                )
                with open(feedback_file, "w") as f:
                    f.write(feedback)

            prompt = self.prompt_builder.build_input_prompt(feedback_text=feedback)
            code_text = self.generate(self.model, self.tokenizer, prompt)

            tag = self._file_tag()
            model_file = (
                self.results_dir / "models" / f"iter{it}{tag}_run{run_idx}.txt"
                if getattr(self.cfg.evaluation, "fit_type", "group") != "individual"
                else self.results_dir / "models" / f"iter{it}{tag}_run{run_idx}_participant{self.df.participant[0]}.txt"
            )

            with open(model_file, "w") as f:
                f.write(code_text)

            iteration_results = []

            for i in range(1, self.cfg.llm.models_per_iteration + 1):
                func_name = f"cognitive_model{i}"
                func_code = extract_model_code(code_text, i)

                if not func_code:
                    continue

                try:
                    fit_res = run_fit(self.df, func_code, cfg=self.cfg, expected_func_name=func_name)

                    mean_metric = float(fit_res["metric_value"])
                    metric_name = fit_res["metric_name"]
                    params = fit_res["param_names"]
                    self.tried_param_sets.append(params)

                    console.print(f"  [bold]{func_name}[/]: mean {metric_name} = [cyan]{mean_metric:.2f}[/]")

                    # --- Individual differences evaluation (optional) ---
                    id_results = None
                    if self.id_eval_data is not None:
                        try:
                            from gecco.offline_evaluation.individual_differences import evaluate_individual_differences
                            id_results = evaluate_individual_differences(
                                fit_res, self.df, self.cfg, id_data=self.id_eval_data
                            )
                        except Exception as e:
                            console.print(f"  [yellow]Individual differences eval failed for {func_name}:[/] {e}")

                    iteration_results.append({
                        "function_name": func_name,
                        "metric_name": metric_name,
                        "metric_value": mean_metric,
                        "param_names": params,
                        "code_file": str(model_file),
                        "individual_differences": id_results,
                        "code": func_code,
                    })

                    if mean_metric < self.best_metric:
                        self.best_metric = mean_metric
                        self.best_model = func_code
                        self.best_iter = it
                        self.best_params = params
                        self.best_param_names = fit_res["param_names"]
                        self.best_param_values = fit_res["parameter_values"]
                        self.best_id_results = id_results
                        console.print(f"  [bold green]New best model:[/] {func_name} ({metric_name}={mean_metric:.2f})")

                        best_model_file = (
                            self.results_dir / "models" / f"best_model{tag}_{run_idx}.txt"
                            if getattr(self.cfg.evaluation, "fit_type", "group") != "individual"
                            else self.results_dir / "models" / f"best_model{tag}_{run_idx}_participant{self.df.participant[0]}.txt"
                        )
                        with open(best_model_file, "w") as f:
                            f.write(func_code)
                        
                        # save best model bic
                        best_bic_file = (
                            self.results_dir / "bics" / f"best_bic{tag}_{run_idx}.json"
                            if getattr(self.cfg.evaluation, "fit_type", "group") != "individual"
                            else self.results_dir / "bics" / f"best_bic{tag}_{run_idx}_participant{self.df.participant[0]}.json"
                        )
                        with open(best_bic_file, "w") as f:
                            json.dump({"bic": mean_metric}, f)

                    # ✅ stop if ANY model beats baseline
                    if baseline_bic is not None and mean_metric < baseline_bic:
                        stop_iterations = True
                        # optional: break here to save compute evaluating remaining models
                        break

                except Exception as e:
                    console.print(f"  [bold red]Error fitting {func_name}:[/] {e}")

            # ✅ always save what happened this iteration (even if stopping)
            bic_file = (
                self.results_dir / "bics" / f"iter{it}{tag}_run{run_idx}.json"
                if getattr(self.cfg.evaluation, "fit_type", "group") != "individual"
                else self.results_dir / "bics" / f"iter{it}{tag}_run{run_idx}_participant{self.df.participant[0]}.json"
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
        if self.best_model is not None and self.best_params:

            # if getattr(self.cfg.evaluation, "fit_type", "group") == "individual":
            param_df = pd.DataFrame(
                self.best_param_values,
                columns=self.best_param_names
            )

            param_dir = self.results_dir / "parameters"
            param_dir.mkdir(parents=True, exist_ok=True)

            tag = self._file_tag()
            param_file = (
                param_dir / f"best_params{tag}_run{run_idx}.csv"
                if getattr(self.cfg.evaluation, "fit_type", "group") != "individual"
                else param_dir / f"best_params{tag}_run{run_idx}_participant{self.df.participant[0]}.csv"
            )

            param_df.to_csv(param_file, index=False)

        # --- Mark client complete in shared registry ---
        if self.shared_registry is not None and self.client_id is not None:
            self.shared_registry.mark_complete(self.client_id)

        return self.best_model, self.best_metric, self.best_params

