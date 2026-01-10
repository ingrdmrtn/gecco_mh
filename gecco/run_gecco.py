# engine/model_search.py
import os
import json
import numpy as np
import pandas as pd

from gecco.offline_evaluation.fit_generated_models import run_fit
from gecco.utils import extract_model_code
from gecco.construct_feedback.feedback import FeedbackGenerator, LLMFeedbackGenerator
from pathlib import Path
from google.genai import types

class GeCCoModelSearch:
    def __init__(self, model, tokenizer, cfg, df, prompt_builder):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.df = df
        self.prompt_builder = prompt_builder

        # --- Choose feedback generator based on config ---
        if hasattr(cfg, "feedback") and getattr(cfg.feedback, "type", "manual") == "llm":
            self.feedback = LLMFeedbackGenerator(cfg, model, tokenizer)
        else:
            self.feedback = FeedbackGenerator(cfg)

        # --- Set project root ---
        self.project_root = Path(__file__).resolve().parents[1]

        # --- Results directory (absolute path) ---
        # self.results_dir = self.project_root / "results" / self.cfg.task.name if self.cfg.evaluation.fit_type != "individual" else self.project_root / "results" / self.cfg.task.name + "_individual"

        self.results_dir = (
            self.project_root / "results" / self.cfg.task.name
            if self.cfg.evaluation.fit_type != "individual"
            else self.project_root / "results" / f"{self.cfg.task.name}_individual"
        )

        (self.results_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "bics").mkdir(parents=True, exist_ok=True)

        # --- Tracking ---
        self.best_model = None
        self.best_metric = np.inf
        self.best_params = []
        self.best_iter = -1
        self.tried_param_sets = []

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

            print(
                f"[GeCCo] Using GPT model '{self.cfg.llm.base_model}' "
                f"(reasoning={reasoning_effort}, verbosity={text_verbosity}, max_output_tokens={max_out})"
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
            reasoning_effort = getattr(self.cfg.llm, "reasoning_effort", "low")

            print(
                f"[GeCCo] Using Gemini model '{self.cfg.llm.base_model}' "
                f"(reasoning={reasoning_effort})"
            )

            config_args = {
                "temperature": self.cfg.llm.temperature,
                "system_instruction": self.cfg.llm.system_prompt,
            }

            if reasoning_effort:
                assert reasoning_effort in ["low", "medium", "high"], \
                    f"Invalid reasoning_effort: {reasoning_effort}. Choose from 'low', 'medium', 'high', or False."
                assert "flash" not in self.cfg.llm.base_model.lower(), \
                    "Reasoning effort is not supported for Gemini Flash models."
                config_args["thinking_config"] = types.ThinkingConfig(
                    thinking_level=reasoning_effort
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
        # Hugging Face-style generation
        # -----------------------------
        else:
            max_new = getattr(self.cfg.llm, "max_output_tokens", getattr(self.cfg.llm, "max_tokens", 2048))

            print(
                f"[GeCCo] Using HF model '{self.cfg.llm.base_model}' "
                f"(max_new_tokens={max_new}, temperature={self.cfg.llm.temperature})"
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output = model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=self.cfg.llm.temperature,
                do_sample=True,
            )
            return tokenizer.decode(output[0], skip_special_tokens=True)

    def run_n_shots(self, run_idx, baseline_bic):
        for it in range(self.cfg.loop.max_iterations):
            print(f"\n[GeCCo] --- Iteration {it} ---")

            stop_iterations = False  # ✅ reset each iteration

            feedback = ""
            if self.best_model is not None:
                feedback = self.feedback.get_feedback(self.best_model, self.tried_param_sets)

            prompt = self.prompt_builder.build_input_prompt(feedback_text=feedback)
            code_text = self.generate(self.model, self.tokenizer, prompt)

            model_file = (
                self.results_dir / "models" / f"iter{it}_run{run_idx}.txt"
                if self.cfg.evaluation.fit_type != "individual"
                else self.results_dir / "models" / f"iter{it}_run{run_idx}_participant{self.df.participant[0]}.txt"
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

                    print(f"[GeCCo] {func_name}: mean {metric_name} = {mean_metric:.2f}")

                    iteration_results.append({
                        "function_name": func_name,
                        "metric_name": metric_name,
                        "metric_value": mean_metric,
                        "param_names": params,
                        "code_file": str(model_file),
                    })

                    if mean_metric < self.best_metric:
                        self.best_metric = mean_metric
                        self.best_model = func_code
                        self.best_iter = it
                        self.best_params = params
                        self.best_param_names = fit_res["param_names"]  # ✅ store names
                        self.best_param_values = fit_res["parameter_values"]
                        print(f"[⭐ GeCCo] New best model: {func_name} ({metric_name}={mean_metric:.2f})")

                        best_model_file = (
                            self.results_dir / "models" / f"best_model_{run_idx}.txt"
                            if self.cfg.evaluation.fit_type != "individual"
                            else self.results_dir / "models" / f"best_model_{run_idx}_participant{self.df.participant[0]}.txt"
                        )
                        with open(best_model_file, "w") as f:
                            f.write(func_code)
                        
                        # save best model bic
                        best_bic_file = (
                            self.results_dir / "bics" / f"best_bic_{run_idx}.json"
                            if self.cfg.evaluation.fit_type != "individual"
                            else self.results_dir / "bics" / f"best_bic_{run_idx}_participant{self.df.participant[0]}.json"
                        )
                        with open(best_bic_file, "w") as f:
                            json.dump({"bic": mean_metric}, f)

                    # ✅ stop if ANY model beats baseline
                    if baseline_bic is not None and mean_metric < baseline_bic:
                        stop_iterations = True
                        # optional: break here to save compute evaluating remaining models
                        break

                except Exception as e:
                    print(f"[⚠️ GeCCo] Error fitting {func_name}: {e}")

            # ✅ always save what happened this iteration (even if stopping)
            bic_file = (
                self.results_dir / "bics" / f"iter{it}_run{run_idx}.json"
                if self.cfg.evaluation.fit_type != "individual"
                else self.results_dir / "bics" / f"iter{it}_run{run_idx}_participant{self.df.participant[0]}.json"
            )
            with open(bic_file, "w") as f:
                json.dump(iteration_results, f, indent=2)

            self.feedback.record_iteration(it, iteration_results)

            if stop_iterations:
                break  # ✅ breaks iterations loop only (run loop unaffected)

        print(
            f"\n[🏁 GeCCo] Finished search. "
            f"Best model (iteration {self.best_iter}) "
            f"{self.cfg.evaluation.metric.upper()}={self.best_metric:.2f}"
        )

        # --- save best parameters ---
        if self.best_model is not None and self.best_params:

            # if self.cfg.evaluation.fit_type == "individual":
            param_df = pd.DataFrame(
                self.best_param_values,
                columns=self.best_param_names
            )

            param_dir = self.results_dir / "parameters"
            param_dir.mkdir(parents=True, exist_ok=True)

            param_file = (
                param_dir / f"best_params_run{run_idx}.csv"
                if self.cfg.evaluation.fit_type != "individual"
                else param_dir / f"best_params_run{run_idx}_participant{self.df.participant[0]}.csv"
            )

            param_df.to_csv(param_file, index=False)


        return self.best_model, self.best_metric, self.best_params

