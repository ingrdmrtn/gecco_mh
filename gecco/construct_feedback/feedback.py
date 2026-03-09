import re
import numpy as np
from datetime import datetime


def _log(msg):
    """Print a timestamped log message."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


class FeedbackGenerator:
    """
    Base feedback handler for guiding the LLM between iterations.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.history = []  # store per-iteration summaries if needed

    def record_iteration(self, iteration_idx, results):
        """
        Store results from a given iteration (list of dicts with param_names, etc.)
        """
        self.history.append({
            "iteration": iteration_idx,
            "results": results,
        })

    # ------------------------------------------------------------------
    # Level 1: BIC trajectory and improvement rate
    # ------------------------------------------------------------------

    def _build_trajectory_summary(self):
        """
        Summarise the best BIC per iteration and compute improvement rate.
        Returns an empty string if fewer than 2 iterations have been recorded.
        """
        if len(self.history) < 2:
            return ""

        iter_bests = []
        for entry in self.history:
            if entry["results"]:
                best_bic = min(r["metric_value"] for r in entry["results"])
                iter_bests.append((entry["iteration"], best_bic))

        if len(iter_bests) < 2:
            return ""

        bics = [b for _, b in iter_bests]
        traj_str = " → ".join(f"iter {i} → {b:.1f}" for i, b in iter_bests)
        lines = [f"BIC trajectory: {traj_str}."]

        n = len(bics)
        half = max(1, n // 2)
        early_drop = bics[0] - bics[half - 1]
        early_rate = early_drop / half

        recent_drop = bics[half] - bics[-1] if n > half else 0.0
        recent_iters = n - half if n > half else 1
        recent_rate = recent_drop / recent_iters

        lines.append(
            f"Rate of improvement: {early_rate:.2f} BIC/iter early, "
            f"{recent_rate:.2f} BIC/iter recently."
        )

        if early_rate > 0 and recent_rate < 0.3 * early_rate:
            lines.append(
                "Improvement has stalled — consider exploring structurally "
                "different mechanisms rather than refining existing ones."
            )
        elif recent_rate >= early_rate:
            lines.append(
                "Improvement is still strong — current direction is productive."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Level 2: BIC landscape — ranked table + parameter importance
    # ------------------------------------------------------------------

    def _count_clients(self):
        """Count the number of distinct clients contributing to history."""
        client_ids = set()
        for entry in self.history:
            cid = entry.get("client_id")
            if cid is not None:
                client_ids.add(cid)
        return len(client_ids)

    def _build_landscape_summary(self):
        """
        Build a ranked table of all models across all iterations, flag
        convergence zones, and identify which parameter types are
        associated with good vs poor BIC.
        """
        if not self.history:
            return ""

        # Collect every model from every iteration
        all_models = []
        for entry in self.history:
            for r in entry["results"]:
                all_models.append({
                    "name": r["function_name"],
                    "bic": r["metric_value"],
                    "params": r["param_names"],
                    "iter": entry["iteration"],
                    "client_id": entry.get("client_id"),
                })

        if not all_models:
            return ""

        all_models.sort(key=lambda x: x["bic"])

        # --- Ranked table (top 10) ---
        n_clients = self._count_clients()
        client_note = f" from {n_clients} parallel search clients" if n_clients > 1 else ""
        lines = [f"Model landscape (all iterations{client_note}, ranked by BIC):"]
        lines.append(f"{'Model':<22} {'BIC':>8}  {'Params':<45}  Iter")
        lines.append("-" * 82)
        for m in all_models[:10]:
            param_str = "[" + ", ".join(m["params"]) + "]"
            lines.append(
                f"{m['name']:<22} {m['bic']:>8.1f}  {param_str:<45}  {m['iter']}"
            )

        # --- Convergence zones ---
        # Pairs of top-6 models with similar BIC but different parameter sets
        bic_threshold = 5.0
        convergence_notes = []
        top_n = min(6, len(all_models))
        for i in range(top_n):
            for j in range(i + 1, top_n):
                a, b = all_models[i], all_models[j]
                if abs(a["bic"] - b["bic"]) < bic_threshold:
                    params_a = set(a["params"])
                    params_b = set(b["params"])
                    if params_a != params_b:
                        only_a = params_a - params_b
                        only_b = params_b - params_a
                        bic_diff = abs(a["bic"] - b["bic"])
                        convergence_notes.append(
                            f"'{a['name']}' (BIC={a['bic']:.1f}, params={sorted(params_a)}) "
                            f"and '{b['name']}' (BIC={b['bic']:.1f}, params={sorted(params_b)}) "
                            f"differ by only {bic_diff:.1f} BIC. "
                            f"Params unique to each: {sorted(only_a) or '{}'} vs {sorted(only_b) or '{}'}. "
                            f"These extra parameters may not be explaining meaningful variance."
                        )

        if convergence_notes:
            lines.append("\nConvergence zones (similar BIC, different structures):")
            for note in convergence_notes[:3]:
                lines.append(f"  • {note}")

        # --- Parameter importance ---
        # Which params appear more often in the top half vs bottom half?
        if len(all_models) >= 4:
            median_idx = len(all_models) // 2
            good_models = all_models[:median_idx]
            poor_models = all_models[median_idx:]

            n_good = len(good_models)
            n_poor = len(poor_models)

            good_counts: dict = {}
            poor_counts: dict = {}
            for m in good_models:
                for p in m["params"]:
                    good_counts[p] = good_counts.get(p, 0) + 1
            for m in poor_models:
                for p in m["params"]:
                    poor_counts[p] = poor_counts.get(p, 0) + 1

            all_param_names = set(good_counts) | set(poor_counts)
            signals = []
            for p in all_param_names:
                g_rate = good_counts.get(p, 0) / n_good
                po_rate = poor_counts.get(p, 0) / n_poor
                if g_rate - po_rate >= 0.3:
                    signals.append((p, g_rate, po_rate, "good"))
                elif po_rate - g_rate >= 0.3:
                    signals.append((p, g_rate, po_rate, "poor"))

            if signals:
                signals.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)
                lines.append("\nParameter importance signals:")
                for p, g, po, direction in signals[:5]:
                    if direction == "good":
                        lines.append(
                            f"  • '{p}' appears in {g*100:.0f}% of top models vs "
                            f"{po*100:.0f}% of weaker models — likely a critical mechanism."
                        )
                    else:
                        lines.append(
                            f"  • '{p}' appears in {po*100:.0f}% of weaker models vs "
                            f"{g*100:.0f}% of top models — may not be capturing meaningful variance."
                        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Level 3: Stagnation detection, factor extraction, top-N code
    # ------------------------------------------------------------------

    def _detect_stagnation(self, threshold=2.0, window=3):
        """
        Check whether the search has stagnated.
        Returns "exploring" if best BIC hasn't improved by more than
        *threshold* over the last *window* iterations, "exploiting" otherwise.
        """
        if len(self.history) < window:
            return "exploiting"

        recent = self.history[-window:]
        best_bics = []
        for entry in recent:
            if entry["results"]:
                best_bics.append(min(r["metric_value"] for r in entry["results"]))

        if len(best_bics) < 2:
            return "exploiting"

        improvement = best_bics[0] - best_bics[-1]
        return "exploring" if improvement < threshold else "exploiting"

    def _extract_factor_usage(self, code):
        """
        Parse model code to identify which psychiatric factors are used.
        Returns a list of factor names found (e.g. ["ad", "cit"] or ["stai"]).
        """
        factors = {
            "ad": r"\bself\.ad\b",
            "cit": r"\bself\.cit\b",
            "sw": r"\bself\.sw\b",
            "stai": r"\bstai\b",
            "oci": r"\boci\b",
        }
        found = []
        for name, pattern in factors.items():
            if re.search(pattern, code):
                found.append(name)
        return found

    def _get_top_n_code(self, n=3):
        """
        Return the code of the top *n* models (by BIC) from history.
        Returns list of (name, bic, code, factors_used) tuples.
        """
        all_models = []
        for entry in self.history:
            for r in entry["results"]:
                code = r.get("code", "")
                if code:
                    all_models.append({
                        "name": r["function_name"],
                        "bic": r["metric_value"],
                        "code": code,
                        "iter": entry["iteration"],
                    })

        if not all_models:
            return []

        all_models.sort(key=lambda x: x["bic"])
        results = []
        for m in all_models[:n]:
            factors = self._extract_factor_usage(m["code"])
            results.append((m["name"], m["bic"], m["code"], factors))
        return results

    def _build_factor_coverage(self):
        """
        Build a summary of which psychiatric factor-mechanism pairings
        have been tried across all models. Only produces output if
        factor usage is detected.
        """
        all_factor_usage = []
        for entry in self.history:
            for r in entry["results"]:
                code = r.get("code", "")
                if code:
                    factors = self._extract_factor_usage(code)
                    if factors:
                        all_factor_usage.append({
                            "name": r["function_name"],
                            "bic": r["metric_value"],
                            "factors": factors,
                            "iter": entry["iteration"],
                        })

        if not all_factor_usage:
            return ""

        lines = ["Psychiatric factor usage across models:"]
        # Count how often each factor appears and its average BIC
        factor_stats = {}
        for m in all_factor_usage:
            for f in m["factors"]:
                if f not in factor_stats:
                    factor_stats[f] = {"count": 0, "bics": []}
                factor_stats[f]["count"] += 1
                factor_stats[f]["bics"].append(m["bic"])

        for f, stats in sorted(factor_stats.items()):
            avg_bic = sum(stats["bics"]) / len(stats["bics"])
            best_bic = min(stats["bics"])
            lines.append(
                f"  - '{f}': used in {stats['count']} model(s), "
                f"avg BIC = {avg_bic:.1f}, best BIC = {best_bic:.1f}"
            )

        # Flag factors never used
        used = set(factor_stats.keys())
        # Only flag missing factors from the set that appears at least once
        # (e.g. if stai is used, don't flag missing ad/cit/sw since it's a different config)
        if used & {"ad", "cit", "sw"}:
            missing = {"ad", "cit", "sw"} - used
            if missing:
                lines.append(
                    f"  - Untried factors: {sorted(missing)} — consider models using these."
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Level 4: Per-participant fit quality
    # ------------------------------------------------------------------

    def _build_fit_quality_summary(self):
        """
        Analyse per-participant fit metrics for the best model in the most
        recent iteration. Flags outliers and reports distribution stats.
        """
        # Find the best model's eval_metrics from the most recent iteration
        if not self.history:
            return ""

        # Get eval_metrics from the best model across all history
        best_metrics = None
        best_bic = float("inf")
        best_name = ""
        for entry in self.history:
            for r in entry["results"]:
                metrics = r.get("eval_metrics", [])
                if metrics and r["metric_value"] < best_bic:
                    best_bic = r["metric_value"]
                    best_metrics = metrics
                    best_name = r["function_name"]

        if not best_metrics or len(best_metrics) < 2:
            return ""

        metrics = np.array(best_metrics)
        mean_val = np.mean(metrics)
        std_val = np.std(metrics)
        min_val = np.min(metrics)
        max_val = np.max(metrics)
        n_total = len(metrics)

        # Outliers: participants with BIC > mean + 2*std
        outlier_threshold = mean_val + 2 * std_val
        n_outliers = int(np.sum(metrics > outlier_threshold))
        pct_outliers = 100 * n_outliers / n_total

        # Well-fit: participants with BIC below median
        median_val = np.median(metrics)

        lines = [
            f"Participant fit quality for best model ({best_name}):",
            f"  Mean BIC: {mean_val:.1f} (std: {std_val:.1f}, min: {min_val:.1f}, max: {max_val:.1f})",
            f"  Median BIC: {median_val:.1f}",
            f"  Poorly-fit participants (BIC > mean + 2σ): {n_outliers}/{n_total} ({pct_outliers:.1f}%)",
        ]

        # Report the spread — large std relative to mean indicates heterogeneity
        if std_val > 0.5 * mean_val:
            lines.append(
                "  High variability in fit quality across participants — "
                "the model may not capture an important source of individual differences."
            )

        return "\n".join(lines)

    def _build_high_level_summary(self, id_results=None):
        """
        Build a concise 2-3 line summary of the search state.
        """
        if not self.history:
            return ""

        # Best model info
        best_bic = float("inf")
        best_name = ""
        best_iter = -1
        for entry in self.history:
            for r in entry["results"]:
                if r["metric_value"] < best_bic:
                    best_bic = r["metric_value"]
                    best_name = r["function_name"]
                    best_iter = entry["iteration"]

        if best_bic == float("inf"):
            return ""

        mode = self._detect_stagnation()
        trend = "improving" if mode == "exploiting" else "stagnating"

        n_clients = self._count_clients()
        client_note = f" across {n_clients} clients" if n_clients > 1 else ""
        lines = [f"Best BIC: {best_bic:.1f} ({best_name}, iter {best_iter}{client_note}). Trend: {trend}."]

        # ID summary
        if id_results is not None:
            mean_r2 = id_results.get("mean_r2", 0.0)
            detail = id_results.get("per_param_detail", {})
            # Find the strongest predictor across all params
            best_pred = ""
            best_beta = 0.0
            best_param = ""
            for pname, info in detail.items():
                for pred, coef in info.get("coefficients", {}).items():
                    if abs(coef) > abs(best_beta):
                        best_beta = coef
                        best_pred = pred
                        best_param = pname
            if best_pred:
                lines.append(
                    f"Individual differences: mean R² = {mean_r2:.3f} "
                    f"(strongest predictor: {best_pred} → {best_param}, β={best_beta:.3f})."
                )
            else:
                lines.append(f"Individual differences: mean R² = {mean_r2:.3f}.")

        # Fit quality summary
        fit_quality = self._build_fit_quality_summary()
        if fit_quality:
            # Extract just the outlier line
            for line in fit_quality.split("\n"):
                if "Poorly-fit" in line:
                    lines.append(line.strip())
                    break

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_feedback(self, best_model, tried_param_sets, id_results=None):
        """
        Construct feedback string for the next prompt.
        Includes high-level summary, BIC trajectory, landscape, fit quality,
        individual differences detail, and stagnation signal.
        """
        previous_parameters = "\n".join(
            [", ".join(s) for s in tried_param_sets]
        )

        # --- High-level summary ---
        parts = []
        summary = self._build_high_level_summary(id_results=id_results)
        if summary:
            parts.append(f"## Summary\n{summary}")

        # --- Detailed breakdown ---

        # BIC trajectory
        trajectory = self._build_trajectory_summary()
        if trajectory:
            parts.append(trajectory)

        # Model landscape
        landscape = self._build_landscape_summary()
        if landscape:
            parts.append(landscape)

        # Participant fit quality
        fit_quality = self._build_fit_quality_summary()
        if fit_quality:
            parts.append(fit_quality)

        # Individual differences
        if id_results is not None:
            parts.append(
                f"Individual Differences Analysis:\n"
                f"{id_results['summary_text']}\n\n"
                "When proposing new models, consider whether parameters could better capture "
                "individual variation in these questionnaire measures. The primary objective "
                "remains minimising BIC (model fit), but higher R² for individual differences "
                "is also desirable."
            )

        # Factor coverage
        factor_coverage = self._build_factor_coverage()
        if factor_coverage:
            parts.append(factor_coverage)

        # Stagnation signal
        mode = self._detect_stagnation()
        if mode == "exploring":
            parts.append(
                "Search status: STAGNATING. BIC has not improved meaningfully "
                "in recent iterations. Try structurally different mechanisms "
                "rather than refining the current approach."
            )
        elif len(self.history) >= 3:
            parts.append(
                "Search status: IMPROVING. Current direction is productive — "
                "refine and build on the best model's mechanisms."
            )

        search_context = ("\n\n".join(parts) + "\n\n") if parts else ""

        # --- Core feedback (custom prompt or default) ---
        if getattr(self.cfg.feedback, "prompt", None):
            core = self.cfg.feedback.prompt.format(
                best_model=best_model,
                previous_parameters=previous_parameters,
            )
        else:
            core = (
                f"## Best Model So Far\n```python\n{best_model}\n```\n\n"
                f"## Parameter Combinations Tested\n{previous_parameters}\n\n"
                "Avoid repeating these exact combinations, "
                "and explore alternative parameter configurations or mechanisms.\n"
            )

        return search_context + core


class LLMFeedbackGenerator(FeedbackGenerator):
    """
    Optional subclass: let an LLM summarize model search performance and propose directions.
    """

    def __init__(self, cfg, model, tokenizer):
        super().__init__(cfg)
        self.model = model
        self.tokenizer = tokenizer

    def get_feedback(self, best_model, tried_param_sets, id_results=None):
        """
        Construct feedback using an LLM judge that analyses the full
        search landscape and provides data-driven guidance.
        """
        # --- Build landscape context (Levels 1-3) ---
        context_parts = []

        trajectory = self._build_trajectory_summary()
        if trajectory:
            context_parts.append(f"## Search Trajectory\n{trajectory}")

        landscape = self._build_landscape_summary()
        if landscape:
            context_parts.append(f"## Model Landscape\n{landscape}")

        # --- Participant fit quality ---
        fit_quality = self._build_fit_quality_summary()
        if fit_quality:
            context_parts.append(f"## Participant Fit Quality\n{fit_quality}")

        factor_coverage = self._build_factor_coverage()
        if factor_coverage:
            context_parts.append(f"## Psychiatric Factor Usage\n{factor_coverage}")

        mode = self._detect_stagnation()
        context_parts.append(f"## Search Status\nMode: {mode.upper()}")

        # --- Top N model code ---
        top_models = self._get_top_n_code(n=3)
        if top_models:
            code_sections = []
            for name, bic, code, factors in top_models:
                factor_str = f" (factors used: {', '.join(factors)})" if factors else ""
                code_sections.append(
                    f"### {name} (BIC = {bic:.1f}){factor_str}\n```python\n{code}\n```"
                )
            context_parts.append(
                "## Top Model Implementations\n" + "\n\n".join(code_sections)
            )

        # --- Best model ---
        context_parts.append(f"## Current Best Model\n```python\n{best_model}\n```")

        # --- Individual differences (with per-predictor detail) ---
        if id_results is not None:
            context_parts.append(
                f"## Individual Differences Analysis\n"
                f"{id_results['summary_text']}\n\n"
                "Consider whether model parameters could better capture "
                "individual variation in these questionnaire measures. "
                "The primary objective remains minimising BIC, but higher R² "
                "for individual differences is also desirable."
            )

        # --- Multi-client context ---
        n_clients = self._count_clients()
        if n_clients > 1:
            context_parts.insert(0,
                f"## Distributed Search\n"
                f"This search is running across {n_clients} parallel clients, "
                f"each exploring different model architectures. The data below "
                f"aggregates results from all clients."
            )

        search_context = "\n\n".join(context_parts)

        # --- Judge prompt ---
        judge_prompt = (
            "You are analysing a computational model search. Be purely data-driven — "
            "report what the BIC values show. Do NOT interpret mechanisms based on "
            "literature or prior knowledge. The goal is novel discovery.\n\n"
            f"{search_context}\n\n"
            "Based on the data above:\n"
            "1. Which parameters or mechanisms consistently appear in well-fitting models?\n"
            "2. Which parameters add complexity without meaningfully improving BIC?\n"
            "3. What parameter or mechanism combinations have not been tried yet?\n"
            "4. Are there patterns in which participants fit poorly? Does the model "
            "struggle with a specific subset of participants?\n"
            "5. Which symptom dimensions (questionnaire factors) most strongly predict "
            "model parameters, and what does this suggest about individual differences?\n"
            "6. Is improvement stagnating? If so, suggest structurally different directions "
            "to explore.\n"
            "7. Provide specific, actionable guidance for generating the next set of models.\n\n"
            "Keep your response concise and focused on actionable next steps. "
            "Do NOT include Python code or pseudocode — describe mechanisms conceptually only."
        )

        _log("[GeCCo] Sending landscape data to judge LLM for feedback")
        feedback_text = self.generate(judge_prompt)
        return feedback_text.strip()

    def generate(self, prompt):
        """
        Unified text generation function for any supported backend.
        """
        if self.model is None:
            raise ValueError("Model not initialized correctly.")
        provider = self.cfg.llm.provider.lower()

        # -----------------------------
        # OpenAI / GPT-style generation
        # -----------------------------
        if "openai" in provider or "gpt" in provider:
            max_out = self.cfg.llm.max_output_tokens
            reasoning_effort = getattr(self.cfg.llm, "reasoning_effort", "medium")
            text_verbosity = getattr(self.cfg.llm, "text_verbosity", "low")

            _log(
                f"[GeCCo] Using GPT model '{self.cfg.llm.base_model}' "
                f"(reasoning={reasoning_effort}, verbosity={text_verbosity}, max_output_tokens={max_out})"
            )

            resp = self.model.responses.create(
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

            print(
                f"[GeCCo] Using Gemini model '{self.cfg.llm.base_model}' "
                f"(reasoning={reasoning_effort})"
            )
            resp = self.model.models.generate_content(
                model=self.cfg.llm.base_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                       temperature=self.cfg.llm.temperature,
                       system_instruction=self.cfg.llm.system_prompt,
                       thinking_config=types.ThinkingConfig(thinking_level=reasoning_effort),
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

            _log(
                f"[GeCCo] Using vLLM model '{self.cfg.llm.base_model}' "
                f"(max_tokens={max_out}, temperature={self.cfg.llm.temperature})"
            )

            resp = self.model.chat.completions.create(
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
            max_new = getattr(self.cfg.llm, "max_output_tokens", getattr(self.cfg.llm, "max_tokens", 2048))

            _log(
                f"[GeCCo] Using HF model '{self.cfg.llm.base_model}' "
                f"(max_new_tokens={max_new}, temperature={self.cfg.llm.temperature})"
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=self.cfg.llm.temperature,
                do_sample=True,
            )
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
