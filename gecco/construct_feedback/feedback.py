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
                })

        if not all_models:
            return ""

        all_models.sort(key=lambda x: x["bic"])

        # --- Ranked table (top 10) ---
        lines = ["Model landscape (all iterations, ranked by BIC):"]
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
    # Public interface
    # ------------------------------------------------------------------

    def get_feedback(self, best_model, tried_param_sets):
        """
        Construct feedback string for the next prompt.
        Prepends Level 1 (BIC trajectory) and/or Level 2 (landscape summary)
        context depending on cfg.feedback.level (default: 2).

        level=0 — no search-history context
        level=1 — BIC trajectory only
        level=2 — BIC trajectory + ranked model landscape
        """
        level = int(getattr(self.cfg.feedback, "level", 2))

        previous_parameters = "\n".join(
            [", ".join(s) for s in tried_param_sets]
        )

        # --- Search-history context block ---
        context_parts = []
        if level >= 1:
            trajectory = self._build_trajectory_summary()
            if trajectory:
                context_parts.append(trajectory)
        if level >= 2:
            landscape = self._build_landscape_summary()
            if landscape:
                context_parts.append(landscape)
        search_context = ("\n\n".join(context_parts) + "\n\n") if context_parts else ""

        # --- Core feedback (custom prompt or default) ---
        if getattr(self.cfg.feedback, "prompt", None):
            core = self.cfg.feedback.prompt.format(
                best_model=best_model,
                previous_parameters=previous_parameters,
            )
        else:
            core = (
                f"Your best model so far:\n  {best_model}.\n"
                f"The parameter combinations tried so far:\n{previous_parameters}\n\n"
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

    def get_feedback(self, best_model, tried_param_sets):
        """
        Construct feedback string for the next prompt using an LLM.
        Level 1 + 2 context is prepended to the LLM prompt so the LLM
        can reason over the full search history.

        level=0 — no search-history context
        level=1 — BIC trajectory only
        level=2 — BIC trajectory + ranked model landscape
        """
        level = int(getattr(self.cfg.feedback, "level", 2))

        previous_parameters = "\n".join(
            [", ".join(s) for s in tried_param_sets]
        )

        # --- Search-history context block ---
        context_parts = []
        if level >= 1:
            trajectory = self._build_trajectory_summary()
            if trajectory:
                context_parts.append(trajectory)
        if level >= 2:
            landscape = self._build_landscape_summary()
            if landscape:
                context_parts.append(landscape)
        search_context = ("\n\n".join(context_parts) + "\n\n") if context_parts else ""

        if getattr(self.cfg.feedback, "prompt", None) is not None:
            core_prompt = self.cfg.feedback.prompt.format(
                best_model=best_model,
                previous_parameters=previous_parameters,
            )
        else:
            core_prompt = (
                f"The best model so far was:\n  {best_model}.\n"
                f"The following parameter combinations have already been explored:\n"
                f"{previous_parameters}\n\n"
                "Please suggest high-level guidance for generating new model variants "
                "that differ conceptually but might still perform well."
            )

        prompt = search_context + core_prompt
        feedback_text = self.generate(prompt)
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
