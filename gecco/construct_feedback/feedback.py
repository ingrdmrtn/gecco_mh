import math
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
            valid_results = [
                r for r in entry["results"]
                if r.get("metric_name") != "RECOVERY_FAILED"
            ]
            if valid_results:
                best_bic = min(r["metric_value"] for r in valid_results)
                iter_bests.append((entry["iteration"], best_bic))

        if len(iter_bests) < 2:
            return ""

        bics = [b for _, b in iter_bests]
        n_iters = len(iter_bests)
        traj_str = " → ".join(f"iter {i} → {b:.1f}" for i, b in iter_bests)
        lines = []
        if n_iters <= 3:
            lines.append(f"(Based on only {n_iters} iterations — trend estimates are preliminary.)")
        lines.append(f"BIC trajectory: {traj_str}.")

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
        recovery_failures = []
        for entry in self.history:
            for r in entry["results"]:
                if r.get("metric_name") == "RECOVERY_FAILED":
                    recovery_failures.append({
                        "name": r["function_name"],
                        "recovery_r": r.get("recovery_r", 0.0),
                        "params": r.get("param_names", []),
                        "iter": entry["iteration"],
                    })
                    continue
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
        lines = [f"Model landscape ({len(all_models)} models across all iterations{client_note}, ranked by BIC):"]
        if len(all_models) < 10:
            lines.append("(Few models evaluated so far — rankings may shift substantially.)")
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

        # --- Recovery failures ---
        if recovery_failures:
            lines.append(
                f"\nParameter recovery failures: {len(recovery_failures)} model(s) rejected "
                f"due to poor parameter identifiability:"
            )
            for rf in recovery_failures[:5]:
                param_str = "[" + ", ".join(rf["params"]) + "]"
                lines.append(
                    f"  • {rf['name']} (r={rf['recovery_r']:.2f}, params={param_str}, iter {rf['iter']})"
                )
            lines.append(
                "  Avoid similar parameter structures — these models have redundant "
                "or unidentifiable parameters."
            )

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

    def _build_search_momentum(self, window=3):
        """
        Compute recent BIC improvement over the last *window* iterations.
        Returns a short descriptive string with the raw numbers,
        or empty string if not enough data.
        """
        if len(self.history) < 2:
            return ""

        recent = self.history[-window:]
        best_bics = []
        for entry in recent:
            valid = [r for r in entry["results"] if r.get("metric_name") != "RECOVERY_FAILED"]
            if valid:
                best_bics.append(min(r["metric_value"] for r in valid))

        if len(best_bics) < 2:
            return ""

        improvement = best_bics[0] - best_bics[-1]
        n = len(best_bics)
        rate = improvement / (n - 1)

        return (
            f"Recent momentum (last {n} iterations): "
            f"BIC improved by {improvement:.1f} total ({rate:.1f}/iter)."
        )

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
                if r.get("metric_name") == "RECOVERY_FAILED":
                    continue
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

    def _build_r2_landscape(self):
        """
        Analyse which parameter types predict symptoms across all models.

        Mines individual_differences data from history to report:
        - Which parameter names tend to have highest R² for symptom prediction
        - Which symptom predictors are most strongly linked to model parameters
        """
        # Collect (param_name, r2, predictor_coefficients) across all models
        param_r2_records: list[dict] = []
        for entry in self.history:
            for r in entry["results"]:
                id_res = r.get("individual_differences")
                if not id_res or not isinstance(id_res, dict):
                    continue
                per_param_r2 = id_res.get("per_param_r2", {})
                per_param_detail = id_res.get("per_param_detail", {})
                model_name = r.get("function_name", "?")
                bic = r.get("metric_value", float("inf"))
                for pname, r2 in per_param_r2.items():
                    coeffs = per_param_detail.get(pname, {}).get("coefficients", {})
                    param_r2_records.append({
                        "param": pname,
                        "r2": r2,
                        "model": model_name,
                        "bic": bic,
                        "coefficients": coeffs,
                    })

        if len(param_r2_records) < 3:
            return ""

        # Group by parameter name: average R² and best R²
        from collections import defaultdict
        param_stats = defaultdict(lambda: {"r2_values": [], "best_coeffs": {}, "best_r2": 0.0})
        for rec in param_r2_records:
            ps = param_stats[rec["param"]]
            ps["r2_values"].append(rec["r2"])
            if rec["r2"] > ps["best_r2"]:
                ps["best_r2"] = rec["r2"]
                ps["best_coeffs"] = rec["coefficients"]

        # Sort by best R² descending
        ranked = sorted(
            param_stats.items(),
            key=lambda x: x[1]["best_r2"],
            reverse=True,
        )

        lines = ["Cross-model parameter-symptom links (which parameter types predict symptoms):"]

        for pname, stats in ranked[:8]:  # Top 8 parameter types
            avg_r2 = sum(stats["r2_values"]) / len(stats["r2_values"])
            best_r2 = stats["best_r2"]
            n_models = len(stats["r2_values"])

            # Find strongest predictor for this parameter
            best_pred = ""
            best_beta = 0.0
            for pred, coef in stats["best_coeffs"].items():
                if abs(coef) > abs(best_beta):
                    best_beta = coef
                    best_pred = pred

            pred_note = f", strongest predictor: {best_pred} (β={best_beta:.3f})" if best_pred else ""
            lines.append(
                f"  - '{pname}': best R² = {best_r2:.3f}, avg R² = {avg_r2:.3f} "
                f"(across {n_models} model(s){pred_note})"
            )

        # Highlight top finding
        if ranked and ranked[0][1]["best_r2"] > 0.05:
            top_param, top_stats = ranked[0]
            top_pred = ""
            top_beta = 0.0
            for pred, coef in top_stats["best_coeffs"].items():
                if abs(coef) > abs(top_beta):
                    top_beta = coef
                    top_pred = pred
            if top_pred:
                lines.append(
                    f"\nStrongest link found so far: '{top_param}' parameters predict "
                    f"'{top_pred}' (R²={top_stats['best_r2']:.3f}). "
                    f"Consider designing parameters that capture similar mechanisms."
                )

        return "\n".join(lines)

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

    def _count_choice_columns(self):
        """Count the number of choice/action columns in input_columns."""
        choice_keywords = ["choice", "action", "decision"]
        input_cols = getattr(self.cfg.data, "input_columns", [])
        n = sum(
            1 for col in input_cols
            if any(kw in col.lower() for kw in choice_keywords)
        )
        return max(n, 1)

    def _build_fit_quality_summary(self):
        """
        Analyse per-participant fit metrics for the best model.
        Reports fit tiers (well-fit / moderate / poor / at chance)
        and flags participants where the model captures nothing.
        """
        if not self.history:
            return ""

        # Get eval_metrics from the best model across all history
        best_metrics = None
        best_bic = float("inf")
        best_name = ""
        best_n_trials = None
        best_param_names = []
        for entry in self.history:
            for r in entry["results"]:
                metrics = r.get("eval_metrics", [])
                if metrics and r["metric_value"] < best_bic:
                    best_bic = r["metric_value"]
                    best_metrics = metrics
                    best_name = r["function_name"]
                    best_n_trials = r.get("participant_n_trials", [])
                    best_param_names = r.get("param_names", [])

        if not best_metrics or len(best_metrics) < 2:
            return ""

        metrics = np.array(best_metrics)
        mean_val = np.mean(metrics)
        std_val = np.std(metrics)
        min_val = np.min(metrics)
        max_val = np.max(metrics)
        median_val = np.median(metrics)
        n_total = len(metrics)

        # Fit tiers
        outlier_threshold = mean_val + 2 * std_val
        n_well_fit = int(np.sum(metrics <= median_val))
        n_moderate = int(np.sum((metrics > median_val) & (metrics <= outlier_threshold)))
        n_poor = int(np.sum(metrics > outlier_threshold))

        lines = [
            f"Participant fit quality for best model ({best_name}, {len(best_param_names)} params):",
            f"  Mean BIC: {mean_val:.1f} (std: {std_val:.1f}, range: [{min_val:.1f}, {max_val:.1f}])",
            f"  Median BIC: {median_val:.1f}",
            f"  Fit tiers:",
            f"    Well-fit (≤ median):       {n_well_fit}/{n_total} ({100*n_well_fit/n_total:.0f}%)",
            f"    Moderate (median to +2σ):  {n_moderate}/{n_total} ({100*n_moderate/n_total:.0f}%)",
            f"    Poor (> mean + 2σ):        {n_poor}/{n_total} ({100*n_poor/n_total:.0f}%)",
        ]

        # Chance-level detection
        if best_n_trials and len(best_n_trials) == n_total:
            n_choice_cols = self._count_choice_columns()
            chance_bics = np.array([
                2 * n_choice_cols * nt * math.log(2)
                for nt in best_n_trials
            ])
            n_at_chance = int(np.sum(metrics >= chance_bics))
            pct_at_chance = 100 * n_at_chance / n_total
            lines.append(
                f"    At chance level:           {n_at_chance}/{n_total} ({pct_at_chance:.0f}%)"
            )
            if n_at_chance > 0:
                lines.append(
                    "  WARNING: Some participants are at chance level — the model is not "
                    "capturing their behavior at all. These participants may use a "
                    "qualitatively different strategy that requires a different model."
                )

        # High heterogeneity signal
        if std_val > 0.5 * mean_val:
            lines.append(
                "  High variability in fit quality across participants — "
                "the model may not capture an important source of individual differences."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cross-model subgroup comparison
    # ------------------------------------------------------------------

    def _build_subgroup_comparison(self):
        """
        Compare per-participant fit across the top models to detect
        subgroups that different models serve better.
        """
        if not self.history:
            return ""

        # Collect all models with eval_metrics
        all_models = []
        for entry in self.history:
            for r in entry["results"]:
                em = r.get("eval_metrics", [])
                if em:
                    all_models.append({
                        "name": r["function_name"],
                        "bic": r["metric_value"],
                        "eval_metrics": em,
                        "iter": entry["iteration"],
                    })

        if len(all_models) < 2:
            return ""

        # Sort by mean BIC and take top 3
        all_models.sort(key=lambda x: x["bic"])
        top = all_models[:3]

        # All must have the same number of participants
        n_participants = len(top[0]["eval_metrics"])
        if not all(len(m["eval_metrics"]) == n_participants for m in top):
            return ""

        # For each participant, find which model fits best
        metrics_matrix = np.array([m["eval_metrics"] for m in top])
        best_model_per_participant = np.argmin(metrics_matrix, axis=0)

        lines = [f"Cross-model subgroup comparison (top {len(top)} models):"]
        for i, m in enumerate(top):
            count = int(np.sum(best_model_per_participant == i))
            pct = 100 * count / n_participants
            lines.append(
                f"  {m['name']} (mean BIC={m['bic']:.1f}): "
                f"best for {count}/{n_participants} participants ({pct:.0f}%)"
            )

        # Flag large disagreements
        bic_range = np.max(metrics_matrix, axis=0) - np.min(metrics_matrix, axis=0)
        median_range = np.median(bic_range)
        std_range = np.std(bic_range)
        large_disagreement = int(np.sum(bic_range > median_range + 2 * std_range))
        if large_disagreement > 0:
            lines.append(
                f"\n  {large_disagreement} participants show large model disagreement "
                f"(BIC spread > median + 2σ across models). These participants may use "
                f"qualitatively different strategies that no single model captures well."
            )

        # If the global best model is NOT best for a substantial portion
        global_best_pct = 100 * np.sum(best_model_per_participant == 0) / n_participants
        if global_best_pct < 70:
            lines.append(
                "\n  The globally best model is NOT best for all participants. "
                "Consider proposing models that target the subgroup where "
                "alternative models excel."
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

        n_clients = self._count_clients()
        client_note = f" across {n_clients} clients" if n_clients > 1 else ""
        lines = [f"Best BIC: {best_bic:.1f} ({best_name}, iter {best_iter}{client_note})."]
        momentum = self._build_search_momentum()
        if momentum:
            lines.append(momentum)

        # ID summary — lead with best per-parameter link
        if id_results is not None:
            mean_r2 = id_results.get("mean_r2", 0.0)
            max_r2 = id_results.get("max_r2", 0.0)
            best_param_name = id_results.get("best_param", "")
            detail = id_results.get("per_param_detail", {})
            # Find the strongest predictor for the best parameter
            best_pred = ""
            best_beta = 0.0
            if best_param_name and best_param_name in detail:
                for pred, coef in detail[best_param_name].get("coefficients", {}).items():
                    if abs(coef) > abs(best_beta):
                        best_beta = coef
                        best_pred = pred
            if best_pred and max_r2 > 0:
                lines.append(
                    f"Individual differences: best param R² = {max_r2:.3f} "
                    f"({best_param_name} ← {best_pred}, β={best_beta:.3f}), "
                    f"mean R² = {mean_r2:.3f}."
                )
            else:
                lines.append(
                    f"Individual differences: best param R² = {max_r2:.3f}, "
                    f"mean R² = {mean_r2:.3f}."
                )

        # Fit quality summary — extract key lines
        fit_quality = self._build_fit_quality_summary()
        if fit_quality:
            for line in fit_quality.split("\n"):
                if "Poor" in line or "At chance" in line:
                    lines.append(line.strip())

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

        # Cross-model subgroup comparison
        subgroup = self._build_subgroup_comparison()
        if subgroup:
            parts.append(subgroup)

        # Individual differences (current best model)
        if id_results is not None:
            parts.append(
                f"Individual Differences Analysis:\n"
                f"{id_results['summary_text']}\n\n"
                "Even a single strong link between a model parameter and a symptom measure is "
                "valuable. You don't need high average R² across all parameters — focus on "
                "designing parameters that could have clear, interpretable links to specific "
                "symptom dimensions. One strong parameter-symptom relationship matters more "
                "than many weak ones."
            )

        # Cross-model parameter-symptom links
        r2_landscape = self._build_r2_landscape()
        if r2_landscape:
            parts.append(r2_landscape)

        # Factor coverage
        factor_coverage = self._build_factor_coverage()
        if factor_coverage:
            parts.append(factor_coverage)

        # Search momentum
        momentum = self._build_search_momentum()
        if momentum:
            parts.append(momentum)

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

        # --- Data quantity context ---
        n_iterations = len(self.history)
        total_models = sum(len(entry["results"]) for entry in self.history)
        quantity_note = (
            f"## Data Quantity\n"
            f"Iterations completed: {n_iterations}\n"
            f"Total models evaluated: {total_models}"
        )
        if n_iterations < 5 or total_models < 10:
            quantity_note += (
                "\nNote: Conclusions drawn from fewer than 5 iterations / 10 models "
                "should be treated as preliminary."
            )
        context_parts.append(quantity_note)

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

        # --- Cross-model subgroup comparison ---
        subgroup = self._build_subgroup_comparison()
        if subgroup:
            context_parts.append(f"## Cross-Model Subgroup Comparison\n{subgroup}")

        factor_coverage = self._build_factor_coverage()
        if factor_coverage:
            context_parts.append(f"## Psychiatric Factor Usage\n{factor_coverage}")

        momentum = self._build_search_momentum()
        if momentum:
            context_parts.append(f"## Search Momentum\n{momentum}")

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
                "Even a single strong link between a model parameter and a symptom measure "
                "is valuable — one clear relationship matters more than many weak ones. "
                "Focus on designing parameters with interpretable links to specific "
                "symptom dimensions."
            )

        # --- Cross-model parameter-symptom links ---
        r2_landscape = self._build_r2_landscape()
        if r2_landscape:
            context_parts.append(f"## Parameter-Symptom Links Across Models\n{r2_landscape}")

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
            "You are critiquing another agent's search to find the optimal cognitive "
            "computational model to fit behavioural data and measures. Be purely data-driven — "
            "report what the BIC values show. Do NOT interpret mechanisms based on "
            "literature or prior knowledge. The goal is novel discovery.\n\n"
            "Qualify the confidence of your claims based on the amount of data available. "
            "With fewer than 5 iterations or 10 total models, use hedging language "
            "(e.g., 'preliminary evidence suggests', 'early indications are'). "
            "Only make strong claims when supported by consistent patterns across "
            "many iterations.\n\n"
            "For each suggestion you make, provide a confidence rating on a scale of "
            "1-10 (1 = speculative/little data, 10 = strong evidence across many iterations). "
            "Format as e.g. '[confidence: 7/10]' inline with each suggestion.\n\n"
            f"{search_context}\n\n"
            "Based on the data above:\n"
            "1. Which parameters or mechanisms consistently appear in well-fitting models?\n"
            "2. Which parameters add complexity without meaningfully improving BIC?\n"
            "3. What parameter or mechanism combinations have not been tried yet?\n"
            "4. Are there patterns in which participants fit poorly? Does the model "
            "struggle with a specific subset of participants?\n"
            "5. What proportion of participants are at chance level (model not capturing "
            "their behavior at all)? What might this suggest about missing mechanisms "
            "or qualitatively different strategies in the population?\n"
            "6. Do different models fit different subgroups of participants better? "
            "If so, what does this suggest about population heterogeneity and "
            "the potential for subgroup-specific model proposals?\n"
            "7. Which symptom dimensions (questionnaire factors) most strongly predict "
            "model parameters, and what does this suggest about individual differences?\n"
            "8. Based on the search momentum, is improvement slowing down? If so, suggest "
            "structurally different directions to explore.\n"
            "9. Provide specific, actionable guidance for generating the next set of models. "
            "Report fit quality proportions (% well-fit, % moderate, % poor, % at chance).\n\n"
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
            reasoning_effort = getattr(self.cfg.llm, "reasoning_effort", None)

            print(
                f"[GeCCo] Using Gemini model '{self.cfg.llm.base_model}' "
                f"(reasoning={reasoning_effort})"
            )

            config_args = {
                "temperature": self.cfg.llm.temperature,
                "system_instruction": self.cfg.llm.system_prompt,
            }

            if reasoning_effort:
                if self.cfg.llm.base_model.lower().startswith("gemini-3"):
                    config_args["thinking_config"] = types.ThinkingConfig(
                        thinking_level=reasoning_effort
                    )
                elif self.cfg.llm.base_model.lower().startswith("gemini-2"):
                    budget_map = {"minimal": 0, "low": 4096, "medium": 12288, "high": 24576}
                    config_args["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=budget_map.get(reasoning_effort, 4096)
                    )

            resp = self.model.models.generate_content(
                model=self.cfg.llm.base_model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_args),
            )
            decoded = resp.text.strip()

            return decoded
        # -----------------------------
        # vLLM (OpenAI-compatible API)
        # -----------------------------
        elif "vllm" in provider:
            max_out = getattr(self.cfg.llm, "max_output_tokens",
                              getattr(self.cfg.llm, "max_tokens", 4096))

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
            max_new = getattr(self.cfg.llm, "max_output_tokens", getattr(self.cfg.llm, "max_tokens", 4096))

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
