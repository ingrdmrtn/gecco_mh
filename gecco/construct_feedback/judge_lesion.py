"""Judge lesion framework for systematic ablation experiments."""

import re
from typing import Any


class JudgeLesion:
    """Intercepts judge verdicts and applies configurable transformations."""

    def __init__(self, cfg):
        self.cfg = cfg.lesion
        self._iteration = 0

    def apply(self, verdict, iteration, original_feedback):
        """Apply the configured lesion to the feedback.

        Parameters
        ----------
        verdict : JudgeVerdict
            The verdict object from the judge.
        iteration : int
            Current iteration number.
        original_feedback : str
            The synthesized_feedback string from the verdict.

        Returns
        -------
        str
            The (possibly modified) feedback string.
        """
        if not self.cfg.enabled:
            return original_feedback

        self._iteration = iteration
        method = getattr(self, f"_lesion_{self.cfg.lesion_type}", None)
        if method is None:
            raise ValueError(f"Unknown lesion type: {self.cfg.lesion_type}")
        return method(verdict, original_feedback)

    def _lesion_complete(self, verdict, feedback):
        """Return empty string — tests if any feedback is better than none."""
        return ""

    def _lesion_noise(self, verdict, feedback):
        """Return fixed generic text — tests if structured feedback beats noise."""
        return self.cfg.noise_text

    def _lesion_summary_only(self, verdict, feedback):
        """Keep only quantitative summary; strip LLM analysis and recommendations.

        Extracts:
        - BIC trajectory information
        - Landscape statistics (number of models, best BIC, etc.)
        - Fit quality metrics
        Removes all qualitative reasoning, analysis paragraphs, and recommendations.
        """
        result = feedback

        rec_pattern = re.compile(
            r"(?i)(key recommendations:?|recommendations:?|next steps:?|suggestions:?).*$",
            re.MULTILINE | re.DOTALL,
        )
        result = rec_pattern.sub("", result)

        if hasattr(verdict, "key_recommendations") and verdict.key_recommendations:
            for rec in verdict.key_recommendations:
                result = result.replace(f"- {rec}", "")
                result = result.replace(f"* {rec}", "")
                result = result.replace(rec, "")

        result = re.sub(r"\n{3,}", "\n\n", result)
        result = result.strip()

        return result if result else feedback

    def _lesion_no_tools(self, verdict, feedback):
        """Remove analysis that required tool use; keep only pre-computed context.

        Keeps:
        - BIC trajectory
        - Top models info
        - Winning model code reference
        Removes any analysis that would have required diagnostic tool queries
        (PPC analysis, parameter recovery details, residual analysis, etc.)
        """
        lines = feedback.split("\n")
        kept_lines = []

        tool_dependent_keywords = [
            "ppc",
            "posterior predictive",
            "parameter recovery",
            "residual",
            "diagnostic",
            "query",
            "database",
            "tool",
            "inspected",
            "analysed",
            "analyzed",
        ]

        for line in lines:
            stripped = line.strip().lower()
            is_tool_dependent = any(kw in stripped for kw in tool_dependent_keywords)

            if not is_tool_dependent:
                kept_lines.append(line)

        result = "\n".join(kept_lines).strip()
        return result if result else feedback

    def _lesion_no_recommendations(self, verdict, feedback):
        """Remove key_recommendations section from feedback."""
        result = feedback

        if hasattr(verdict, "key_recommendations") and verdict.key_recommendations:
            rec_text = "\n".join(f"- {r}" for r in verdict.key_recommendations)
            result = result.replace(rec_text, "")

        result = re.sub(
            r"(?i)(key recommendations:?|recommendations:?|next steps:?).*?(?=\n\n|\Z)",
            "",
            result,
            flags=re.DOTALL,
        )

        return result.strip()

    def _lesion_no_citations(self, verdict, feedback):
        """Remove references to specific past models."""
        result = feedback

        result = re.sub(
            r"(?i)(model[_\s]?\w+[_\s]?\d+|iter[_\s]?\d+[_\s]?(?:run[_\s]?\d+)?|participant[_\s]?\d+)",
            "a previous model",
            result,
        )

        result = re.sub(
            r"(?i)(as shown in|as demonstrated by|following|similar to|building on)\s+(model|iter|participant)\s*\w*",
            r"\1 a previous model",
            result,
        )

        return result

    def _lesion_no_diagnostics(self, verdict, feedback):
        """Remove PPC/residual analysis sections."""
        result = feedback

        diagnostic_section_patterns = [
            r"(?i)(predictive (model )?check|ppc).*?(?=\n\n|\Z)",
            r"(?i)(residual (analysis|pattern|inspection)).*?(?=\n\n|\Z)",
            r"(?i)(posterior predictive).*?(?=\n\n|\Z)",
            r"(?i)(diagnostic (tool|result|output|analysis)).*?(?=\n\n|\Z)",
            r"(?i)(parameter recovery).*?(?=\n\n|\Z)",
        ]

        for pattern in diagnostic_section_patterns:
            result = re.sub(pattern, "", result, flags=re.DOTALL)

        return result.strip()
