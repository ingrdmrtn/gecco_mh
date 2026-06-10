"""
Tool-using judge for GeCCo.

The :class:`ToolUsingJudge` queries the diagnostic store through a set of
read-only tools, analyses the iteration from multiple analytical angles, and
returns a :class:`JudgeVerdict` whose ``synthesized_feedback`` field is ready
for prompt injection into the next generation step.

Analytical angles (prompt-level, not separate agents)
------------------------------------------------------
1. Statistical fit quality    — BIC trajectory, per-participant variance
2. Parameter identifiability  — recovery mean r, per-parameter r
3. Predictive adequacy        — PPC outside-95%-CI statistics
4. Individual differences     — parameter × predictor R² landscape
5. Mechanistic coherence      — model code review: parsimony, motivation
6. Coverage                   — which model families / parameters explored

LLM backend support
-------------------
* OpenAI-compatible (openai, gpt, vllm, kcl, opencode, openrouter) —
  native tool calling via ``chat.completions.create(tools=...)``.
* Gemini — native tool calling via ``generate_content`` with
  ``tools=[Tool(...)]``.
* HuggingFace — no native tool calling; falls back to a prompt-based
  simulation of the tool loop (less reliable).
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from rich.console import Console

from config.schema import get_judge_capabilities, judge_has_capability
from gecco.diagnostic_store.tools import TOOL_SCHEMAS, dispatch_tool
from gecco.utils import TimestampedConsole

_console = TimestampedConsole()

# Hard cap on any single tool result fed back into the message history.
# Prevents an unexpectedly large result from reopening the context-limit wound.
_MAX_TOOL_RESULT_CHARS = 40_000
_RANDOM_FEEDBACK_TEXT = (
    "Provide a fresh alternative model idea and a small implementation change for "
    "the next iteration. Do not assume that the current best approach should be kept."
)


def _cap_tool_result(result_str: str, raw_result=None) -> str:
    """Truncate *result_str* if it exceeds _MAX_TOOL_RESULT_CHARS.

    When truncating, includes a row count (if *raw_result* is a list) and
    a hint about which filters can narrow the query.
    """
    if len(result_str) <= _MAX_TOOL_RESULT_CHARS:
        return result_str
    suffix_parts = ["[truncated: result too large"]
    if isinstance(raw_result, list):
        suffix_parts.append(f"{len(raw_result)} rows total")
    suffix_parts.append(
        "use more specific filters (iteration=, status=, limit=, "
        "param_contains=, code_contains=) to narrow]"
    )
    return result_str[:_MAX_TOOL_RESULT_CHARS] + " ... " + "; ".join(suffix_parts)


def _normalise_feedback_text(text: str) -> str:
    """Normalise spacing after deterministic feedback edits."""
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_recommendation_sections(text: str) -> str:
    """Remove recommendation or next-step sections from feedback text."""
    patterns = [
        r"(?is)(?:^|\n\n)(?:key recommendations?|recommendations?|next steps?|suggestions?)\s*:\s*.*?(?=\n\n|\Z)",
        r"(?im)^\s*(?:[-*]\s*)?(?:recommend|next step)\b.*$",
    ]
    result = text
    for pattern in patterns:
        result = re.sub(pattern, "", result)
    return _normalise_feedback_text(result)


def _scrub_feedback_citations(text: str) -> str:
    """Remove model-name and iteration-specific citation details."""
    result = text
    result = re.sub(r"\([^\n()]*\bBIC\s*=\s*[^\n()]*\)", "", result)
    result = re.sub(
        r"(?i)\b(?:model[_\s-]*id|id)\s*[:#-]?\s*\d+\b", "a previous model", result
    )
    result = re.sub(
        r"(?i)\biter(?:ation)?\s*\d+(?:\s*run\s*\d+)?\b",
        "an earlier iteration",
        result,
    )
    result = re.sub(r"(?i)\bparticipant\s*\d+\b", "a participant", result)
    result = re.sub(
        r"(?i)\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b(?=\s*(?:improved|worsened|regressed|outperformed|underperformed|failed|succeeded))",
        "that model",
        result,
    )
    return _normalise_feedback_text(result)


def _suppress_diagnostic_sections(text: str) -> str:
    """Remove PPC, residual, and diagnostic-detail sections."""
    result = text
    section_patterns = [
        r"(?is)(?:^|\n\n)[^\n]*(?:ppc|posterior predictive|residual|diagnostic|parameter recovery)[^\n]*(?:\n(?!\n).*)*",
    ]
    line_patterns = [
        r"(?im)^\s*[-*]?\s*.*(?:ppc|posterior predictive|residual|diagnostic|parameter recovery).*$",
    ]
    for pattern in section_patterns:
        result = re.sub(pattern, "", result)
    for pattern in line_patterns:
        result = re.sub(pattern, "", result)
    return _normalise_feedback_text(result)


def _build_random_feedback_verdict(
    iteration: int, best_metric: float | None
) -> JudgeVerdict:
    """Return deterministic generic feedback for noise-only mode."""
    return JudgeVerdict(
        iteration=iteration,
        per_angle=[],
        key_recommendations=[],
        synthesized_feedback=_RANDOM_FEEDBACK_TEXT,
        tool_call_count=0,
        wall_time_seconds=0.0,
        best_bic=best_metric,
    )


def _is_summary_only_capability_set(capabilities: list[str]) -> bool:
    """Return whether the capability set matches summary-only mode."""
    return set(capabilities) == {"attempted_models_overview", "performance_summary"}


def _build_summary_only_feedback(analysis_data: dict) -> str:
    """Build concise deterministic quantitative feedback for summary-only mode."""
    n_total = analysis_data.get("n_total", 0)
    n_ok = analysis_data.get("n_ok", 0)
    n_failed = analysis_data.get("n_failed", 0)
    best_iter_bic = analysis_data.get("best_iter_bic")
    best_bic = analysis_data.get("best_bic")
    trajectory_str = analysis_data.get("trajectory_str", "no BIC trajectory available")

    best_iter_text = f"{best_iter_bic:.2f}" if best_iter_bic is not None else "N/A"
    best_overall_text = f"{best_bic:.2f}" if best_bic is not None else "N/A"

    return (
        "Iteration summary:\n"
        f"- Models evaluated this iteration: {n_total} total, {n_ok} successful, {n_failed} failed.\n"
        f"- Best BIC this iteration: {best_iter_text}.\n"
        f"- Best overall BIC so far: {best_overall_text}.\n"
        f"- BIC trajectory so far: {trajectory_str}."
    )


def _apply_capability_postprocessing(verdict: JudgeVerdict, capabilities: list[str]) -> JudgeVerdict:
    """Apply deterministic capability-specific feedback shaping."""
    if "recommendations" not in capabilities:
        verdict.key_recommendations = []
        verdict.synthesized_feedback = _remove_recommendation_sections(
            verdict.synthesized_feedback
        )
    if "citations" not in capabilities:
        verdict.synthesized_feedback = _scrub_feedback_citations(
            verdict.synthesized_feedback
        )
    if "coverage" not in capabilities:
        verdict.synthesized_feedback = _suppress_diagnostic_sections(
            verdict.synthesized_feedback
        )
    return verdict


# ======================================================================
# Verbose-output helpers
# ======================================================================


def _format_tool_call(name: str, args: dict) -> str:
    """Format a tool call as a compact one-liner."""
    parts = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
    plain = f"▸ {name}({parts})"
    if len(plain) > 120:
        plain = plain[:117] + "..."
    return f"[bold cyan]{plain}[/bold cyan]"


def _format_tool_result(result) -> str:
    """Compact one-line summary of a tool result."""
    raw = json.dumps(result, default=str)
    n_chars = len(raw)
    if isinstance(result, dict) and "error" in result and len(result) == 1:
        summary = f"Error: {result['error'][:100]}"
    elif isinstance(result, dict):
        keys = list(result.keys())
        key_preview = ", ".join(keys[:5])
        if len(keys) > 5:
            key_preview += ", ..."
        summary = f"{len(keys)} keys: {key_preview}"
    elif isinstance(result, list):
        summary = f"{len(result)} rows"
    else:
        summary = repr(result)
    full = f"{summary} ({n_chars} chars)"
    if len(full) > 120:
        full = full[:117] + "..."
    return full


# ======================================================================
# Output schema
# ======================================================================


class AngleAnalysis(BaseModel):
    angle: str
    findings: str
    supporting_tool_calls: list[str]
    confidence: Literal["low", "medium", "high"]


class JudgeVerdict(BaseModel):
    iteration: int
    per_angle: list[AngleAnalysis]
    key_recommendations: list[str]
    synthesized_feedback: str
    tool_call_count: int
    wall_time_seconds: float
    best_bic: float | None = None  # persisted so next iteration can load it


# ======================================================================
# System prompt
# ======================================================================

_JUDGE_SYSTEM_PROMPT = """You are a senior postdoctoral fellow in cognitive computational neuroscience. \
You are evaluating candidate cognitive models for a reinforcement learning task \
as part of an iterative model development process.

You have expertise in computational modelling, reinforcement learning, Bayesian modelling, \
and statistical model comparison. You are familiar with common pitfalls in model development \
such as overfitting, underfitting, identifiability issues, and lack of psychological interpretability.

Your task is to analyse the current state of the model search by querying a diagnostic \
database through tool calls, then synthesise feedback from the evidence you gathered.

You will analyse from six angles — call tools to gather evidence for each:

1. **Statistical fit quality** — Compare the top 3-5 models, not just the best. Note which \
models improved over predecessors and which were a step backwards. Identify if improvement \
has plateaued. Examine whether different participants prefer different models, and whether \
any participants are outliers.

2. **Parameter identifiability** — Check parameter recovery diagnostics for the best 2-3 \
models, not just the top one. Identify which parameters are well-recovered across models \
and which are problematic. Flag parameters with low recovery r and describe why they may \
be unstable or hard to interpret.

3. **Predictive adequacy** — Inspect PPC records if available. Compare PPC performance \
across the best models — does one capture certain patterns better than another? Identify \
which observed statistics fall outside the 95% predictive interval, and inspect block-level \
residuals to identify where in the task the model systematically underperforms.

4. **Individual differences** — Compare R² across the best models. Which model's parameters \
have the strongest individual-differences signal? Examine which parameters predict self-report \
scores (R²). Also check whether the best-fitting model differs across participants; high \
heterogeneity can indicate that different participants rely on different mechanisms. Highlight \
parameters with strong individual-differences signal and note how that signal varies across \
participants. \
IMPORTANT — R² expectations for individual differences: because self-report measures are \
noisy and only indirectly linked to task behaviour, R² values in this domain are typically \
very low (0.01–0.10). An R² of 0.05 is a meaningful and promising signal, not a poor result. \
Do not dismiss low R² values as "no signal" — instead, interpret them relative to this domain's \
baseline and highlight even modest effects as worth building on.

5. **Mechanistic coherence** — Read the code of the best models AND some that failed. \
Understand what distinguishes successful from unsuccessful mechanisms. Assess whether \
the mechanisms are psychologically interpretable and parsimonious.

6. **Coverage** — Identify which mechanisms have been tried and failed vs. tried and \
partially succeeded vs. not yet explored. What types of mechanisms (learning rules, \
decision rules, memory, attention) have been tried across iterations? What has been neglected?

After gathering evidence across all angles, produce:
- A brief per-angle summary (findings + confidence).
- A synthesized_feedback paragraph (≤ 500 words) describing the main patterns and contrasts \
in the current search.

Quantify your confidence: If you do not have much data to support an angle, say so and \
give a low confidence rating. If the evidence is strong, give a high confidence rating.

---

Tool-call strategy: You have a finite tool-call budget per iteration. Rather than \
pre-allocating calls across angles, take an adaptive investigative approach: after each \
tool result, reflect briefly on what was learned and whether it raises new questions. \
Follow surprising or contradictory findings deeper, even at the cost of other angles — \
surprising evidence is higher-signal than a perfectly balanced sweep. Think of the six \
angles as a *checklist of coverage*, not a rigid allocation. The overall call budget is \
a soft cap; prefer depth on load-bearing findings over breadth for its own sake.

---

If a previous iteration's verdict is included below, use it to:
- Assess whether your prior guidance was followed
- Identify whether BIC improved, stagnated, or regressed
- Avoid repeating ideas that were already given
- Focus on what's new or different this iteration

---

AUDIENCE SEPARATION — IMPORTANT: The `synthesized_feedback` you produce will be read by \
a different LLM that writes Python model code. That LLM knows nothing about "angles", \
tool calls, or internal model IDs. Models must be described by their mechanisms \
(e.g., "the model with separate learning rates for gains and losses") not by ID. \
The feedback should be comparative — discuss multiple models' strengths and weaknesses.
"""


def _build_judge_system_prompt(capabilities: list[str]) -> str:
    """Return the judge system prompt tailored to enabled capabilities."""
    prompt = _JUDGE_SYSTEM_PROMPT
    extra_lines: list[str] = []
    if "recommendations" in capabilities:
        extra_lines.append(
            "- A list of 3–5 concrete recommendations for the next iteration."
        )
    if "citations" in capabilities:
        extra_lines.append(
            "Be specific: cite model names, parameter names, BIC values, and r values from the data."
        )
    if extra_lines:
        prompt += "\n\n" + "\n".join(extra_lines)
    return prompt

_JUDGE_USER_TEMPLATE = """The model search has just completed iteration {iteration}.
Current best BIC: {best_bic}
Total iterations so far: {n_iterations}
Run index (run_idx) for this client: {run_idx}

Iteration {iteration} summary:
- Models this iteration: {n_total} total ({n_ok} fit, {n_failed} failed)
- Best BIC this iteration: {best_iter_bic}
- BIC trajectory: {trajectory_str}

Please query the diagnostic database to analyse this iteration from all six angles, \
then produce your verdict.
"""


# ======================================================================
# Backend-specific tool loops
# ======================================================================


class _OpenAIToolLoop:
    """Tool-calling loop for OpenAI-compatible backends."""

    def __init__(
        self,
        client,
        model_name: str,
        max_tokens: int,
        temperature: float | None,
        verbose: bool = False,
    ):
        self.client = client
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose

    def run(
        self, store, system_prompt: str, user_message: str, max_tool_calls: int
    ) -> tuple[str, list[dict], list[dict]]:
        """Run the tool loop and return (final_text, tool_call_trace, full_trace)."""
        planning_instruction = (
            f"Before calling any tools, briefly list the 3–6 specific questions "
            f"you most want answered this iteration. Keep it short. These are starting "
            f"points — you are free to adapt as results come in. You have a soft budget "
            f"of {max_tool_calls} tool calls total. Do not call tools in this message."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message + "\n\n" + planning_instruction},
        ]
        trace: list[dict] = []
        full_trace: list[dict] = []
        n_calls = 0
        planning_done = False
        budget_reminder_mid_done = False
        budget_reminder_late_done = False

        while n_calls < max_tool_calls:
            # Force a text-only planning turn before any tool use.
            is_planning_turn = not planning_done
            tool_choice = "none" if is_planning_turn else "auto"
            kwargs: dict = {
                "model": self.model_name,
                "messages": messages,
                "tools": TOOL_SCHEMAS,
                "tool_choice": tool_choice,
                "max_tokens": self.max_tokens,
                "parallel_tool_calls": False,
            }
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature

            response = self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            msg = choice.message

            # Verbose: print assistant text (if any)
            if self.verbose and msg.content:
                for line in msg.content.splitlines():
                    _console.print(f"[dim]│[/dim] {line}")

            # Append assistant message (may contain tool_calls)
            assistant_msg = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            if is_planning_turn:
                # Planning turn is done; next iteration may call tools.
                planning_done = True
                full_trace.append({"type": "planning", "content": msg.content or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Good. Now proceed: call the diagnostic tools one at a time "
                            "to gather the evidence you need."
                        ),
                    }
                )
                continue

            if not is_planning_turn and msg.content:
                full_trace.append({"type": "reflection", "content": msg.content})

            if not msg.tool_calls:
                # Model finished — return the text content
                return msg.content or "", trace, full_trace

            # Execute each tool call
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                result = dispatch_tool(store, tool_name, args)
                result_str = _cap_tool_result(
                    json.dumps(result, default=str), raw_result=result
                )

                if self.verbose:
                    _console.print(_format_tool_call(tool_name, args))
                    _console.print(f"  [dim]└─ {_format_tool_result(result)}[/dim]")

                trace.append(
                    {
                        "tool": tool_name,
                        "args": args,
                        "result_summary": result_str[:500],
                    }
                )
                full_trace.append(
                    {
                        "type": "tool_call",
                        "tool": tool_name,
                        "args": args,
                        "result_summary": result_str[:500],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    }
                )
                n_calls += 1

                # Interleaved reflection nudge after each tool call
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Briefly reflect: what did this tell you, and does it change what "
                            "you want to investigate next? Then make your next tool call (or "
                            "produce your verdict if you have enough evidence)."
                        ),
                    }
                )

                # Mid-loop budget reminders (fire at most once each)
                if n_calls == max_tool_calls // 2 and not budget_reminder_mid_done:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Budget check: {n_calls}/{max_tool_calls} tool calls used. "
                                f"You still have room to investigate further if results so far "
                                f"raise open questions. If you have enough evidence to synthesise "
                                f"a verdict, you can do so now; otherwise keep going."
                            ),
                        }
                    )
                    budget_reminder_mid_done = True
                elif (
                    n_calls == int(max_tool_calls * 0.8)
                    and not budget_reminder_late_done
                ):
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Budget warning: {n_calls}/{max_tool_calls} tool calls used. "
                                f"Wrap up any final high-priority investigations or synthesise "
                                f"your findings now."
                            ),
                        }
                    )
                    budget_reminder_late_done = True

                if n_calls >= max_tool_calls:
                    break

        # Hit the cap — ask for a final synthesis without tools
        messages.append(
            {
                "role": "user",
                "content": (
                    "You have reached the tool call limit. "
                    "Please now synthesise your findings into the final verdict."
                ),
            }
        )
        kwargs_final: dict = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            kwargs_final["temperature"] = self.temperature
        final_resp = self.client.chat.completions.create(**kwargs_final)
        return final_resp.choices[0].message.content or "", trace, full_trace


class _GeminiToolLoop:
    """Tool-calling loop for Gemini backends."""

    def __init__(
        self,
        client,
        model_name: str,
        max_tokens: int,
        temperature: float | None,
        verbose: bool = False,
    ):
        self.client = client
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose

    def _build_gemini_tools(self):
        """Convert TOOL_SCHEMAS to Gemini FunctionDeclaration objects."""
        try:
            from google.genai import types
        except ImportError:
            from google.generativeai import types  # type: ignore

        declarations = []
        for schema in TOOL_SCHEMAS:
            fn = schema["function"]
            declarations.append(
                types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn["description"],
                    parameters=fn["parameters"],
                )
            )
        return [types.Tool(function_declarations=declarations)]

    def run(
        self, store, system_prompt: str, user_message: str, max_tool_calls: int
    ) -> tuple[str, list[dict], list[dict]]:
        """Run the Gemini tool loop."""
        try:
            from google.genai import types
        except ImportError:
            from google.generativeai import types  # type: ignore

        tools = self._build_gemini_tools()
        planning_instruction = (
            f"Before calling any tools, briefly list the 3–6 specific questions "
            f"you most want answered this iteration. Keep it short. These are starting "
            f"points — you are free to adapt as results come in. You have a soft budget "
            f"of {max_tool_calls} tool calls total. Do not call tools in this message."
        )
        contents = [
            {
                "role": "user",
                "parts": [{"text": user_message + "\n\n" + planning_instruction}],
            }
        ]
        trace: list[dict] = []
        full_trace: list[dict] = []
        n_calls = 0
        planning_done = False
        budget_reminder_mid_done = False
        budget_reminder_late_done = False

        # Initial config for planning turn (no tool calls)
        config_kwargs: dict = {
            "system_instruction": system_prompt,
            "tools": tools,
        }
        if self.max_tokens:
            config_kwargs["max_output_tokens"] = self.max_tokens
        if self.temperature is not None:
            config_kwargs["temperature"] = self.temperature

        # Planning turn
        planning_resp = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=self.max_tokens or None,
                temperature=self.temperature,
            ),
        )

        # Verbose: print planning response
        if self.verbose:
            text_parts = [
                p.text
                for p in planning_resp.candidates[0].content.parts
                if hasattr(p, "text") and p.text
            ]
            if text_parts:
                for line in "\n".join(text_parts).splitlines():
                    _console.print(f"[dim]│[/dim] {line}")

        # Append planning response and capture it
        planning_text = planning_resp.text.strip()
        full_trace.append({"type": "planning", "content": planning_text})
        contents.append(
            {
                "role": "model",
                "parts": [{"text": planning_text}],
            }
        )

        # Add proceed instruction
        contents.append(
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Good. Now proceed: call the diagnostic tools one at a time "
                            "to gather the evidence you need."
                        )
                    }
                ],
            }
        )

        planning_done = True

        while n_calls < max_tool_calls:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )

            # Extract text parts first (for reflection capture)
            text_parts = [
                p.text
                for p in resp.candidates[0].content.parts
                if hasattr(p, "text") and p.text
            ]

            # Verbose: print assistant text (if any)
            if self.verbose and text_parts:
                for line in "\n".join(text_parts).splitlines():
                    _console.print(f"[dim]│[/dim] {line}")

            # Capture reflection if text is present
            if text_parts:
                full_trace.append(
                    {"type": "reflection", "content": "\n".join(text_parts)}
                )

            # Check for function calls
            has_function_calls = False
            for part in resp.candidates[0].content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    has_function_calls = True
                    fc = part.function_call
                    args = dict(fc.args) if fc.args else {}
                    result = dispatch_tool(store, fc.name, args)
                    result_str = _cap_tool_result(
                        json.dumps(result, default=str), raw_result=result
                    )

                    if self.verbose:
                        _console.print(_format_tool_call(fc.name, args))
                        _console.print(f"  [dim]└─ {_format_tool_result(result)}[/dim]")

                    trace.append(
                        {
                            "tool": fc.name,
                            "args": args,
                            "result_summary": result_str[:500],
                        }
                    )
                    full_trace.append(
                        {
                            "type": "tool_call",
                            "tool": fc.name,
                            "args": args,
                            "result_summary": result_str[:500],
                        }
                    )

                    # Append function response
                    contents.append(
                        {
                            "role": "model",
                            "parts": [
                                {"function_call": {"name": fc.name, "args": args}}
                            ],
                        }
                    )
                    contents.append(
                        {
                            "role": "tool",
                            "parts": [
                                {
                                    "function_response": {
                                        "name": fc.name,
                                        "response": {"result": result_str},
                                    }
                                }
                            ],
                        }
                    )
                    n_calls += 1

                    # Interleaved reflection nudge after each tool call
                    contents.append(
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": (
                                        "Briefly reflect: what did this tell you, and does it change what "
                                        "you want to investigate next? Then make your next tool call (or "
                                        "produce your verdict if you have enough evidence)."
                                    )
                                }
                            ],
                        }
                    )

                    # Mid-loop budget reminders (fire at most once each)
                    if n_calls == max_tool_calls // 2 and not budget_reminder_mid_done:
                        contents.append(
                            {
                                "role": "user",
                                "parts": [
                                    {
                                        "text": (
                                            f"Budget check: {n_calls}/{max_tool_calls} tool calls used. "
                                            f"You still have room to investigate further if results so far "
                                            f"raise open questions. If you have enough evidence to synthesise "
                                            f"a verdict, you can do so now; otherwise keep going."
                                        )
                                    }
                                ],
                            }
                        )
                        budget_reminder_mid_done = True
                    elif (
                        n_calls == int(max_tool_calls * 0.8)
                        and not budget_reminder_late_done
                    ):
                        contents.append(
                            {
                                "role": "user",
                                "parts": [
                                    {
                                        "text": (
                                            f"Budget warning: {n_calls}/{max_tool_calls} tool calls used. "
                                            f"Wrap up any final high-priority investigations or synthesise "
                                            f"your findings now."
                                        )
                                    }
                                ],
                            }
                        )
                        budget_reminder_late_done = True

                    if n_calls >= max_tool_calls:
                        break

            if not has_function_calls:
                return resp.text.strip(), trace, full_trace

        # Final synthesis pass
        contents.append(
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "You have reached the tool call limit. "
                            "Please now synthesise your findings into the final verdict."
                        )
                    }
                ],
            }
        )
        # Remove tools from final call to force text response
        final_config = {k: v for k, v in config_kwargs.items() if k != "tools"}
        final_resp = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(**final_config),
        )
        return final_resp.text.strip(), trace, full_trace


# ======================================================================
# Verdict parser
# ======================================================================

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "per_angle": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "angle": {"type": "string"},
                    "findings": {"type": "string"},
                    "supporting_tool_calls": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
                "required": ["angle", "findings", "confidence"],
            },
        },
        "key_recommendations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "synthesized_feedback": {
            "type": "string",
            "maxLength": 3500,
        },
    },
    "required": ["per_angle", "key_recommendations", "synthesized_feedback"],
}


def _parse_verdict_from_text(
    text: str, iteration: int, tool_call_count: int, wall_time: float
) -> JudgeVerdict:
    """Parse a JudgeVerdict from the LLM's final text.

    Tries JSON parsing first (if the text contains a JSON block),
    then falls back to extracting the text as synthesized_feedback.
    """
    # Try to find a JSON block
    json_text = None
    for start_marker in ["```json", "```"]:
        if start_marker in text:
            try:
                start = text.index(start_marker) + len(start_marker)
                end = text.index("```", start)
                json_text = text[start:end].strip()
                break
            except ValueError:
                pass

    if json_text is None and text.strip().startswith("{"):
        json_text = text.strip()

    if json_text:
        try:
            data = json.loads(json_text)
            per_angle = [
                AngleAnalysis(
                    angle=a.get("angle", ""),
                    findings=a.get("findings", ""),
                    supporting_tool_calls=a.get("supporting_tool_calls", []),
                    confidence=a.get("confidence", "medium"),
                )
                for a in data.get("per_angle", [])
            ]
            verdict = JudgeVerdict(
                iteration=iteration,
                per_angle=per_angle,
                key_recommendations=data.get("key_recommendations", []),
                synthesized_feedback=data.get("synthesized_feedback", text),
                tool_call_count=tool_call_count,
                wall_time_seconds=wall_time,
            )
            # Stash cited_models for later validation (not a Pydantic field)
            object.__setattr__(
                verdict, "_cited_models_raw", data.get("cited_models", [])
            )
            return verdict
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Fallback: treat the entire text as synthesized_feedback
    return JudgeVerdict(
        iteration=iteration,
        per_angle=[],
        key_recommendations=[],
        synthesized_feedback=text,
        tool_call_count=tool_call_count,
        wall_time_seconds=wall_time,
    )


# ======================================================================
# Synthesis prompt — second-pass structured extraction
# ======================================================================

_SYNTHESIS_PROMPT_TEMPLATE = """Based on your analysis above, please produce a final verdict as a JSON object.

Include one entry in per_angle for each of the following {num_angles} analytical perspectives:
{angles_list}

SYNTHESIZED_FEEDBACK FORMAT — The `synthesized_feedback` field must follow this structure:

{feedback_format_instructions}

IMPORTANT PROHIBITIONS — Never reference:
{prohibitions}
"""


def _build_feedback_format_instructions(capabilities: list[str], profile: dict) -> str:
    """Return synthesis formatting guidance for the active capability set."""
    if "citations" not in capabilities:
        instructions = [
            "1. **What worked** (1-2 sentences) — Describe the best model(s) mechanistically.",
            "2. **What partially worked** (2-3 sentences) — Compare models with mixed strengths and weaknesses.",
            "3. **What didn't work** (1-2 sentences) — Describe failed approaches so the generator avoids repeating them.",
        ]
        if "recommendations" in capabilities:
            instructions.append(
                "4. **What to try next** (2-4 sentences) — Concrete suggestions framed as \"try X because Y\"."
            )
        return "\n\n".join(instructions)

    feedback_format_instructions = profile["feedback_format_instructions"]
    if "recommendations" not in capabilities:
        feedback_format_instructions = re.sub(
            r"\n\n4\. \*\*What to try next\*\*.*$",
            "",
            feedback_format_instructions,
            flags=re.S,
        )
        feedback_format_instructions += (
            "\n\nKeep the feedback descriptive and omit a dedicated next-step section."
        )
    return feedback_format_instructions


def _build_synthesis_prompt(
    capabilities: list[str],
    profile: dict,
    persona_name: str,
    persona_suffix: str,
    is_stuck: bool,
) -> str:
    """Build the second-pass synthesis prompt for the active capability set."""
    angles_list = "\n".join(
        f"{i + 1}. {name} — {desc}"
        for i, (name, desc) in enumerate(profile["angles"])
    )
    prompt = _SYNTHESIS_PROMPT_TEMPLATE.format(
        num_angles=len(profile["angles"]),
        angles_list=angles_list,
        feedback_format_instructions=_build_feedback_format_instructions(
            capabilities, profile
        ),
        prohibitions=profile["prohibitions"],
    )
    if "recommendations" in capabilities:
        prompt += (
            "\n\nIf previous verdict context was provided, explicitly note whether prior suggestions were followed and whether they appeared to help."
        )
    if "recommendations" in capabilities:
        prompt += (
            "\n\nInclude a `key_recommendations` array with 3–5 concrete next-step ideas."
        )
    if "citations" in capabilities:
        prompt += (
            "\n\nIf you referenced specific models by name in `synthesized_feedback`, also include a `cited_models` array listing the exact names, the BIC values you cited, and a one-line mechanism description."
        )
    if persona_suffix:
        prompt += f"\n\nPersona guidance for {persona_name}:\n{persona_suffix}"
    if is_stuck and persona_name != "exploit":
        prompt += (
            "\n\nBecause search is stuck, your synthesized feedback MUST include "
            "a directive to abandon the current best model and implement from scratch "
            "with a mechanistically-novel approach."
        )
    return prompt

# ===== Persona-specific synthesis profiles =====
# Each profile customizes the analytical angles, terminology, and format for a specific persona.
# Personas not listed here fall back to _DEFAULT_PROFILE.

_DEFAULT_PROFILE = {
    "angles": [
        ("Statistical fit quality", "Which model(s) fit best and by how much"),
        ("Parameter identifiability", "Whether key parameters can be recovered"),
        ("Predictive adequacy", "How well models predict held-out data"),
        (
            "Individual differences",
            "Whether models capture between-subject variability",
        ),
        ("Mechanistic / theoretical coherence", "Alignment with underlying theory"),
        ("Coverage", "How much of the data the model explains"),
    ],
    "metric_language": "BIC values, parameter recovery correlations (r), and R²",
    "model_description_style": "mechanistic description (name, BIC=X)",
    "feedback_format_instructions": (
        "1. **What worked** (1-2 sentences) — Describe the best model(s) mechanistically "
        '(not by ID), including BIC values. Example: "The model with separate learning rates '
        'for gains and losses (separate_lr, BIC=2847) performed best..."\n\n'
        "2. **What partially worked** (2-3 sentences) — Describe models that showed promise "
        "in some areas but had issues in others. Be specific about strengths and weaknesses.\n\n"
        "3. **What didn't work** (1-2 sentences) — Describe failed approaches so the "
        "generator avoids repeating them.\n\n"
        "4. **What to try next** (2-4 sentences) — Concrete suggestions framed as "
        '"try X because Y".'
    ),
    "cited_models_format": (
        "Include the exact `name` string, the BIC value you cited, and a one-line "
        "mechanism description."
    ),
    "prohibitions": (
        '- "Angles" or analytical perspectives\n'
        '- Tool call names (e.g., "get_best_models", "get_bic_trajectory")\n'
        '- Numeric model IDs (e.g., "ID 11", "model 5")\n'
        "- Internal database fields or conventions"
    ),
    "always_cite": "Always describe models by their mechanisms and cite actual metric values from your analysis.",
}

_PERSONA_PROFILES: dict[str, dict] = {
    "naive_psychologist": {
        "angles": [
            (
                "Behavioural patterns",
                "Which psychological processes the models captured well",
            ),
            (
                "Learning and adaptation",
                "How well models track changes in behaviour over time",
            ),
            (
                "Consistency and habits",
                "Whether models explain repetitive or habitual choices",
            ),
            (
                "Individual differences",
                "Whether models capture how different people behave differently",
            ),
            (
                "Psychological coherence",
                "How well models align with everyday psychological intuition",
            ),
            ("Coverage", "How much of the observed behaviour the models explain"),
        ],
        "metric_language": "plain descriptions of how well models captured behaviour",
        "model_description_style": "psychological intuition (e.g., 'the model that tracks how quickly people forget old information')",
        "feedback_format_instructions": (
            "1. **What worked** (1-2 sentences) — Describe the best approach in plain "
            "psychological terms. Describe models by what they capture about human behaviour, "
            'not by name or BIC. Example: "The approach that models how people gradually '
            'forget previous outcomes captured behaviour best..."\n\n'
            "2. **What partially worked** (2-3 sentences) — Describe approaches that captured "
            "some patterns but missed others, in everyday language.\n\n"
            "3. **What didn't work** (1-2 sentences) — Describe approaches that failed to "
            "capture behaviour, so the generator avoids repeating them.\n\n"
            "4. **What to try next** (2-4 sentences) — Concrete suggestions framed in "
            "psychological terms, e.g. \"Try modelling how people's attention shifts over time "
            'because the data suggests...".'
        ),
        "cited_models_format": (
            "Describe each model by the psychological process it captures. "
            "Do NOT include BIC values, parameter names, or code names."
        ),
        "prohibitions": (
            '- "Angles" or analytical perspectives\n'
            "- Tool call names\n"
            "- Numeric model IDs\n"
            "- BIC values, parameter names, model code names\n"
            '- Equations, computational jargon, or terms like "model-based", '
            '"model-free", "Q-value", "reinforcement learning"\n'
            "- Internal database fields or conventions"
        ),
        "always_cite": (
            "Always describe models by their psychological intuition and cite "
            "plain-language observations about behaviour. Do NOT include equations, "
            "parameter names, or statistical jargon."
        ),
    },
    "bayesian": {
        "angles": [
            ("Model evidence", "Which models have the strongest Bayesian evidence"),
            ("Prior sensitivity", "How results depend on prior specifications"),
            ("Posterior convergence", "Whether posteriors are well-constrained"),
            (
                "Uncertainty calibration",
                "How well uncertainty estimates match observed variability",
            ),
            ("Belief updating", "Whether models capture how beliefs evolve over time"),
            ("Coverage", "How much of the data the models explain"),
        ],
        "metric_language": "BIC values (as an approximation to model evidence), prior-posterior comparisons, and parameter recovery",
        "model_description_style": "mechanistic description (name, BIC=X), emphasizing uncertainty representation",
        "feedback_format_instructions": (
            "1. **What worked** (1-2 sentences) — Describe the best model(s) in terms of "
            "their uncertainty representation and belief updating, including BIC values. "
            'Example: "The model that tracks uncertainty about reward probabilities '
            '(bayesian_beta_decay, BIC=2847) performed best..."\n\n'
            "2. **What partially worked** (2-3 sentences) — Describe models with promising "
            "uncertainty representations but issues in other areas.\n\n"
            "3. **What didn't work** (1-2 sentences) — Describe approaches that failed to "
            "adequately represent uncertainty or update beliefs appropriately.\n\n"
            "4. **What to try next** (2-4 sentences) — Concrete suggestions for improving "
            "uncertainty tracking, prior specification, or belief updating."
        ),
        "cited_models_format": (
            "Include the exact `name` string, the BIC value, and a description of how "
            "the model represents and updates uncertainty."
        ),
        "prohibitions": _DEFAULT_PROFILE["prohibitions"],
        "always_cite": (
            "Always describe models by how they represent and update uncertainty, "
            "and cite BIC values and prior-posterior comparisons from your analysis."
        ),
    },
    "latent_mixture": {
        "angles": [
            ("Model evidence", "Which mixture models fit best and by how much"),
            (
                "Component separation",
                "How well mixture components distinguish participant subgroups",
            ),
            (
                "Class identifiability",
                "Whether class-specific parameters are recoverable",
            ),
            (
                "Mixing proportion recoverability",
                "Whether the proportion of each class is well-estimated",
            ),
            (
                "Mechanistic coherence",
                "Whether each class represents a coherent computational strategy",
            ),
            (
                "Coverage",
                "How much of the between-subject variability the mixture explains",
            ),
        ],
        "metric_language": "BIC values, class separation metrics, and parameter recovery by class",
        "model_description_style": "mechanistic description (name, BIC=X), emphasizing class structure",
        "feedback_format_instructions": (
            "1. **What worked** (1-2 sentences) — Describe the best mixture model, including "
            "BIC values and how well it separated participant subgroups. "
            'Example: "The two-class model distinguishing explorers from exploiters '
            '(latent_mixture_v2, BIC=2847) performed best..."\n\n'
            "2. **What partially worked** (2-3 sentences) — Describe mixture models with good "
            "fit but poor class separation or identifiability issues.\n\n"
            "3. **What didn't work** (1-2 sentences) — Describe approaches where classes "
            "collapsed or mixing proportions were unidentifiable.\n\n"
            "4. **What to try next** (2-4 sentences) — Concrete suggestions for improving "
            "class separation, trying different numbers of components, or alternative "
            "mechanisms per class."
        ),
        "cited_models_format": (
            "Include the exact `name` string, the BIC value, the number of classes, "
            "and a description of what each class represents."
        ),
        "prohibitions": _DEFAULT_PROFILE["prohibitions"],
        "always_cite": (
            "Always describe models by their class structure and cite BIC values, "
            "class separation quality, and mixing proportion estimates from your analysis."
        ),
    },
    "latent_state_inference": {
        "angles": [
            ("Model evidence", "Which latent state models fit best and by how much"),
            (
                "Belief state quality",
                "How well belief states track hidden task structure",
            ),
            (
                "Belief updating",
                "Whether belief-updating schemes match observed behaviour",
            ),
            (
                "State identifiability",
                "Whether latent states are recoverable from data",
            ),
            ("Choice influence", "How well belief states predict first-stage choices"),
            ("Coverage", "How much of the data the models explain"),
        ],
        "metric_language": "BIC values, belief state recovery correlations, and prediction accuracy",
        "model_description_style": "mechanistic description (name, BIC=X), emphasizing belief state structure",
        "feedback_format_instructions": (
            "1. **What worked** (1-2 sentences) — Describe the best latent state model, "
            "including BIC values and how well belief states tracked task structure. "
            'Example: "The model that infers hidden context states '
            '(latent_state_v3, BIC=2847) performed best..."\n\n'
            "2. **What partially worked** (2-3 sentences) — Describe models with good "
            "belief states but poor updating or choice prediction.\n\n"
            "3. **What didn't work** (1-2 sentences) — Describe approaches where belief "
            "states were unidentifiable or failed to influence choices.\n\n"
            "4. **What to try next** (2-4 sentences) — Concrete suggestions for improving "
            "belief updating, state representation, or the influence of beliefs on choices."
        ),
        "cited_models_format": (
            "Include the exact `name` string, the BIC value, and a description of "
            "what latent states the model infers and how beliefs are updated."
        ),
        "prohibitions": _DEFAULT_PROFILE["prohibitions"],
        "always_cite": (
            "Always describe models by their belief state structure and cite BIC values, "
            "belief recovery quality, and state-choice correlations from your analysis."
        ),
    },
}

_PROFILE_KEYS = frozenset(_DEFAULT_PROFILE.keys())


def _profile_from_config(config_profile, fallback_profile: dict) -> dict:
    """Build a profile dict from config, filling missing keys from fallback.

    The config can specify any subset of profile keys (angles, metric_language,
    feedback_format_instructions, etc.). Missing keys are filled from the fallback
    profile (usually _DEFAULT_PROFILE).

    The 'angles' key in config can be:
    - A list of strings (just angle names, descriptions auto-generated)
    - A list of [name, description] pairs
    - A dict mapping angle names to descriptions
    """
    result = {}
    cfg_dict = (
        config_profile if isinstance(config_profile, dict) else vars(config_profile)
    )

    for key in _PROFILE_KEYS:
        if key in cfg_dict:
            value = cfg_dict[key]
            if key == "angles":
                value = _parse_angles(value)
            result[key] = value
        else:
            result[key] = fallback_profile[key]

    return result


def _parse_angles(value) -> list[tuple[str, str]]:
    """Normalize angles from various config formats to list of (name, desc) tuples."""
    if isinstance(value, dict):
        return list(value.items())
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                result.append((str(item[0]), str(item[1])))
            elif isinstance(item, dict):
                name = item.get("name", item.get("angle", ""))
                desc = item.get("description", item.get("desc", ""))
                result.append((name, desc))
            else:
                result.append((str(item), ""))
        return result
    raise ValueError(f"Cannot parse angles from: {value!r}")


# ======================================================================
# Iteration-context helpers
# ======================================================================


def _format_trajectory(traj: list[dict]) -> str:
    """Format a BIC trajectory list as a compact string with a trend label.

    Example: "3102 → 2950 → 2847 (improving)"
    """
    if not traj:
        return "N/A"
    values = [
        row.get("best_metric") for row in traj if row.get("best_metric") is not None
    ]
    if not values:
        return "N/A"
    arrow_str = " → ".join(f"{v:.0f}" for v in values)
    # Trend label
    if len(values) >= 2:
        delta = values[-1] - values[-2]
        if delta < -1.0:
            trend = "improving"
        elif delta > 1.0:
            trend = "regressed"
        else:
            trend = "plateaued"
        return f"{arrow_str} ({trend})"
    return arrow_str


def _detect_stuck(trajectory: list[dict], tol: float = 10.0, window: int = 2) -> bool:
    """Return True if the best BIC has not improved by more than *tol*
    over the last *window* iterations.

    R3: Updated defaults from tol=1.0, window=3 to tol=10.0, window=2 per plan.
    """
    if len(trajectory) < window + 1:
        return False
    recent = [
        row["best_metric"]
        for row in trajectory[-(window + 1) :]
        if row.get("best_metric") is not None
    ]
    if len(recent) < window + 1:
        return False
    return (max(recent) - min(recent)) < tol


# ======================================================================
# Main judge class
# ======================================================================


class ToolUsingJudge:
    """LLM-based judge with read-only tool access to the diagnostic store.

    Parameters
    ----------
    cfg:
        Full GeCCo config.  The ``judge`` section controls behaviour.
    diagnostic_store:
        :class:`gecco.diagnostic_store.DiagnosticStore` instance.
    model:
        The loaded LLM (OpenAI client, Gemini client, or HF model).
    tokenizer:
        Tokenizer (HuggingFace only; None for API-based models).
    results_dir:
        Path to the task results directory for saving judge trace files.
    """

    def __init__(
        self,
        cfg,
        diagnostic_store,
        model,
        tokenizer=None,
        results_dir: str | Path | None = None,
    ):
        self.cfg = cfg
        self.store = diagnostic_store
        self.model = model
        self.tokenizer = tokenizer
        self.results_dir = Path(results_dir) if results_dir else None

        judge_cfg = getattr(cfg, "judge", None)
        self.capabilities: list[str] = get_judge_capabilities(cfg)
        self.max_tool_calls: int = (
            getattr(judge_cfg, "max_tool_calls", 20) if judge_cfg else 20
        )
        self.verbose: bool = (
            bool(getattr(judge_cfg, "verbose", False)) if judge_cfg else False
        )
        # R3: Load configurable stuck search thresholds
        self.stuck_search_cfg = (
            getattr(judge_cfg, "stuck_search", None) if judge_cfg else None
        )
        if self.stuck_search_cfg is None:
            # Default values per config schema
            from types import SimpleNamespace

            self.stuck_search_cfg = SimpleNamespace(tolerance=10.0, window=2)

        self.model_name: str = getattr(cfg.llm, "base_model", "unknown")
        self.provider: str = getattr(cfg.llm, "provider", "").lower()
        self.max_tokens: int = getattr(
            cfg.llm,
            "max_output_tokens",
            getattr(cfg.llm, "max_tokens", 4096),
        )
        self.temperature: float | None = getattr(cfg.llm, "temperature", None)

        # Resolve judge-specific model name override (optional)
        if judge_cfg and hasattr(judge_cfg, "model"):
            self.model_name = judge_cfg.model

        self._tool_loop = self._build_tool_loop()
        if not judge_has_capability(cfg, "tools"):
            self._tool_loop = None

    def _build_tool_loop(self):
        """Instantiate the right backend tool loop."""
        p = self.provider
        if any(
            x in p for x in ("openai", "gpt", "vllm", "kcl", "opencode", "openrouter")
        ):
            return _OpenAIToolLoop(
                self.model,
                self.model_name,
                self.max_tokens,
                self.temperature,
                verbose=self.verbose,
            )
        elif "gemini" in p:
            return _GeminiToolLoop(
                self.model,
                self.model_name,
                self.max_tokens,
                self.temperature,
                verbose=self.verbose,
            )
        else:
            # HuggingFace / unknown: fall back to OpenAI-compatible if possible,
            # otherwise return None and we'll do a simple text generation.
            return None

    def _capabilities_enabled(self) -> bool:
        """Return whether any judge capabilities are enabled."""
        return bool(self.capabilities)

    def _empty_verdict(self, iteration: int, best_metric: float | None) -> JudgeVerdict:
        """Build an explicit empty-feedback verdict."""
        return JudgeVerdict(
            iteration=iteration,
            per_angle=[],
            key_recommendations=[],
            synthesized_feedback="",
            tool_call_count=0,
            wall_time_seconds=0.0,
            best_bic=best_metric,
        )

    def get_feedback_analysis(
        self,
        iteration: int,
        run_idx: int = 0,
        tag: str = "",
        best_model: str | None = None,
        best_metric: float | None = None,
        recovery_failures: list[dict] | None = None,
        prev_had_success: bool = True,
        **kwargs,
    ) -> dict:
        """R2: Run analysis phase only, return analysis text and trace.

        This is called once per iteration by the orchestrator, then the trace
        is reused for all persona-specific synthesis passes.

        Returns
        -------
        dict
            {
                'iteration': int,
                'analysis_text': str,
                'trace': list[dict],
                'full_trace': list[dict],
                'best_bic': float,
                'is_stuck': bool,
                'trajectory': list[dict],
            }
        """
        if not self._capabilities_enabled():
            return {
                "iteration": iteration,
                "analysis_text": "",
                "trace": [],
                "full_trace": [],
                "best_bic": best_metric,
                "is_stuck": False,
                "trajectory": [],
                "best_bic_str": (
                    f"{best_metric:.2f}" if best_metric is not None else "N/A"
                ),
                "wall_time": 0.0,
                "short_circuit": True,
                "no_capabilities": True,
            }

        if self.capabilities == ["random_feedback"]:
            return {
                "iteration": iteration,
                "analysis_text": _RANDOM_FEEDBACK_TEXT,
                "trace": [],
                "full_trace": [],
                "best_bic": best_metric,
                "is_stuck": False,
                "trajectory": [],
                "best_bic_str": (
                    f"{best_metric:.2f}" if best_metric is not None else "N/A"
                ),
                "wall_time": 0.0,
                "short_circuit": True,
                "random_feedback_only": True,
            }

        t0 = time.time()

        # --- Attempt to short-circuit if previous iteration had only recovery failures ---
        if recovery_failures and not prev_had_success and self.results_dir:
            shortcut = self._try_shortcut_from_recovery_failure(
                iteration=iteration,
                run_idx=run_idx,
                tag=tag,
                recovery_failures=recovery_failures,
            )
            if shortcut is not None:
                if self.verbose:
                    _console.print(
                        f"[bold magenta]◆ Judge (iter {iteration})[/bold magenta]"
                        f" — short-circuit: reusing verdict from earlier iteration"
                    )
                # For short-circuit, we still need to return analysis structure
                return {
                    "iteration": iteration,
                    "analysis_text": shortcut.get("analysis_text", ""),
                    "synthesized_feedback": shortcut.get("synthesized_feedback", {}),
                    "trace": [],
                    "full_trace": [],
                    "best_bic": shortcut.get("best_bic", best_metric),
                    "is_stuck": False,
                    "trajectory": [],
                    "wall_time": 0.0,
                    "metadata": shortcut.get("metadata", {}),
                    "shortcut_verdict_payload": shortcut.get(
                        "shortcut_verdict_payload", {}
                    ),
                    "short_circuit": True,  # Flag to skip re-synthesis
                }

        # --- Pre-compute iteration delta context ---
        from gecco.diagnostic_store.tools import get_bic_trajectory as _get_bic_traj

        iter_row = self.store.fetchone(
            "SELECT COUNT(*) AS n_total, "
            "COUNT(CASE WHEN status='ok' THEN 1 END) AS n_ok, "
            "COUNT(CASE WHEN status!='ok' THEN 1 END) AS n_failed, "
            "MIN(CASE WHEN status='ok' THEN metric_value END) AS best_iter "
            "FROM models WHERE iteration = ? AND split = 'train'",
            [iteration],
        )
        n_total = iter_row.get("n_total", 0) if iter_row else 0
        n_ok = iter_row.get("n_ok", 0) if iter_row else 0
        n_failed = iter_row.get("n_failed", 0) if iter_row else 0
        best_iter_bic_raw = iter_row.get("best_iter") if iter_row else None
        best_iter_bic = (
            f"{best_iter_bic_raw:.2f}" if best_iter_bic_raw is not None else "N/A"
        )

        traj = _get_bic_traj(self.store)
        trajectory_str = _format_trajectory(traj)

        # --- Stuck-search detection (R3: use configurable thresholds) ---
        is_stuck = _detect_stuck(
            traj,
            tol=self.stuck_search_cfg.tolerance,
            window=self.stuck_search_cfg.window,
        )

        # --- Build analysis context ---
        best_bic_str = f"{best_metric:.2f}" if best_metric is not None else "N/A"
        n_iterations = iteration + 1  # iterations seen so far (0-indexed)

        user_message = _JUDGE_USER_TEMPLATE.format(
            iteration=iteration,
            best_bic=best_bic_str,
            n_iterations=n_iterations,
            run_idx=run_idx,
            n_total=n_total,
            n_ok=n_ok,
            n_failed=n_failed,
            best_iter_bic=best_iter_bic,
            trajectory_str=trajectory_str,
        )

        if "citations" in self.capabilities:
            user_message += (
                "\n\nIMPORTANT — database conventions:\n"
                f"- \"iteration\" and \"run_idx\" are different fields.  \"iteration\" is the search step "
                f"(currently {iteration}); \"run_idx\" is the client/run identifier ({run_idx}).  Do NOT "
                "pass the iteration number as run_idx.  Most tools do not require run_idx — omit it "
                "to query across all runs.\n"
                f"- The best model is identified by its BIC value ({best_bic_str}).  Use get_best_models() "
                "to find it in the database and obtain its model_id.  Do not search by name.\n\n"
                "Model naming convention: Each model has a `name` field (e.g., \"rwg_alpha_beta\") "
                "set by the LLM that generated it. Names are NOT guaranteed unique across "
                "iterations. When referring to models in your synthesized_feedback, describe "
                "them by their mechanism AND include their name + BIC in parentheses, "
                "e.g., \"the model with separate learning rates for gains and losses "
                "(separate_lr_gain_loss, BIC=2847)\". The BIC disambiguates models that "
                "share a name. Never use the numeric model_id (e.g., \"model 5\", \"ID 11\") — "
                "the generator LLM cannot look up IDs."
            )

        # --- Stuck-search directive (R3: generic form for reuse across personas) ---
        if is_stuck:
            user_message += (
                f"\n\n\u26a0 Search appears stuck: best BIC has not improved by >{self.stuck_search_cfg.tolerance} "
                f"points over the last {self.stuck_search_cfg.window} iterations. "
                "Consider mechanistically-distant pivots: different model families, learning rules, "
                "or decision rules — not incremental tweaks. Explain why the pivot addresses a pattern "
                "in the data that current mechanisms miss."
            )

        # --- Previous verdict context (R8: truncate to avoid compounding bias) ---
        if self.results_dir:
            prev_verdict = self._load_previous_verdict(iteration, run_idx, tag)
            if prev_verdict is not None:
                prev_iter = prev_verdict.get("iteration", "?")
                prev_bic = prev_verdict.get("best_bic")
                prev_bic_str = f"{prev_bic:.2f}" if prev_bic is not None else "N/A"
                prev_recs = prev_verdict.get("key_recommendations", [])
                # R8: Drop synthesized_feedback to avoid rhetorical carryover bias
                rec_bullets = "".join(f"\n  - {r}" for r in prev_recs)
                guidance_label = (
                    "Key recommendations given"
                    if "recommendations" in self.capabilities
                    else "Prior guidance tracked"
                )
                guidance_text = (
                    f":{rec_bullets}"
                    if "recommendations" in self.capabilities
                    else f": {len(prev_recs)} items"
                )
                user_message += (
                    f"\n\nPrevious iteration ({prev_iter}) verdict:\n"
                    f"- Best BIC at that time: {prev_bic_str}\n"
                    f"- {guidance_label}{guidance_text}\n\n"
                    + (
                        "Use this to assess whether prior recommendations were followed, "
                        "whether BIC improved/stagnated/regressed, and to avoid repeating suggestions."
                        if "recommendations" in self.capabilities
                        else "Use this to assess whether prior guidance was followed, "
                        "whether BIC improved/stagnated/regressed, and to avoid repeating ideas."
                    )
                )

        if _is_summary_only_capability_set(self.capabilities):
            return {
                "iteration": iteration,
                "analysis_text": _build_summary_only_feedback(
                    {
                        "n_total": n_total,
                        "n_ok": n_ok,
                        "n_failed": n_failed,
                        "best_iter_bic": best_iter_bic_raw,
                        "best_bic": best_metric,
                        "trajectory_str": trajectory_str,
                    }
                ),
                "trace": [],
                "full_trace": [],
                "best_bic": best_metric,
                "best_iter_bic": best_iter_bic_raw,
                "is_stuck": is_stuck,
                "trajectory": traj,
                "best_bic_str": best_bic_str,
                "trajectory_str": trajectory_str,
                "n_total": n_total,
                "n_ok": n_ok,
                "n_failed": n_failed,
                "wall_time": 0.0,
                "summary_only": True,
            }

        # --- Run tool loop (analysis phase) ---
        if self.verbose:
            _console.print(
                f"[bold magenta]◆ Judge (iter {iteration})[/bold magenta]"
                f" — model: [cyan]{self.model_name}[/cyan]"
            )

        if self._tool_loop is not None:
            system_prompt = _build_judge_system_prompt(self.capabilities)
            final_text, trace, full_trace = self._tool_loop.run(
                self.store,
                system_prompt,
                user_message,
                self.max_tool_calls,
            )
        else:
            # Fallback: no tool calling — generate a plain text summary
            final_text = self._fallback_generate(user_message)
            trace = []
            full_trace = []

        wall_time = time.time() - t0

        if self.verbose:
            _console.print(
                f"[bold magenta]◆ Judge analysis[/bold magenta]"
                f" ({len(trace)} tool calls, {wall_time:.1f}s wall time)"
            )

        return {
            "iteration": iteration,
            "analysis_text": final_text,
            "trace": trace,
            "full_trace": full_trace,
            "best_bic": best_metric,
            "best_iter_bic": best_iter_bic_raw,
            "is_stuck": is_stuck,
            "trajectory": traj,
            "best_bic_str": best_bic_str,
            "trajectory_str": trajectory_str,
            "n_total": n_total,
            "n_ok": n_ok,
            "n_failed": n_failed,
            "wall_time": wall_time,
        }

    def synthesize_for_persona(
        self,
        analysis_data: dict,
        persona_name: str,
        persona_suffix: str = "",
        persona_config=None,
    ) -> tuple[str, dict]:
        """R2: Synthesize structured verdict for a specific persona.

        Parameters
        ----------
        analysis_data : dict
            Output from get_feedback_analysis()
        persona_name : str
            Name of the persona (e.g., 'exploit', 'explore')
        persona_suffix : str
            System prompt suffix for this persona
        persona_config : optional
            Full persona config section (from cfg.clients.<name>). If it contains
            a 'profile' key, those values override the default profile for this persona.

        Returns
        -------
        tuple[str, dict]
            (synthesized_feedback, verdict_dict)
        """
        analysis_text = analysis_data["analysis_text"]
        trace = analysis_data["trace"]
        iteration = analysis_data["iteration"]
        best_bic = analysis_data["best_bic"]
        is_stuck = analysis_data["is_stuck"]

        if _is_summary_only_capability_set(self.capabilities):
            verdict = JudgeVerdict(
                iteration=iteration,
                per_angle=[],
                key_recommendations=[],
                synthesized_feedback=_build_summary_only_feedback(analysis_data),
                tool_call_count=len(trace),
                wall_time_seconds=analysis_data.get("wall_time", 0.0),
                best_bic=best_bic,
            )
            return verdict.synthesized_feedback, verdict.__dict__

        # Build persona-specific synthesis prompt from profile
        profile = _PERSONA_PROFILES.get(persona_name, _DEFAULT_PROFILE).copy()
        # Allow config to override profile values
        if persona_config is not None:
            config_profile = getattr(persona_config, "profile", None)
            if config_profile is not None:
                profile.update(_profile_from_config(config_profile, profile))
        synthesis_prompt = _build_synthesis_prompt(
            self.capabilities,
            profile,
            persona_name,
            persona_suffix,
            is_stuck,
        )

        # Second pass: extract structured verdict
        if self.verbose:
            _console.print(
                f"[bold magenta]◆ Judge synthesis[/bold magenta]"
                f" for persona '{persona_name}'"
            )

        structured_text = self._request_structured_verdict_with_suffix(
            analysis_text,
            trace,
            synthesis_prompt,
            _build_judge_system_prompt(self.capabilities),
        )

        wall_time = analysis_data.get("wall_time", 0)
        verdict = _parse_verdict_from_text(
            structured_text, iteration, len(trace), wall_time
        )
        verdict.best_bic = best_bic
        verdict = _apply_capability_postprocessing(verdict, self.capabilities)

        if self.verbose:
            _console.print(
                f"[bold magenta]◆ Judge verdict[/bold magenta]"
                f" ({persona_name}): {len(verdict.synthesized_feedback)} chars"
            )

        return verdict.synthesized_feedback, verdict.__dict__

    def _request_structured_verdict_with_suffix(
        self,
        analysis_text: str,
        trace: list[dict],
        synthesis_prompt: str,
        system_prompt: str,
    ) -> str:
        """R2: Ask the LLM to format analysis as structured JSON with custom synthesis prompt."""
        if self.verbose:
            _console.print(
                "[bold magenta]◆ Extracting structured verdict...[/bold magenta]"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": analysis_text},
            {"role": "user", "content": synthesis_prompt},
        ]
        p = self.provider
        if any(
            x in p for x in ("openai", "gpt", "vllm", "kcl", "opencode", "openrouter")
        ):
            kwargs: dict = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
            }
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            resp = self.model.chat.completions.create(**kwargs)
            result = resp.choices[0].message.content or analysis_text
            if self.verbose:
                _console.print(
                    f"[dim]  └─ structured verdict: {len(result)} chars[/dim]"
                )
            return result
        elif "gemini" in p:
            try:
                from google.genai import types
            except ImportError:
                from google.generativeai import types  # type: ignore
            config_kwargs: dict = {"system_instruction": system_prompt}
            if self.max_tokens:
                config_kwargs["max_output_tokens"] = self.max_tokens
            if self.temperature is not None:
                config_kwargs["temperature"] = self.temperature
            contents = [
                {"role": "model", "parts": [{"text": analysis_text}]},
                {"role": "user", "parts": [{"text": synthesis_prompt}]},
            ]
            resp = self.model.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            result = resp.text.strip()
            if self.verbose:
                _console.print(
                    f"[dim]  └─ structured verdict: {len(result)} chars[/dim]"
                )
            return result
        return analysis_text

    def _load_previous_verdict(
        self, iteration: int, run_idx: int, tag: str
    ) -> dict[str, Any] | None:
        """Walk backwards through saved verdict files to find the most recent
        non-short-circuit verdict for this run, or return None.

        Skips short-circuit files (``short_circuit=True``) so that the context
        block reflects a real analysis, not a recycled one.
        """
        if self.results_dir is None:
            return None
        for k in range(iteration - 1, -1, -1):
            verdict_path = (
                self.results_dir / "judge" / f"iter{k}{tag}_run{run_idx}.json"
            )
            if verdict_path.exists():
                try:
                    with open(verdict_path) as f:
                        data = json.load(f)
                    if not data.get("short_circuit", False):
                        return data
                except (json.JSONDecodeError, IOError):
                    pass
        return None

    def _validate_cited_models(self, cited_models: list[dict]) -> list[dict]:
        """Check each cited model's name against the store.

        Returns the subset that could NOT be verified (for audit logging).
        """
        unverified = []
        for cm in cited_models:
            name = cm.get("name", "")
            if not name:
                continue
            rows = self.store.fetchall(
                "SELECT metric_value FROM models WHERE name = ? LIMIT 5", [name]
            )
            if not rows:
                unverified.append(cm)
        return unverified

    def _try_shortcut_from_recovery_failure(
        self,
        iteration: int,
        run_idx: int,
        tag: str,
        recovery_failures: list[dict],
    ) -> dict[str, Any] | None:
        """Attempt to short-circuit the full judge analysis when the previous
        iteration only produced recovery failures.

        Walks backwards through saved judge verdict files to find the most recent
        "real" (non-short-circuit) verdict, then appends a note about the recovery
        failures and returns a canonical shortcut payload reusing that verdict's
        structured content.

        Returns None if no previous verdict can be found, or if the short-circuit
        is not applicable; the caller should fall through to the full analysis.
        """
        # --- Locate the source verdict (most recent non-short-circuit verdict) ---
        source_verdict_iter = None
        source_verdict_data = None

        for k in range(iteration - 1, -1, -1):
            verdict_path = (
                self.results_dir / "judge" / f"iter{k}{tag}_run{run_idx}.json"
            )
            if verdict_path.exists():
                try:
                    with open(verdict_path) as f:
                        data = json.load(f)
                    # Stop at the first file that is NOT marked as short-circuit
                    if not data.get("short_circuit", False):
                        source_verdict_iter = k
                        source_verdict_data = data
                        break
                except (json.JSONDecodeError, IOError):
                    pass

        if source_verdict_data is None:
            return None

        # --- Build the failure note ---
        failure_lines = [
            "Update — previous iteration candidate(s) rejected for poor parameter recovery:"
        ]
        for fail in recovery_failures:
            name = fail.get("name", "unknown")
            mean_r = fail.get("mean_r")
            per_param_r = fail.get("per_param_r") or {}

            mean_r_str = f"{mean_r:.2f}" if mean_r is not None else "unknown"
            failure_lines.append(f"- {name}: mean r={mean_r_str}")

            # List worst 3 parameters
            if per_param_r:
                worst_params = sorted(
                    per_param_r.items(),
                    key=lambda x: x[1] if x[1] is not None else 1.0,
                )[:3]
                if worst_params:
                    param_strs = [
                        f"{pname} r={r:.2f}" if r is not None else f"{pname} r=unknown"
                        for pname, r in worst_params
                    ]
                    failure_lines.append(f"    worst: {', '.join(param_strs)}")

        failure_lines.append(
            "Do not repropose these mechanisms without addressing the "
            "identifiability issues."
        )
        failure_note = "\n".join(failure_lines)

        previous_feedback = source_verdict_data.get("synthesized_feedback", {})
        if isinstance(previous_feedback, str):
            previous_feedback = {"default": previous_feedback}
        elif not isinstance(previous_feedback, dict):
            previous_feedback = {"default": str(previous_feedback)}

        combined_feedback = {
            persona_name: (
                f"{failure_note}\n\n"
                f"--- Previous verdict (state unchanged since iter {source_verdict_iter}) ---\n"
                f"{feedback_text}"
            )
            for persona_name, feedback_text in previous_feedback.items()
        }
        if not combined_feedback:
            combined_feedback = {
                "default": (
                    f"{failure_note}\n\n"
                    f"--- Previous verdict (state unchanged since iter {source_verdict_iter}) ---"
                )
            }

        metadata = dict(source_verdict_data.get("metadata", {}) or {})
        metadata.update(
            {
                "shortcut_reason": "recovery_failure",
                "source_iteration": source_verdict_iter,
                "recovery_failures": recovery_failures,
            }
        )

        return {
            "analysis_text": next(iter(combined_feedback.values()), ""),
            "synthesized_feedback": combined_feedback,
            "metadata": metadata,
            "shortcut_verdict_payload": {
                "per_angle": source_verdict_data.get("per_angle", []) or [],
                "key_recommendations": source_verdict_data.get(
                    "key_recommendations", []
                )
                or [],
            },
            "best_bic": source_verdict_data.get("best_bic"),
        }

    def _fallback_generate(self, user_message: str) -> str:
        """Simple one-shot generation for backends without tool calling."""
        system_prompt = _build_judge_system_prompt(self.capabilities)
        context_parts = [system_prompt, "\n\n", user_message]

        # Pull a minimal context from the store directly
        try:
            from gecco.diagnostic_store.tools import get_bic_trajectory, get_best_models

            traj = get_bic_trajectory(self.store)
            best = get_best_models(self.store, k=5)
            context_parts.append(f"\n\nBIC trajectory: {json.dumps(traj, default=str)}")
            context_parts.append(f"\n\nTop models: {json.dumps(best, default=str)}")
        except Exception:
            pass

        prompt = "".join(context_parts)

        if self.tokenizer is not None:
            # HuggingFace
            max_new = getattr(
                self.cfg.llm,
                "max_output_tokens",
                getattr(self.cfg.llm, "max_tokens", 2048),
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(
                **inputs, max_new_tokens=max_new, do_sample=True
            )
            return self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Rewrite tool-dependent instructions in the user message for the
        # no-tools case so the LLM doesn't attempt queries it can't make.
        user_message_no_tools = user_message.replace(
            "Please query the diagnostic database to analyse this iteration "
            "from all six angles, then produce your verdict.",
            "You do NOT have access to diagnostic tool calls.  Use the "
            "pre-computed statistics and trajectory printed above to "
            "analyse this iteration, then produce your verdict.",
        )

        p = self.provider
        if any(
            x in p for x in ("openai", "gpt", "vllm", "kcl", "opencode", "openrouter")
        ):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message_no_tools},
            ]
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
            }
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            resp = self.model.chat.completions.create(**kwargs)
            content = (
                resp.choices[0].message.content
                if resp.choices and resp.choices[0].message
                else None
            )
            if self.verbose:
                _console.print(
                    f"[dim]  └─ fallback analysis: "
                    f"{len(content) if content else 0} chars[/dim]"
                )
            return content or ""
        elif "gemini" in p:
            try:
                from google.genai import types
            except ImportError:
                from google.generativeai import types
            contents = [{"role": "user", "parts": [{"text": user_message_no_tools}]}]
            resp = self.model.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=self.max_tokens or None,
                    temperature=self.temperature,
                ),
            )
            return resp.text
        return "Judge feedback unavailable (no tool-calling backend configured)."

    def _save_trace(
        self,
        verdict: JudgeVerdict,
        trace: list[dict],
        iteration: int,
        run_idx: int,
        tag: str = "",
        full_trace: list[dict] | None = None,
        extra_payload: dict | None = None,
    ) -> None:
        """Write the judge trace to results/{task}/judge/iterN_runX.json.

        Parameters
        ----------
        extra_payload:
            Optional dict of additional fields to merge into the JSON payload.
            Used for short-circuit markers and source iteration references.
        """
        judge_dir = self.results_dir / "judge"
        judge_dir.mkdir(parents=True, exist_ok=True)

        tag = tag or ""
        fname = judge_dir / f"iter{iteration}{tag}_run{run_idx}.json"

        payload = {
            "iteration": iteration,
            "run_idx": run_idx,
            "tag": tag,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_call_count": verdict.tool_call_count,
            "wall_time_seconds": verdict.wall_time_seconds,
            "best_bic": verdict.best_bic,
            "tool_call_trace": trace,
            "full_trace": full_trace if full_trace is not None else [],
            "per_angle": [a.model_dump() for a in verdict.per_angle],
            "key_recommendations": verdict.key_recommendations,
            "synthesized_feedback": verdict.synthesized_feedback,
        }
        if extra_payload:
            payload.update(extra_payload)
        with open(fname, "w") as f:
            json.dump(payload, f, indent=2, default=str)
