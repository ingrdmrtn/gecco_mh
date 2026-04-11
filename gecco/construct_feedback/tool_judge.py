"""
Tool-using judge for GeCCo.

The :class:`ToolUsingJudge` queries the diagnostic store through a set of
read-only tools, analyses the iteration from multiple analytical angles, and
returns a :class:`JudgeVerdict` whose ``synthesized_feedback`` field is
drop-in compatible with the text returned by
:class:`gecco.construct_feedback.feedback.FeedbackGenerator`.

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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from rich.console import Console

from gecco.diagnostic_store.tools import TOOL_SCHEMAS, dispatch_tool

_console = Console()

# Hard cap on any single tool result fed back into the message history.
# Prevents an unexpectedly large result from reopening the context-limit wound.
_MAX_TOOL_RESULT_CHARS = 40_000


def _cap_tool_result(result_str: str) -> str:
    """Truncate *result_str* if it exceeds _MAX_TOOL_RESULT_CHARS."""
    if len(result_str) <= _MAX_TOOL_RESULT_CHARS:
        return result_str
    return (
        result_str[:_MAX_TOOL_RESULT_CHARS]
        + " ... [truncated: result too large; use more specific filters]"
    )


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
    if isinstance(result, dict):
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
database through tool calls, then synthesise actionable feedback for the next iteration.

You will analyse from six angles — call tools to gather evidence for each:

1. **Statistical fit quality** — examine the BIC trajectory, whether it is still \
improving, which models have the best per-participant fit, whether different participants \
prefer different models, and whether any participants are outliers.

2. **Parameter identifiability** — check parameter recovery diagnostics for the best \
models. Flag parameters with low recovery r and suggest whether they should be removed \
or reparameterised.

3. **Predictive adequacy** — inspect PPC records if available. Identify which observed \
statistics fall outside the 95% predictive interval, and inspect block-level residuals \
to identify where in the task the model systematically underperforms.

4. **Individual differences** — examine which parameters predict self-report scores \
(R²). Also check whether the best-fitting model differs across participants; high \
heterogeneity can indicate a need for hybrid or mixture mechanisms. Highlight parameters \
with strong individual-differences signal and suggest building on them.

5. **Mechanistic / theoretical coherence** — read the code of the best 2–3 models. \
Assess whether the mechanisms are psychologically interpretable and parsimonious.

6. **Coverage** — what types of mechanisms (learning rules, decision rules, memory, \
attention) have been tried across iterations? What has been neglected?

After gathering evidence across all angles, produce:
- A brief per-angle summary (findings + confidence).
- A list of 3–5 concrete recommendations for the next iteration.
- A synthesized_feedback paragraph (≤ 300 words) to improve the next iteration of \
    model development.

Be specific: cite model names, parameter names, BIC values, and r values from the data.

Quantify your confidence: If you do not have much data to support an angle, say so and \
give a low confidence rating. If the evidence is strong, give a high confidence rating.
"""

_JUDGE_USER_TEMPLATE = """The model search has just completed iteration {iteration}.
Current best BIC: {best_bic}
Total iterations so far: {n_iterations}
Run index (run_idx) for this client: {run_idx}

IMPORTANT — database conventions:
- "iteration" and "run_idx" are different fields.  "iteration" is the search step \
(currently {iteration}); "run_idx" is the client/run identifier ({run_idx}).  Do NOT \
pass the iteration number as run_idx.  Most tools do not require run_idx — omit it \
to query across all runs.
- The best model is identified by its BIC value ({best_bic}).  Use get_best_models() \
to find it in the database and obtain its model_id.  Do not search by name.

Please query the diagnostic database to analyse this iteration from all six angles, \
then produce your verdict.
"""


# ======================================================================
# Backend-specific tool loops
# ======================================================================

class _OpenAIToolLoop:
    """Tool-calling loop for OpenAI-compatible backends."""

    def __init__(self, client, model_name: str, max_tokens: int,
                 temperature: float | None, verbose: bool = False):
        self.client = client
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose

    def run(self, store, system_prompt: str, user_message: str,
            max_tool_calls: int) -> tuple[str, list[dict]]:
        """Run the tool loop and return (final_text, tool_call_trace)."""
        planning_instruction = (
            "Before calling any tools, first write a brief plan (a few sentences): "
            "outline which of the six angles you will investigate and which tools "
            "you expect to call. Do not call any tools in this first message — "
            "you will be able to call them in your next message."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message + "\n\n" + planning_instruction},
        ]
        trace: list[dict] = []
        n_calls = 0
        planning_done = False

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
                messages.append({
                    "role": "user",
                    "content": (
                        "Good. Now proceed: call the diagnostic tools one at a "
                        "time to gather the evidence you need. After each tool "
                        "result, briefly reflect on what you learned before "
                        "deciding on the next call."
                    ),
                })
                continue

            if not msg.tool_calls:
                # Model finished — return the text content
                return msg.content or "", trace

            # Execute each tool call
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                result = dispatch_tool(store, tool_name, args)
                result_str = _cap_tool_result(json.dumps(result, default=str))

                if self.verbose:
                    _console.print(_format_tool_call(tool_name, args))
                    _console.print(f"  [dim]└─ {_format_tool_result(result)}[/dim]")

                trace.append({
                    "tool": tool_name,
                    "args": args,
                    "result_summary": result_str[:500],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })
                n_calls += 1
                if n_calls >= max_tool_calls:
                    break

        # Hit the cap — ask for a final synthesis without tools
        messages.append({
            "role": "user",
            "content": (
                "You have reached the tool call limit. "
                "Please now synthesise your findings into the final verdict."
            ),
        })
        kwargs_final: dict = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            kwargs_final["temperature"] = self.temperature
        final_resp = self.client.chat.completions.create(**kwargs_final)
        return final_resp.choices[0].message.content or "", trace


class _GeminiToolLoop:
    """Tool-calling loop for Gemini backends."""

    def __init__(self, client, model_name: str, max_tokens: int,
                 temperature: float | None, verbose: bool = False):
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

    def run(self, store, system_prompt: str, user_message: str,
            max_tool_calls: int) -> tuple[str, list[dict]]:
        """Run the Gemini tool loop."""
        try:
            from google.genai import types
        except ImportError:
            from google.generativeai import types  # type: ignore

        tools = self._build_gemini_tools()
        contents = [{"role": "user", "parts": [{"text": user_message}]}]
        trace: list[dict] = []
        n_calls = 0

        config_kwargs: dict = {
            "system_instruction": system_prompt,
            "tools": tools,
        }
        if self.max_tokens:
            config_kwargs["max_output_tokens"] = self.max_tokens
        if self.temperature is not None:
            config_kwargs["temperature"] = self.temperature

        while n_calls < max_tool_calls:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )

            # Verbose: print assistant text (if any)
            if self.verbose:
                text_parts = [
                    p.text for p in resp.candidates[0].content.parts
                    if hasattr(p, "text") and p.text
                ]
                if text_parts:
                    for line in "\n".join(text_parts).splitlines():
                        _console.print(f"[dim]│[/dim] {line}")

            # Check for function calls
            has_function_calls = False
            for part in resp.candidates[0].content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    has_function_calls = True
                    fc = part.function_call
                    args = dict(fc.args) if fc.args else {}
                    result = dispatch_tool(store, fc.name, args)
                    result_str = _cap_tool_result(json.dumps(result, default=str))

                    if self.verbose:
                        _console.print(_format_tool_call(fc.name, args))
                        _console.print(f"  [dim]└─ {_format_tool_result(result)}[/dim]")

                    trace.append({
                        "tool": fc.name,
                        "args": args,
                        "result_summary": result_str[:500],
                    })

                    # Append function response
                    contents.append({
                        "role": "model",
                        "parts": [{"function_call": {"name": fc.name, "args": args}}],
                    })
                    contents.append({
                        "role": "tool",
                        "parts": [{"function_response": {
                            "name": fc.name, "response": {"result": result_str}
                        }}],
                    })
                    n_calls += 1
                    if n_calls >= max_tool_calls:
                        break

            if not has_function_calls:
                return resp.text.strip(), trace

        # Final synthesis pass
        contents.append({
            "role": "user",
            "parts": [{"text": (
                "You have reached the tool call limit. "
                "Please now synthesise your findings into the final verdict."
            )}],
        })
        # Remove tools from final call to force text response
        final_config = {k: v for k, v in config_kwargs.items() if k != "tools"}
        final_resp = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(**final_config),
        )
        return final_resp.text.strip(), trace


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
        "synthesized_feedback": {"type": "string"},
    },
    "required": ["per_angle", "key_recommendations", "synthesized_feedback"],
}


def _parse_verdict_from_text(text: str, iteration: int,
                              tool_call_count: int,
                              wall_time: float) -> JudgeVerdict:
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
            return JudgeVerdict(
                iteration=iteration,
                per_angle=per_angle,
                key_recommendations=data.get("key_recommendations", []),
                synthesized_feedback=data.get("synthesized_feedback", text),
                tool_call_count=tool_call_count,
                wall_time_seconds=wall_time,
            )
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

_SYNTHESIS_PROMPT = """Based on your analysis above, please produce a final verdict as a JSON object with this exact structure:

```json
{
  "per_angle": [
    {
      "angle": "Statistical fit quality",
      "findings": "...",
      "supporting_tool_calls": ["get_bic_trajectory", "get_best_models"],
      "confidence": "high"
    }
  ],
  "key_recommendations": [
    "Recommendation 1...",
    "Recommendation 2..."
  ],
  "synthesized_feedback": "A concise paragraph (≤ 300 words) ready to be injected into the next model-generation prompt."
}
```

Include one entry in per_angle for each of the six angles:
1. Statistical fit quality
2. Parameter identifiability
3. Predictive adequacy
4. Individual differences
5. Mechanistic / theoretical coherence
6. Coverage

The synthesized_feedback must be actionable and specific, citing model names and metric values.
"""


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

    def __init__(self, cfg, diagnostic_store, model, tokenizer=None,
                 results_dir: str | Path | None = None):
        self.cfg = cfg
        self.store = diagnostic_store
        self.model = model
        self.tokenizer = tokenizer
        self.results_dir = Path(results_dir) if results_dir else None

        judge_cfg = getattr(cfg, "judge", None)
        self.max_tool_calls: int = getattr(judge_cfg, "max_tool_calls", 20) if judge_cfg else 20
        self.verbose: bool = bool(getattr(judge_cfg, "verbose", False)) if judge_cfg else False
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

    def _build_tool_loop(self):
        """Instantiate the right backend tool loop."""
        p = self.provider
        if any(x in p for x in ("openai", "gpt", "vllm", "kcl", "opencode", "openrouter")):
            return _OpenAIToolLoop(
                self.model, self.model_name, self.max_tokens, self.temperature,
                verbose=self.verbose,
            )
        elif "gemini" in p:
            return _GeminiToolLoop(
                self.model, self.model_name, self.max_tokens, self.temperature,
                verbose=self.verbose,
            )
        else:
            # HuggingFace / unknown: fall back to OpenAI-compatible if possible,
            # otherwise return None and we'll do a simple text generation.
            return None

    def get_feedback(
        self,
        iteration: int,
        run_idx: int = 0,
        tag: str = "",
        best_model: str | None = None,
        best_metric: float | None = None,
        recovery_failures: list[dict] | None = None,
        prev_had_success: bool = True,
        **kwargs,
    ) -> JudgeVerdict:
        """Run the tool-using judge and return a :class:`JudgeVerdict`.

        Parameters
        ----------
        iteration:
            Current iteration index.
        run_idx:
            Run index (for distributed mode).
        tag:
            File tag used to disambiguate distributed or participant-specific runs.
        best_model:
            Code of the current best model (for context).
        best_metric:
            Current best BIC/metric value.
        recovery_failures:
            List of recovery failures from the previous iteration. If non-empty and
            prev_had_success is False, the judge will attempt to short-circuit and
            reuse the previous verdict with a failure note appended.
        prev_had_success:
            Whether the previous iteration had any successful model fits.

        Returns
        -------
        JudgeVerdict
            Contains ``synthesized_feedback`` ready for prompt injection.
        """
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
                return shortcut

        # --- Build context ---
        best_bic_str = f"{best_metric:.2f}" if best_metric is not None else "N/A"
        n_iterations = iteration + 1  # iterations seen so far (0-indexed)

        user_message = _JUDGE_USER_TEMPLATE.format(
            iteration=iteration,
            best_bic=best_bic_str,
            n_iterations=n_iterations,
            run_idx=run_idx,
        )

        # --- Run tool loop ---
        if self.verbose:
            _console.print(
                f"[bold magenta]◆ Judge (iter {iteration})[/bold magenta]"
                f" — model: [cyan]{self.model_name}[/cyan]"
            )

        if self._tool_loop is not None:
            final_text, trace = self._tool_loop.run(
                self.store,
                _JUDGE_SYSTEM_PROMPT,
                user_message,
                self.max_tool_calls,
            )
        else:
            # Fallback: no tool calling — generate a plain text summary
            final_text = self._fallback_generate(user_message)
            trace = []

        # --- Second pass: extract structured verdict ---
        if self._tool_loop is not None:
            structured_text = self._request_structured_verdict(final_text, trace)
        else:
            structured_text = final_text

        wall_time = time.time() - t0

        if self.verbose:
            _console.print(
                f"[bold magenta]◆ Judge verdict[/bold magenta]"
                f" ({len(trace)} tool calls, {wall_time:.1f}s wall time)"
            )

        verdict = _parse_verdict_from_text(
            structured_text, iteration, len(trace), wall_time
        )

        # --- Persist audit trace ---
        if self.results_dir:
            self._save_trace(verdict, trace, iteration, run_idx, tag)

        return verdict

    def _try_shortcut_from_recovery_failure(
        self,
        iteration: int,
        run_idx: int,
        tag: str,
        recovery_failures: list[dict],
    ) -> JudgeVerdict | None:
        """Attempt to short-circuit the full judge analysis when the previous
        iteration only produced recovery failures.

        Walks backwards through saved judge verdict files to find the most recent
        "real" (non-short-circuit) verdict, then appends a note about the recovery
        failures and returns a new JudgeVerdict reusing that verdict's content.

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
                    failure_lines.append(
                        f"    worst: {', '.join(param_strs)}"
                    )

        failure_lines.append(
            "Do not repropose these mechanisms without addressing the "
            "identifiability issues."
        )
        failure_note = "\n".join(failure_lines)

        # --- Compose synthesised feedback ---
        previous_feedback = source_verdict_data.get("synthesized_feedback", "")
        combined_feedback = (
            f"{failure_note}\n\n"
            f"--- Previous verdict (state unchanged since iter {source_verdict_iter}) ---\n"
            f"{previous_feedback}"
        )

        # --- Construct verdict with source data ---
        per_angle = [
            AngleAnalysis(
                angle=a.get("angle", ""),
                findings=a.get("findings", ""),
                supporting_tool_calls=a.get("supporting_tool_calls", []),
                confidence=a.get("confidence", "medium"),
            )
            for a in source_verdict_data.get("per_angle", [])
        ]

        verdict = JudgeVerdict(
            iteration=iteration,
            per_angle=per_angle,
            key_recommendations=source_verdict_data.get("key_recommendations", []),
            synthesized_feedback=combined_feedback,
            tool_call_count=0,
            wall_time_seconds=0.1,  # Very small wall time for shortcut
        )

        # --- Save the shortcut trace ---
        if self.results_dir:
            self._save_trace(
                verdict,
                [],  # No tool calls
                iteration,
                run_idx,
                tag,
                extra_payload={
                    "short_circuit": True,
                    "source_iter": source_verdict_iter,
                    "recovery_failures": recovery_failures,
                },
            )

        return verdict

    def _request_structured_verdict(self, analysis_text: str,
                                      trace: list[dict]) -> str:
        """Ask the LLM to format its analysis as structured JSON."""
        if self.verbose:
            _console.print("[bold magenta]◆ Extracting structured verdict...[/bold magenta]")

        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "assistant", "content": analysis_text},
            {"role": "user", "content": _SYNTHESIS_PROMPT},
        ]
        p = self.provider
        if any(x in p for x in ("openai", "gpt", "vllm", "kcl", "opencode", "openrouter")):
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
                _console.print(f"[dim]  └─ structured verdict: {len(result)} chars[/dim]")
            return result
        elif "gemini" in p:
            try:
                from google.genai import types
            except ImportError:
                from google.generativeai import types  # type: ignore
            config_kwargs: dict = {"system_instruction": _JUDGE_SYSTEM_PROMPT}
            if self.max_tokens:
                config_kwargs["max_output_tokens"] = self.max_tokens
            if self.temperature is not None:
                config_kwargs["temperature"] = self.temperature
            contents = [
                {"role": "model", "parts": [{"text": analysis_text}]},
                {"role": "user", "parts": [{"text": _SYNTHESIS_PROMPT}]},
            ]
            resp = self.model.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            result = resp.text.strip()
            if self.verbose:
                _console.print(f"[dim]  └─ structured verdict: {len(result)} chars[/dim]")
            return result
        return analysis_text

    def _fallback_generate(self, user_message: str) -> str:
        """Simple one-shot generation for backends without tool calling."""
        context_parts = [_JUDGE_SYSTEM_PROMPT, "\n\n", user_message]

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
            max_new = getattr(self.cfg.llm, "max_output_tokens",
                               getattr(self.cfg.llm, "max_tokens", 2048))
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=True)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
        return "Judge feedback unavailable (no tool-calling backend configured)."

    def _save_trace(self, verdict: JudgeVerdict, trace: list[dict],
                    iteration: int, run_idx: int, tag: str = "",
                    extra_payload: dict | None = None) -> None:
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
            "tool_call_trace": trace,
            "per_angle": [a.model_dump() for a in verdict.per_angle],
            "key_recommendations": verdict.key_recommendations,
            "synthesized_feedback": verdict.synthesized_feedback,
        }
        if extra_payload:
            payload.update(extra_payload)
        with open(fname, "w") as f:
            json.dump(payload, f, indent=2, default=str)
