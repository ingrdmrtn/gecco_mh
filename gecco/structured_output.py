"""
Structured output utilities for GeCCo model generation.

Provides JSON schema definitions, response parsing with fallback to regex
extraction, and reflection/self-critique prompt construction.
"""

import json
import re
from datetime import datetime
from typing import List, Dict, Optional

from gecco.utils import extract_model_code


def _log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


# ============================================================
# JSON Schema for structured model generation responses
# ============================================================

def get_model_schema(n_models: int, include_analysis: bool = True) -> dict:
    """
    Build the JSON schema for the model generation response.

    Parameters
    ----------
    n_models : int
        Number of models expected.
    include_analysis : bool
        Whether to include the analysis scratchpad field.

    Returns
    -------
    dict
        JSON schema compatible with OpenAI/Gemini structured output.
    """
    model_properties = {
        "name": {
            "type": "string",
            "description": (
                "Concise descriptive snake_case name (2-4 words) capturing the "
                "model's key mechanism, e.g. 'dual_lr_perseveration', "
                "'bayesian_transition_learner', 'attention_weighted_mbmf'"
            ),
        },
        "rationale": {
            "type": "string",
            "description": (
                "One sentence explaining the model's hypothesis and why "
                "this architecture might improve on previous models"
            ),
        },
        "parameters": {
            "type": "array",
            "description": (
                "List of model parameters with their names and bounds. "
                "Must match the parameters unpacked from model_parameters "
                "in the code, in the same order."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Parameter name matching the variable in the code",
                    },
                    "lower_bound": {
                        "type": "number",
                        "description": "Lower bound for optimization (e.g. 0)",
                    },
                    "upper_bound": {
                        "type": "number",
                        "description": "Upper bound for optimization (e.g. 1 for rates, 10 for inverse temperature)",
                    },
                },
                "required": ["name", "lower_bound", "upper_bound"],
                "additionalProperties": False,
            },
        },
        "code": {
            "type": "string",
            "description": (
                "Complete Python function definition. The function must be "
                "named cognitive_modelN (where N is its position in the list, "
                "starting from 1)"
            ),
        },
    }
    required = ["name", "rationale", "parameters", "code"]

    if include_analysis:
        model_properties["analysis"] = {
            "type": "string",
            "description": (
                "Free-form reasoning about the search state: what mechanisms "
                "have been tried, what's working, what's missing, and why "
                "this particular architecture might improve fit. Think step "
                "by step before writing code."
            ),
        }
        required = ["analysis", "name", "rationale", "parameters", "code"]

    return {
        "type": "object",
        "properties": {
            "models": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": model_properties,
                    "required": required,
                    "additionalProperties": False,
                },
                "minItems": n_models,
                "maxItems": n_models,
            },
        },
        "required": ["models"],
        "additionalProperties": False,
    }


def get_schema_instructions(n_models: int, include_analysis: bool = True) -> str:
    """
    Build prompt instructions describing the expected JSON output format.

    Used as a fallback for providers that don't support native structured
    output (vLLM, HuggingFace) — the schema is described in the prompt.
    """
    analysis_field = ""
    analysis_example = ""
    if include_analysis:
        analysis_field = (
            '      "analysis": "Your reasoning about what mechanisms to try and why, '
            'based on the search state and feedback...",\n'
        )
        analysis_example = (
            "  - `analysis`: Free-form reasoning about what's been tried, "
            "what's working, and why this architecture might improve fit. "
            "Think step by step before writing code.\n"
        )

    return f"""### Output Format
You MUST respond with valid JSON in exactly this format (no markdown, no extra text):

{{
  "models": [
    {{
{analysis_field}      "name": "descriptive_snake_case_name",
      "rationale": "One sentence explaining the model's hypothesis",
      "parameters": [
        {{"name": "alpha", "lower_bound": 0, "upper_bound": 1}},
        {{"name": "beta", "lower_bound": 0, "upper_bound": 10}}
      ],
      "code": "def cognitive_model1(action_1, state, action_2, reward, model_parameters):\\n    ..."
    }}
  ]
}}

Field descriptions:
{analysis_example}  - `name`: Concise descriptive snake_case name (2-4 words) capturing the key mechanism (e.g. `dual_lr_perseveration`, `bayesian_transition_learner`, `attention_weighted_mbmf`). Must be unique across all models you propose.
  - `rationale`: One sentence explaining the model's hypothesis and why this architecture might improve on previous models.
  - `parameters`: List of model parameters with bounds. Each entry has `name` (matching the variable in code), `lower_bound`, and `upper_bound`. Must match the parameters unpacked from `model_parameters` in the same order.
  - `code`: Complete Python function definition. Functions must be named `cognitive_model1`, `cognitive_model2`, etc.

You must provide exactly {n_models} model(s) in the `models` array.
Important: Ensure all string values are properly escaped JSON (newlines as \\n, quotes as \\", etc.)."""


# ============================================================
# Response parsing
# ============================================================

def parse_model_response(
    text: str,
    n_models: int,
    structured_output: bool = True,
) -> tuple:
    """
    Parse an LLM response into a list of model dicts.

    Attempts JSON parsing first (if structured_output is True), then
    falls back to regex extraction with generic names.

    Parameters
    ----------
    text : str
        Raw LLM response text.
    n_models : int
        Number of models expected.
    structured_output : bool
        Whether structured JSON output was requested.

    Returns
    -------
    tuple of (list of dict, bool)
        Models list and a flag indicating whether JSON parsing succeeded.
        Falls back to regex extraction with generic names if JSON fails.
        Each dict has keys: name, rationale, code, analysis (optional).
    """
    if structured_output:
        parsed = _try_parse_json(text)
        if parsed is not None:
            return parsed, True

        _log("[GeCCo] Structured output parsing failed — falling back to regex extraction")

    # Fallback: regex extraction
    return _fallback_regex_extraction(text, n_models), False


def _try_parse_json(text: str) -> Optional[List[Dict]]:
    """Attempt to parse JSON from the response text."""
    # Try direct JSON parse
    try:
        data = json.loads(text.strip())
        if "models" in data and isinstance(data["models"], list):
            return _validate_models(data["models"])
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1).strip())
            if "models" in data and isinstance(data["models"], list):
                return _validate_models(data["models"])
        except json.JSONDecodeError:
            pass

    # Try finding a JSON object in the text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0))
            if "models" in data and isinstance(data["models"], list):
                return _validate_models(data["models"])
        except json.JSONDecodeError:
            pass

    return None


def _validate_models(models: list) -> Optional[List[Dict]]:
    """Validate and normalize parsed model dicts."""
    result = []
    for i, m in enumerate(models):
        if not isinstance(m, dict):
            return None
        code = m.get("code", "")
        if not code:
            return None

        result.append({
            "name": m.get("name", f"cognitive_model{i + 1}"),
            "rationale": m.get("rationale", ""),
            "code": code,
            "analysis": m.get("analysis", ""),
        })
    return result if result else None


def _fallback_regex_extraction(text: str, n_models: int) -> List[Dict]:
    """Fall back to regex extraction with generic names."""
    models = []
    for i in range(1, n_models + 1):
        code = extract_model_code(text, i)
        if code:
            models.append({
                "name": f"cognitive_model{i}",
                "rationale": "",
                "code": code,
                "analysis": "",
            })
    return models


# ============================================================
# Reflection / self-critique prompt
# ============================================================

def build_reflection_prompt(models: List[Dict], guardrails: list = None,
                            use_json: bool = True) -> str:
    """
    Build a self-critique prompt for reviewing generated models.

    Parameters
    ----------
    models : list of dict
        Parsed model dicts from the initial generation.
    guardrails : list of str, optional
        Model guardrails from config.
    use_json : bool
        If True, ask for JSON output (for structured output providers).
        If False, ask for plain Python code blocks (regex-friendly fallback).

    Returns
    -------
    str
        Prompt for the reflection/self-critique LLM call.
    """
    models_text = ""
    for i, m in enumerate(models):
        models_text += f"\n### Model {i + 1}: {m['name']}\n"
        if m.get("rationale"):
            models_text += f"Rationale: {m['rationale']}\n"
        models_text += f"```python\n{m['code']}\n```\n"

    guardrails_text = ""
    if guardrails:
        guardrails_text = (
            "\n### Guardrails to check against\n"
            + "\n".join(f"- {g}" for g in guardrails)
        )

    checks = """Review the following cognitive models. For each model, check:

1. **Parameter usage**: Are all parameters meaningfully used? Flag any that are defined but never influence the output.
2. **Numerical stability**: Check for division by zero, log of zero or negative values, exp overflow, NaN propagation.
3. **Docstring accuracy**: Do the parameter bounds in the docstring match how parameters are actually used?
4. **Logic errors**: Are Q-value updates, probability calculations, and learning rules implemented correctly?
5. **Code quality**: Any obvious bugs, off-by-one errors, or incorrect array indexing?"""

    if use_json:
        output_instructions = (
            "\nFor each model, either confirm it is correct or provide a corrected version.\n"
            "Respond with the same JSON format as before — a JSON object with a \"models\" array.\n"
            "If a model is correct, return it unchanged. If it needs fixes, return the corrected "
            "version with an updated rationale noting what was fixed.\n"
            "Do NOT change model names."
        )
    else:
        output_instructions = (
            "\nFor each model, either confirm it is correct or provide a corrected version.\n"
            "Return each (possibly corrected) function in a ```python ... ``` code block, "
            "in the same order as above. Keep function names exactly as they are "
            "(cognitive_model1, cognitive_model2, etc.).\n"
            "If a model is correct, repeat it unchanged."
        )

    return f"{checks}\n{guardrails_text}\n{models_text}\n{output_instructions}"


# ============================================================
# Provider-specific response format helpers
# ============================================================

def get_gemini_schema(schema: dict) -> dict:
    """Strip fields unsupported by Gemini (additionalProperties, minItems, maxItems)."""
    def _clean(obj):
        if isinstance(obj, dict):
            return {
                k: _clean(v) for k, v in obj.items()
                if k not in ("additionalProperties", "minItems", "maxItems")
            }
        if isinstance(obj, list):
            return [_clean(item) for item in obj]
        return obj
    return _clean(schema)


def get_openai_response_format(schema: dict) -> dict:
    """Build OpenAI Responses API text format spec."""
    return {
        "type": "json_schema",
        "name": "cognitive_models",
        "schema": schema,
        "strict": True,
    }


def get_vllm_response_format() -> dict:
    """Build vLLM-compatible response format."""
    return {"type": "json_object"}
