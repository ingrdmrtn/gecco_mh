"""
Structured output utilities for GeCCo model generation.

Provides JSON schema definitions, response parsing with fallback to regex
extraction, and reflection/self-critique prompt construction.
"""

import ast
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any

from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError

from gecco.utils import extract_model_code


# ============================================================
# Pydantic schemas for LLM response validation
# ============================================================


class ModelParameter(BaseModel):
    """Schema for a single model parameter with bounds."""

    name: str = Field(
        ...,
        min_length=1,
        pattern=r"^[a-z_][a-z0-9_]*$",
        description="Parameter name in snake_case",
    )
    lower_bound: float = Field(
        ..., ge=-100, le=1000, description="Lower bound for optimization"
    )
    upper_bound: float = Field(
        ..., ge=-100, le=1000, description="Upper bound for optimization"
    )

    @model_validator(mode="after")
    def bounds_order_valid(self):
        if self.upper_bound < self.lower_bound:
            raise ValueError(
                f"upper_bound ({self.upper_bound}) must be >= lower_bound ({self.lower_bound})"
            )
        return self


class LLMModelResponse(BaseModel):
    """Schema for a single model's JSON response from the LLM."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=50,
        pattern=r"^[a-z_][a-z0-9_]*$",
        description="Snake_case model name (2-4 words)",
    )
    rationale: str = Field(
        ..., min_length=10, description="One sentence explaining the model's hypothesis"
    )
    parameters: List[ModelParameter] = Field(
        ..., min_length=1, description="List of model parameters with bounds"
    )
    code: str = Field(
        ..., min_length=10, description="Complete Python function definition"
    )

    @field_validator("code")
    @classmethod
    def validate_syntax(cls, v):
        """Validate Python syntax using AST parser. Strips markdown fences first."""
        # Strip markdown code fences if present
        code_match = re.search(r"```\s*python(.*?)```", v, flags=re.S | re.I)
        if code_match:
            v = code_match.group(1).strip()
        else:
            fence_match = re.search(r"```(.*?)```", v, flags=re.S)
            if fence_match:
                v = fence_match.group(1).strip()
        try:
            ast.parse(v)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in model code: {e}")
        return v

    analysis: Optional[str] = Field(
        default=None, description="LLM's reasoning scratchpad"
    )

    @model_validator(mode="after")
    def params_match_code(self):
        param_match = re.search(
            r"(?:^|\n)\s*([\w\s,]+?)\s*=\s*model_parameters",
            self.code,
        )
        if param_match:
            code_params = {p.strip() for p in param_match.group(1).split(",")}
            declared_params = {p.name for p in self.parameters}
            if code_params != declared_params:
                raise ValueError(
                    f"Parameter mismatch: code unpacks {sorted(code_params)} "
                    f"but JSON declares {sorted(declared_params)}"
                )
        return self


class LLMResponseSchema(BaseModel):
    """Schema for the complete LLM response containing multiple models."""

    models: List[LLMModelResponse] = Field(
        ..., min_length=1, description="List of generated models"
    )

    @field_validator("models")
    @classmethod
    def unique_model_names(cls, v):
        names = [m.name for m in v]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate model names: {names}")
        return v


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
                "Complete Python function definition. Must start with the "
                "@njit decorator on its own line, followed by the function "
                "definition on the next line (i.e. '@njit\\ndef cognitive_modelN(...):...'). "
                "The function must be named cognitive_modelN (where N is its "
                "position in the list, starting from 1). Encode newlines as "
                "\\n within the JSON string."
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
      "code": "@njit\\ndef cognitive_model1(action_1, state, action_2, reward, model_parameters):\\n    ..."
    }}
  ]
}}

Field descriptions:
{analysis_example}  - `name`: Concise descriptive snake_case name (2-4 words) capturing the key mechanism (e.g. `dual_lr_perseveration`, `bayesian_transition_learner`, `attention_weighted_mbmf`). Must be unique across all models you propose.
  - `rationale`: One sentence explaining the model's hypothesis and why this architecture might improve on previous models.
  - `parameters`: List of model parameters with bounds. Each entry has `name` (matching the variable in code), `lower_bound`, and `upper_bound`. Must match the parameters unpacked from `model_parameters` in the same order.
  - `code`: Complete Python function definition. Must begin with `@njit` on its own line, then `def cognitive_model1(...)` on the next line. Functions must be named `cognitive_model1`, `cognitive_model2`, etc. Encode newlines as `\\n` within the JSON string.

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

        _log(
            "[GeCCo] Structured output parsing failed — falling back to regex extraction"
        )

    # Fallback: regex extraction
    return _fallback_regex_extraction(text, n_models), False


def _strip_thinking_tags(text: str) -> str:
    """Remove thinking/reasoning blocks produced by reasoning models.

    Handles:
    - <think>...</think> XML tags (DeepSeek, Qwen, etc.)
    - GLM-5.x "Mirrored thinking" inline prefix lines
      e.g. "json Mirrored thinkingHere is the output:"
    """
    # Remove <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove GLM-style "* Mirrored thinking*" prefix lines.
    # These appear as: "json Mirrored thinking<prose>\n\njson Mirrored thinking..."
    # Strip every segment that matches "* Mirrored thinking<text>" up to a JSON
    # boundary (``` or {), keeping only the part after the last such prefix.
    if "Mirrored thinking" in text:
        # Split on any "Mirrored thinking" occurrence and keep only what follows
        # the last one (the actual output).
        parts = re.split(r"(?:json\s+)?Mirrored thinking", text)
        text = parts[-1]
    return text.strip()


def _try_parse_dict(data) -> Optional[List[Dict]]:
    """Try to interpret a parsed JSON value as a model list."""
    if isinstance(data, dict):
        if "models" in data and isinstance(data["models"], list):
            return _validate_models(data["models"])
        # Single model object (e.g. Minimax outputs one dict directly)
        if "code" in data:
            return _validate_models([data])
    if isinstance(data, list):
        return _validate_models(data)
    return None


def _try_parse_json(text: str) -> Optional[List[Dict]]:
    """Attempt to parse JSON from the response text."""
    # Strip reasoning/thinking blocks first
    text = _strip_thinking_tags(text)

    # Try direct JSON parse
    try:
        data = json.loads(text.strip())
        result = _try_parse_dict(data)
        if result is not None:
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1).strip())
            result = _try_parse_dict(data)
            if result is not None:
                return result
        except json.JSONDecodeError:
            pass

    # Try finding a JSON object in the text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0))
            result = _try_parse_dict(data)
            if result is not None:
                return result
        except json.JSONDecodeError:
            pass

    return None


def _normalise_code(code: str) -> str:
    """Apply deterministic fixes for known LLM code generation quirks.

    All transforms here are safe and idempotent — they fix structural problems
    that the LLM should never have emitted rather than semantic errors.  Applied
    on EVERY code path (JSON parse, regex fallback, fix responses) so the
    validation / compilation stages never see these known-bad patterns.
    """
    # Fix double-escaped sequences (minimax and similar over-escape JSON strings).
    n_escaped = code.count("\\n")
    n_real = code.count("\n")
    if n_escaped > n_real:
        code = code.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
    # Strip markdown fences if the code field itself contains them.
    fence_match = re.search(r"```\s*(?:python)?(.*?)```", code, flags=re.S | re.I)
    if fence_match:
        code = fence_match.group(1).strip()
    # Fix @njit on same line as def: "@njit def func():" → "@njit\ndef func():"
    code = re.sub(r"@((?:numba\.)?njit)\s+def\s+", r"@\1\ndef ", code)
    # Add @njit if missing entirely (e.g. stripped during review/fix cycles).
    if "@njit" not in code and "@numba.njit" not in code:
        code = re.sub(r"(def\s+cognitive_model\w*\s*\()", r"@njit\n\1", code, count=1)
    # LLMs sometimes write the Python builtin max(arr, axis=) instead of np.max(arr, axis=).
    # np.max with axis= is fine in Numba 0.43+; the builtin never accepts axis=.
    code = re.sub(r"\bmax\s*\(([^)]*axis\s*=)", r"np.max(\1", code)
    code = re.sub(r"\bmin\s*\(([^)]*axis\s*=)", r"np.min(\1", code)
    # Fix doubled np prefix: np.np.max → np.max
    code = re.sub(r"\bnp\.np\.", "np.", code)
    # Strip trailing lone `}` that leaks from malformed JSON code fields.
    code = re.sub(r"\n\}\s*$", "", code)
    return code


def _validate_models(models: list) -> Optional[List[Dict]]:
    """Validate and normalize parsed model dicts using Pydantic.

    Returns:
        List of validated model dicts if validation succeeds, None otherwise.
        Returning None triggers the fallback regex extraction in parse_model_response().
    """
    processed = []
    for i, m in enumerate(models):
        if not isinstance(m, dict):
            return None
        # Normalise field name: some models return "corrected_code" instead of "code"
        if "code" not in m and "corrected_code" in m:
            m = {**m, "code": m["corrected_code"]}
        code = m.get("code", "")
        if not code:
            return None
        code = _normalise_code(code)
        # Infer minimal parameters from code when missing (common in fix responses).
        # The fix flow only uses `code` — original parameters are preserved upstream.
        if "parameters" not in m or not m["parameters"]:
            param_match = re.search(
                r"(?:^|\n)\s*([\w\s,]+?)\s*=\s*model_parameters", code
            )
            if param_match:
                names = [p.strip() for p in param_match.group(1).split(",")]
                m = {
                    **m,
                    "parameters": [
                        {"name": n, "lower_bound": 0, "upper_bound": 1}
                        for n in names
                    ],
                }
        processed.append({**m, "code": code})

    try:
        validated = LLMResponseSchema(models=processed)
        return [
            {
                "name": m.name,
                "rationale": m.rationale,
                "parameters": [
                    {
                        "name": p.name,
                        "lower_bound": p.lower_bound,
                        "upper_bound": p.upper_bound,
                    }
                    for p in m.parameters
                ],
                "code": m.code,
                "analysis": m.analysis,
            }
            for m in validated.models
        ]
    except ValidationError as e:
        print(f"[⚠️ GeCCo] Pydantic validation failed: {e}")
        # JSON parsed successfully — the data exists in `processed`. Return it
        # as-is so the caller can use the structured fields (name, rationale,
        # parameters, code) rather than falling back to regex extraction on the
        # raw JSON text, which is strictly worse (picks up JSON artifacts).
        # The validation-retry loop downstream will catch and fix code errors.
        return [
            {
                "name": m.get("name", f"cognitive_model{i + 1}"),
                "rationale": m.get("rationale", ""),
                "parameters": m.get("parameters", []),
                "code": m.get("code", ""),
                "analysis": m.get("analysis", None),
            }
            for i, m in enumerate(processed)
            if m.get("code")
        ] or None


@dataclass
class ValidationResult:
    """Result of validating a single model."""

    is_valid: bool
    errors: List[str]
    spec: Any  # ModelSpec if valid, None otherwise


def validate_single_model(model: Dict[str, Any]) -> "ValidationResult":
    """
    Validate a single parsed model dict against Pydantic schema.

    Parameters
    ----------
    model : dict
        Parsed model dict with keys: name, rationale, parameters, code, analysis.

    Returns
    -------
    ValidationResult
        is_valid: True if model passes all validation checks.
        errors: List of error messages if validation failed.
        spec: ModelSpec object if valid, None otherwise.
    """
    from gecco.offline_evaluation.utils import build_model_spec
    from gecco.offline_evaluation.exceptions import ModelValidationError

    errors = []

    if not model.get("code"):
        errors.append("Missing required field: code")
        return ValidationResult(is_valid=False, errors=errors, spec=None)

    try:
        func_name = model.get("name", "cognitive_model")
        spec = build_model_spec(
            code=model["code"],
            expected_func_name=func_name,
            structured_params=model.get("parameters"),
        )
        return ValidationResult(is_valid=True, errors=[], spec=spec)

    except ModelValidationError as e:
        errors.append(f"{e.error_type}: {e.message}")
        if "validation_errors" in e.details:
            for val_err in e.details["validation_errors"]:
                errors.append(f"  {val_err}")
        if "pydantic_errors" in e.details:
            for pyd_err in e.details["pydantic_errors"]:
                errors.append(f"  {pyd_err.get('msg', str(pyd_err))}")
        if "forbidden_patterns" in e.details:
            errors.append(f"  Forbidden patterns: {e.details['forbidden_patterns']}")

    except Exception as e:
        errors.append(f"Unexpected validation error: {str(e)}")

    return ValidationResult(is_valid=False, errors=errors, spec=None)


def build_correction_prompt(
    model: Dict[str, Any],
    model_index: int,
    validation_errors: List[str],
    schema_instructions: str,
) -> str:
    """
    Build a prompt asking the LLM to fix validation errors.

    Parameters
    ----------
    model : dict
        The model that failed validation.
    model_index : int
        Position of this model in the iteration (1-indexed).
    validation_errors : list of str
        Full error trace from validation.
    schema_instructions : str
        Schema instructions from get_schema_instructions() — same as used in main prompt.

    Returns
    -------
    str
        Prompt for the LLM to correct the model.
    """
    error_trace = "\n".join(f"  • {err}" for err in validation_errors)

    rationale = model.get("rationale", "N/A")
    code = model.get("code", "")
    params = model.get("parameters", [])

    params_str = ""
    if params:
        params_str = "Current parameters:\n"
        for p in params:
            params_str += (
                f"  - {p.get('name', '?')}: "
                f"[{p.get('lower_bound', 0)}, {p.get('upper_bound', 1)}]\n"
            )
    else:
        params_str = (
            "WARNING: The `parameters` field is missing from this model. "
            "You MUST extract the parameters from the code and include them "
            "in your response.\n"
        )

    return f"""The following model has validation errors and needs to be corrected.

### Model {model_index}: {model.get("name", "unnamed")}

{params_str}
Rationale: {rationale}

```python
{code}
```

### Validation Errors
{error_trace}

### Instructions
Please provide a corrected version of this model that:
1. Fixes ALL validation errors listed above
2. Maintains the same cognitive mechanism (don't change the core idea)
3. Uses the same function name and parameters

IMPORTANT: Your response MUST include ALL fields: `name`, `rationale`, `parameters`, and `code`.
The `parameters` array is REQUIRED — it must list every parameter unpacked from `model_parameters` in the code, with `name`, `lower_bound`, and `upper_bound` for each.

{schema_instructions}

Respond with the CORRECTED model in the exact JSON format described above, containing exactly 1 model.
"""


def _fallback_regex_extraction(text: str, n_models: int) -> List[Dict]:
    """Fall back to regex extraction with generic names."""
    models = []
    for i in range(1, n_models + 1):
        code = extract_model_code(text, i)
        if code:
            models.append(
                {
                    "name": f"cognitive_model{i}",
                    "rationale": "",
                    "code": _normalise_code(code),
                    "analysis": "",
                }
            )
    return models


# ============================================================
# Review-and-Fix for cognitive models
# ============================================================


def get_review_schema() -> dict:
    """
    JSON schema for structured review output.

    Returns
    -------
    dict
        JSON schema compatible with OpenAI/Gemini structured output.
    """
    return {
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "description": "Name of the model being reviewed",
                        },
                        "overall_assessment": {
                            "type": "string",
                            "enum": [
                                "passes",
                                "minor_issues",
                                "major_issues",
                                "critical_failure",
                            ],
                            "description": "Overall assessment of the model",
                        },
                        "issues": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "numerical_stability",
                                            "rl_correctness",
                                            "uninitialized_variable",
                                            "parameter_bounds_mismatch",
                                            "indexing_error",
                                            "missing_initialization",
                                            "incorrect_update",
                                            "log_likelihood_error",
                                        ],
                                        "description": "Type of issue found",
                                    },
                                    "location": {
                                        "type": "string",
                                        "description": "Location in code (e.g., 'line 45' or 'softmax computation')",
                                    },
                                    "severity": {
                                        "type": "string",
                                        "enum": ["critical", "warning", "info"],
                                        "description": "Severity level",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Description of the issue",
                                    },
                                    "suggested_fix": {
                                        "type": "string",
                                        "description": "Optional suggestion for how to fix",
                                    },
                                },
                                "required": ["type", "severity", "description"],
                                "additionalProperties": False,
                            },
                            "description": "List of issues found in the model",
                        },
                    },
                    "required": ["model_name", "overall_assessment", "issues"],
                    "additionalProperties": False,
                },
                "description": "List of model reviews",
            }
        },
        "required": ["reviews"],
        "additionalProperties": False,
    }


def build_review_prompt(
    models: List[Dict],
    guardrails: list = None,
    persona: str = None,
    focus_areas: list = None,
) -> str:
    """
    Build a structured code review prompt for cognitive models.

    Parameters
    ----------
    models : list of dict
        Parsed model dicts from the initial generation.
    guardrails : list of str, optional
        Model guardrails from config.
    persona : str, optional
        Custom reviewer persona. If None, uses default.
    focus_areas : list of str, optional
        Specific areas to focus review on. If None, uses default set.

    Returns
    -------
    str
        Prompt for the review LLM call. Returns structured JSON review, not code.
    """
    if persona is None:
        persona = (
            "You are an expert code reviewer for computational cognitive models. "
            "Your PhD is in computational neuroscience with 10+ years of experience "
            "debugging reinforcement learning models, likelihood functions, and "
            "hierarchical Bayesian models. You identify bugs BEFORE code execution."
        )

    if focus_areas is None:
        focus_areas = [
            "numerical_stability",
            "rl_correctness",
            "initialization",
            "parameter_bounds",
            "choice_indexing",
            "likelihood_correctness",
            "numba_compatibility",
        ]

    focus_descriptions = {
        "numerical_stability": """
- Softmax overflow: exp(beta * Q) can overflow for beta > 5 with Q > 2. Use exp(beta * (Q - max_Q)) or similar stabilization.
- Log of zero/negative: add epsilon (1e-10) INSIDE log(), not outside.
- Division by zero: probability denominators must not be zero.
- NaN propagation: initialize arrays to 0.0, never leave uninitialized.""",
        "rl_correctness": """
- Learning rate updates: Q += alpha * delta, not Q = alpha * delta (Q must retain prior value).
- TD error sign: delta = reward - Q[current] (not Q[current] - reward).
- Eligibility traces: lambda * alpha * delta_2 propagates to Q1, not just updates Q2.
- Model-based values: MB Q-values are transition_matrix @ max(Q_stage2), computed each trial.""",
        "initialization": """
- All Q-values must be initialized to zeros (or small values) BEFORE the trial loop.
- Stage-2 Q-values need shape (n_states, n_actions) or equivalent.
- Perseveration indicators, eligibility trace variables must be reset/initialized.""",
        "parameter_bounds": """
- Inverse temperature (beta, temperature): typically [0, 10] or [0, 30], NOT [0, 1].
- Learning rates (alpha): [0, 1].
- MB/MF mixing weights (w): [0, 1].
- Docstrings must accurately describe bounds used in optimization.""",
        "choice_indexing": """
- p_choice[t] = probs[action[t]] — index by the OBSERVED action from data, not argmax.
- State indexing: use state[trial] for current trial, not state[trial+1].
- Array shapes: Q[stage, action] indexing must match shape (n_stages, n_actions).""",
        "likelihood_correctness": """
- Return NEGATIVE log-likelihood (for minimization).
- Sum over ALL choice probabilities across both stages.
- Use log(p_choice + eps) inside the sum, not outside.""",
        "numba_compatibility": """
- All code runs inside @njit (Numba nopython mode). Only NumPy functions and basic Python are supported.
- Use np.max() / np.min() instead of Python built-in max() / min() for arrays — built-in max/min do NOT support axis= in Numba.
- Use np.abs(), np.exp(), np.log(), np.sum() — NOT abs(), math.exp(), math.log(), sum() on arrays.
- No Python lists, dicts, sets, or list comprehensions — use NumPy arrays only.
- No string operations, f-strings, or print() calls.
- No scipy, pandas, or other non-NumPy imports — only numpy is available inside @njit.
- np.zeros() shape must be a tuple of constants or integer variables, not Python lists.""",
    }

    focus_text = ""
    for area in focus_areas:
        if area in focus_descriptions:
            focus_text += f"\n### {area}\n{focus_descriptions[area]}\n"

    models_text = ""
    for i, m in enumerate(models):
        models_text += f"\n### Model {i + 1}: {m['name']}\n"
        if m.get("rationale"):
            models_text += f"Rationale: {m['rationale']}\n"
        if m.get("parameters"):
            params_str = ", ".join(
                f"{p['name']}: [{p.get('lower_bound', 0)}, {p.get('upper_bound', 1)}]"
                for p in m["parameters"]
            )
            models_text += f"Parameters: {params_str}\n"
        models_text += f"```python\n{m['code']}\n```\n"

    guardrails_text = ""
    if guardrails:
        guardrails_text = "\n### Constraints (from task config)\n" + "\n".join(
            f"- {g}" for g in guardrails
        )

    schema_str = json.dumps(get_review_schema(), indent=2)

    return f"""{persona}

## Your Task

Review each cognitive model for correctness BEFORE execution. Identify bugs, numerical issues, 
and modeling mistakes. Return your review as structured JSON.

## Focus Areas

{focus_text}
{guardrails_text}

## Models to Review

{models_text}

## Output Format

Return ONLY valid JSON matching this schema:

```json
{schema_str}
```

**Important:**
- For each model, set `overall_assessment` to one of: "passes", "minor_issues", "major_issues", "critical_failure"
- List every issue found with `type`, `severity`, `description`, and optionally `location` and `suggested_fix`
- If the model passes, set `issues` to an empty array `[]`
- Do NOT return corrected code — only return the review JSON"""


def build_fix_prompt(
    original_models: List[Dict], review: Dict, guardrails: list = None
) -> str:
    """
    Build a prompt asking the generating agent to fix models based on review.

    Parameters
    ----------
    original_models : list of dict
        Original model dicts with code.
    review : dict
        Structured review from parse_review_response().
    guardrails : list of str, optional
        Model guardrails from config.

    Returns
    -------
    str
        Prompt for the fix LLM call. Returns corrected models in JSON format.
        Returns empty string if no issues found.
    """
    # Map model names to issues
    issues_by_name = {}
    for r in review.get("reviews", []):
        name = r.get("model_name", "")
        issues = r.get("issues", [])
        assessment = r.get("overall_assessment", "passes")
        if issues or assessment != "passes":
            issues_by_name[name] = {"assessment": assessment, "issues": issues}

    if not issues_by_name:
        return ""

    # Build prompt with issues for each model
    models_text = ""
    for i, m in enumerate(original_models):
        name = m.get("name", f"cognitive_model{i + 1}")
        models_text += f"\n### Model {i + 1}: {name}\n"

        if name in issues_by_name:
            review_data = issues_by_name[name]
            models_text += f"Assessment: {review_data['assessment']}\n"
            models_text += "Issues found:\n"
            for issue in review_data["issues"]:
                sev = issue.get("severity", "warning")
                typ = issue.get("type", "unknown")
                desc = issue.get("description", "")
                loc = issue.get("location", "")
                fix = issue.get("suggested_fix", "")

                models_text += f"  - [{sev}] {typ}"
                if loc:
                    models_text += f" ({loc})"
                models_text += f": {desc}"
                if fix:
                    models_text += f" Suggested fix: {fix}"
                models_text += "\n"
        else:
            models_text += "Assessment: passes (no issues)\n"

        models_text += f"\nOriginal code:\n```python\n{m['code']}\n```\n"

    guardrails_text = ""
    if guardrails:
        guardrails_text = "\n### Constraints\n" + "\n".join(
            f"- {g}" for g in guardrails
        )

    return f"""The following models have issues that need to be fixed before fitting.

{models_text}
{guardrails_text}

## Your Task

Fix each model to address ALL identified issues. Return the corrected models in the same
JSON format as your original generation:

- Keep model names exactly the same
- Update the code to address each issue
- Update the rationale if the fix changes the model's approach
- Keep parameters the same unless bounds were incorrect
- IMPORTANT: Every function MUST use the @njit decorator for performance (e.g. @njit above def)

If a model has critical issues that cannot be fixed, note this in the rationale and 
propose an alternative architecture that addresses the core problems.

Return ONLY valid JSON with "models" array containing the corrected models."""


def parse_review_response(text: str) -> Dict:
    """
    Parse structured review JSON from LLM response.

    Parameters
    ----------
    text : str
        Raw LLM response text.

    Returns
    -------
    dict
        Parsed review dict with 'reviews' key containing list of model reviews.
        Returns empty reviews list if parsing fails.
    """
    text = _strip_thinking_tags(text)

    # Try direct JSON parse
    try:
        data = json.loads(text.strip())
        if "reviews" in data and isinstance(data["reviews"], list):
            return data
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    json_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1).strip())
            if "reviews" in data:
                return data
        except json.JSONDecodeError:
            pass

    # Try finding JSON object in text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0))
            if "reviews" in data:
                return data
        except json.JSONDecodeError:
            pass

    _log("[GeCCo] Review parsing failed — returning empty review")
    return {"reviews": []}


# ============================================================
# Legacy reflection prompt (kept for backward compatibility)
# ============================================================


def build_reflection_prompt(
    models: List[Dict], guardrails: list = None, use_json: bool = True
) -> str:
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
        guardrails_text = "\n### Guardrails to check against\n" + "\n".join(
            f"- {g}" for g in guardrails
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
            'Respond with the same JSON format as before — a JSON object with a "models" array.\n'
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
                k: _clean(v)
                for k, v in obj.items()
                if k not in ("additionalProperties", "minItems", "maxItems")
            }
        if isinstance(obj, list):
            return [_clean(item) for item in obj]
        return obj

    return _clean(schema)


def _strip_unsupported_strict_keywords(schema: dict) -> dict:
    """Strip JSON Schema keywords unsupported by OpenAI strict mode.

    ``strict: true`` rejects schemas containing ``minItems``, ``maxItems``,
    ``minimum``, ``maximum``, ``pattern``, and similar validation-only
    keywords.  Remove them so the schema is accepted.
    """
    _UNSUPPORTED = {"minItems", "maxItems", "minimum", "maximum", "pattern"}

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if k not in _UNSUPPORTED}
        if isinstance(obj, list):
            return [_clean(item) for item in obj]
        return obj

    return _clean(schema)


def get_openai_response_format(schema: dict) -> dict:
    """Build OpenAI Responses API text format spec."""
    return {
        "type": "json_schema",
        "name": "cognitive_models",
        "schema": _strip_unsupported_strict_keywords(schema),
        "strict": True,
    }


def get_chat_json_schema_format(schema: dict) -> dict:
    """
    Build a response_format dict for providers supporting json_schema in
    the Chat Completions API (OpenRouter, and OpenAI chat.completions).

    This is distinct from get_openai_response_format(), which targets the
    OpenAI Responses API (text.format, not response_format).

    Parameters
    ----------
    schema : dict
        JSON schema dict, e.g. from get_model_schema() or get_review_schema().

    Returns
    -------
    dict
        response_format value to pass to chat.completions.create().
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "cognitive_models",
            "schema": _strip_unsupported_strict_keywords(schema),
            "strict": True,
        },
    }


def get_vllm_response_format() -> dict:
    """Build vLLM-compatible response format."""
    return {"type": "json_object"}


def get_openai_compatible_response_format() -> dict:
    """
    JSON mode for any OpenAI-compatible API.

    Use this for providers that support JSON mode but not full JSON Schema:
    - vLLM (self-hosted)
    - KCL (internal)
    - OpenCode Zen
    - OpenRouter

    Ensures output is syntactically valid JSON. Schema enforcement is handled
    by the Pydantic validation layer (see pydantic_schema_enforcement.md).

    Returns
    -------
    dict
        OpenAI-compatible response format specification.
    """
    return {"type": "json_object"}
