import ast
import re
import types
from typing import Any, Dict, List, Optional
import numpy as np
import numba

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
    ValidationError,
)

# ============================================================
# Code validation schema
# ============================================================

FORBIDDEN_PATTERNS = {
    # Dangerous/System access
    "import os",
    "import sys",
    "import subprocess",
    "import eval",
    "import exec",
    "import compile",
    "__import__",
    "open(",
    "file(",
    "input(",
    "raw_input(",
    "__builtins__",
    "__class__",
    "__bases__",
    "globals()",
    "locals()",
    "vars(",
    "dir(",
    # Already-injected packages (LLMs should use these without importing)
    "import numba",
    "from numba",
    "import njit",
    "import numpy",
    "from numpy",
    "import np",
    "import scipy",
    "from scipy",
    "import json",
    "from json",
    "import math",
    "from math",
    "import itertools",
    "from itertools",
}


class CodeValidationSchema(BaseModel):
    """Validates that model code is safe, well-formed, and uses @njit."""

    code: str

    @field_validator("code")
    @classmethod
    def validate_safety(cls, v):
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in v:
                raise ValueError(f"Forbidden pattern in code: {pattern}")
        return v

    @field_validator("code")
    @classmethod
    def validate_syntax(cls, v):
        try:
            ast.parse(v)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}")
        return v

    @field_validator("code")
    @classmethod
    def validate_njit_decorator(cls, v):
        if "@njit" not in v and "@numba.njit" not in v:
            raise ValueError(
                "Model must use @njit decorator for performance. "
                "Add @njit above the function definition. "
                "Example: @njit\\ndef cognitive_model(...):"
            )
        return v

    @field_validator("code")
    @classmethod
    def validate_parameter_unpacking(cls, v):
        if "model_parameters" not in v:
            raise ValueError(
                "Code must unpack parameters from model_parameters. "
                "Add parameter unpacking like: alpha, beta = model_parameters"
            )
        return v


# ============================================================
# Pydantic model for unified representation of model specifications
# ============================================================


class ModelSpec(BaseModel):
    """Unified representation of a cognitive model specification."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., min_length=1)
    func: Any = Field(..., description="Compiled cognitive model function")
    param_names: List[str] = Field(..., min_length=1)
    bounds: Dict[str, List[float]] = Field(...)

    @field_validator("bounds")
    @classmethod
    def bounds_match_params(cls, v, info):
        param_names = info.data.get("param_names", [])
        missing = set(param_names) - set(v.keys())
        if missing:
            raise ValueError(f"Missing bounds for parameters: {sorted(missing)}")
        return v

    def model_post_init(self, __context):
        if not callable(self.func):
            raise ValueError(f"func must be callable, got {type(self.func)}")


# ============================================================
# Helper utilities
# ============================================================


def _extract_code_from_markdown(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    # Try ```python block
    code_match = re.search(r"```\s*python(.*?)```", text, flags=re.S | re.I)
    if code_match:
        return code_match.group(1).strip()

    # Try generic ``` block
    code_match = re.search(r"```(.*?)```", text, flags=re.S)
    if code_match:
        return code_match.group(1).strip()
    return text.strip()


def _extract_code_block(text: str) -> str:
    """Extract the Python code from an LLM output text."""
    code = _extract_code_from_markdown(text)

    # If no code block found, look for class/def definitions
    if "class " not in code and "def " not in code:
        func_match = re.search(r"((?:class|def)\s+\w+\s*\(.*)", text, flags=re.S)
        if func_match:
            return func_match.group(0).strip()

    return code


def is_class_based_code(code: str) -> bool:
    """Check if code uses the class-based structure."""
    return "CognitiveModelBase" in code or "make_cognitive_model" in code


def get_base_class_code(cfg) -> str:
    """
    Get the base class code from config.
    Extracts clean Python code from the config's abstract_base_model field.
    """
    if hasattr(cfg, "llm") and hasattr(cfg.llm, "abstract_base_model"):
        raw = cfg.llm.abstract_base_model
        return _extract_code_from_markdown(raw)

    raise ValueError("No abstract_base_model found in config.llm")


def _safe_exec_user_code(
    code: str, inject_base_class: bool = False, base_class_code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Safely execute user code and return namespace.

    Args:
        code: User code to execute
        inject_base_class: Whether to inject base class first
        base_class_code: The base class code (required if inject_base_class=True)
    """
    import json as _json, math as _math, scipy as _scipy, itertools as _itertools

    ns: Dict[str, Any] = {
        "np": np,
        "json": _json,
        "math": _math,
        "scipy": _scipy,
        "itertools": _itertools,
        "numba": numba,
        "njit": numba.njit,
        # JavaScript-style literals (common from non-Python-native LLMs)
        "true": True,
        "false": False,
        "null": None,
    }

    # Strip non-ASCII characters that some models inject (e.g. ✓, →)
    code = code.encode("ascii", errors="ignore").decode("ascii")

    try:
        if inject_base_class:
            if base_class_code is None:
                raise ValueError("base_class_code required when inject_base_class=True")
            exec(base_class_code, ns)
        exec(code, ns)
    except Exception as e:
        print(f"[⚠️ GeCCo] Error executing model code: {e}")
        raise
    return ns


def _find_first_function(ns: Dict[str, Any]) -> Optional[Any]:
    """Return the first callable function defined in the executed namespace."""
    for k, v in ns.items():
        if isinstance(v, types.FunctionType) and not k.startswith("_"):
            return v
    return None


def find_softmax_index_in_list(target_list: List[str]) -> List[int]:
    """Find indices of beta/softmax parameters in a list."""
    search_terms = {
        "beta",
        "beta1",
        "beta2",
        "beta_1",
        "beta_2",
        "softmax",
        "softmax_beta",
        "theta",
        "temperature",
        "inverse_temperature",
    }
    return [
        i for i, element in enumerate(target_list) if element.lower() in search_terms
    ]


# ============================================================
# Parameter & bounds extraction
# ============================================================


def extract_parameter_names(text: str) -> List[str]:
    """
    Extract parameter names from code.
    Handles both function-based and class-based patterns.
    """
    # Pattern 1: Class-based - look in unpack_parameters for self.x, self.y = model_parameters
    for line in text.splitlines():
        stripped = line.strip()
        if "model_parameters" in stripped and "self." in stripped and "=" in stripped:
            match = re.match(
                r"^(self\.\w+(?:\s*,\s*self\.\w+)*)\s*=\s*model_parameters", stripped
            )
            if match:
                lhs = match.group(1)
                params = re.findall(r"self\.(\w+)", lhs)
                if params:
                    return params

    # Pattern 2: Function-based - look for var1, var2 = model_parameters
    for line in text.splitlines():
        stripped = line.strip()
        if (
            "model_parameters" in stripped
            and "=" in stripped
            and "self." not in stripped
        ):
            match = re.match(r"^([\w\s,]+?)\s*=\s*model_parameters", stripped)
            if match:
                lhs = match.group(1)
                params = [p.strip() for p in lhs.split(",") if p.strip()]
                if len(params) >= 1:
                    return params

    # Pattern 3: Fallback - any multi-variable assignment
    for line in text.splitlines():
        # Strip leading/trailing whitespace to be robust to indentation
        stripped = line.strip()
        # Match lines like: a, b, c = something
        match = re.match(r"^([\w\s,]+?)\s*=\s*[A-Za-z_][A-Za-z0-9_]*$", stripped)
        if match:
            lhs = match.group(1)
            params = [p.strip() for p in lhs.split(",") if p.strip()]
            if len(params) > 1:
                return params
    return []


def extract_parameter_names_from_class(code: str, class_name: str) -> List[str]:
    """Extract parameter names from a specific class's unpack_parameters method."""
    class_pattern = rf"class\s+{class_name}\s*\([^)]*\):\s*(.*?)(?=\nclass\s+\w+\s*\(|cognitive_model\d*\s*=|$)"
    class_match = re.search(class_pattern, code, re.DOTALL)

    if not class_match:
        class_pattern = rf"class\s+{class_name}\s*\([^)]*\):(.*?)(?=\n\S)"
        class_match = re.search(class_pattern, code, re.DOTALL)

    if not class_match:
        return []

    class_code = class_match.group(0)
    return extract_parameter_names(class_code)


def parse_bounds_from_docstring(doc: Optional[str]) -> Dict[str, List[float]]:
    """Extract parameter bounds from docstrings."""
    if not doc:
        return {}

    bounds: Dict[str, List[float]] = {}

    # --- Case 1: explicit per-parameter bounds ---
    # Allows optional descriptive text in parens between param name and bounds,
    # e.g. "alpha (learning_rate): [0, 1]" or "alpha: [0, 1]"
    explicit_pattern = re.compile(
        r"""
        (?:^|\n)\s*[-*]?\s*
        ([A-Za-z_][A-Za-z0-9_]*)       # parameter name
        (?:\s*\([^)]*\))?               # optional description in parens, e.g. (learning_rate)
        [^()\[\]\n]*?                   # separator chars (colon, spaces, etc.)
        [\(\[\{]\s*
        ([\-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*
        [,\s]+\s*
        ([\-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*
        [\)\]\}]
        """,
        flags=re.I | re.X | re.M,
    )

    for name, lo, hi in explicit_pattern.findall(doc):
        try:
            bounds[name.lower()] = [float(lo), float(hi)]
            bounds[name] = [float(lo), float(hi)]
        except ValueError:
            continue

    return bounds


def _get_docstring_from_class(code: str, class_name: str) -> str:
    """Extract docstring from a class definition."""
    pattern = rf'class\s+{class_name}\s*\([^)]*\):\s*(?:"""(.*?)"""|\'\'\'(.*?)\'\'\')'
    match = re.search(pattern, code, re.DOTALL)
    if match:
        return match.group(1) or match.group(2) or ""
    return ""


# ============================================================
# High-level builder
# ============================================================


def build_model_spec(
    code: str,
    expected_func_name: str = "cognitive_model",
    cfg=None,
    base_class_code: Optional[str] = None,
    structured_params: Optional[List[Dict[str, Any]]] = None,
) -> ModelSpec:
    """
    Build ModelSpec from LLM-generated code.
    Automatically detects class-based vs function-based code.

    Args:
        code: Raw code string from LLM
        expected_func_name: Name of the function to extract
        cfg: Config object (used to get base class code if not provided)
        base_class_code: Base class code (optional, extracted from cfg if not provided)
        structured_params: Optional list of parameter dicts from structured JSON output,
            each with 'name', 'lower_bound', 'upper_bound'. Takes priority over
            docstring-parsed bounds when available.

    Returns:
        ModelSpec with compiled function, parameter names, and bounds
    """
    # --- Stage 1: Code safety validation ---
    try:
        CodeValidationSchema(code=code)
    except ValidationError as e:
        from gecco.offline_evaluation.exceptions import CodeSafetyError

        raise CodeSafetyError(
            "Code validation failed",
            details={"pydantic_errors": e.errors()},
        )

    # Enforce cognitive_model* naming for generated models (not baselines)
    is_baseline = not re.match(r"cognitive_model\d*$", expected_func_name)
    if not is_baseline and not re.search(r"def\s+cognitive_model\d*\s*\(", code):
        from gecco.offline_evaluation.exceptions import CodeSafetyError

        raise CodeSafetyError(
            "No cognitive_model function found. "
            "Define a function named cognitive_model (or cognitive_model1, etc.)",
            details={"invalid_func_name": expected_func_name},
        )

    # Extract code block if wrapped in markdown
    code = _extract_code_block(code)
    is_class = is_class_based_code(code)

    # Get base class code if needed
    if is_class and base_class_code is None:
        if cfg is not None:
            base_class_code = get_base_class_code(cfg)
        else:
            raise ValueError("cfg or base_class_code required for class-based models")

    # Execute code
    ns = _safe_exec_user_code(
        code, inject_base_class=is_class, base_class_code=base_class_code
    )

    # Get the function
    func = ns.get(expected_func_name)
    if func is None:
        func = _find_first_function(ns)
    if func is None:
        # Log the first few lines to help diagnose extraction issues
        preview = code[:200].replace("\n", "\\n") if code else "(empty)"
        raise ValueError(
            f"Function '{expected_func_name}' not found in code. "
            f"Code preview: {preview}"
        )

    # Extract parameters and docstring
    if is_class:
        class_match = re.search(
            rf"{expected_func_name}\s*=\s*make_cognitive_model\s*\(\s*(\w+)\s*\)", code
        )
        if class_match:
            class_name = class_match.group(1)
        else:
            num_match = re.search(r"(\d+)$", expected_func_name)
            class_name = (
                f"ParticipantModel{num_match.group(1)}"
                if num_match
                else "ParticipantModel"
            )

        param_names = extract_parameter_names_from_class(code, class_name)
        doc = _get_docstring_from_class(code, class_name)
    else:
        param_names = extract_parameter_names(code)
        doc = func.__doc__ or ""

    # Build bounds from structured JSON params (preferred) or docstring (fallback)
    structured_bounds: Dict[str, List[float]] = {}
    if structured_params:
        for sp in structured_params:
            name = sp.get("name", "")
            lo = sp.get("lower_bound")
            hi = sp.get("upper_bound")
            if name and lo is not None and hi is not None:
                structured_bounds[name] = [float(lo), float(hi)]
        # If structured params provide parameter names and code extraction
        # found none, use the structured param names as authoritative
        if structured_bounds and not param_names:
            param_names = [sp["name"] for sp in structured_params if sp.get("name")]

    docstring_bounds = parse_bounds_from_docstring(doc)

    # Fill in bounds: structured > docstring > defaults
    default_bound = [0, 1]
    beta_bound = [0, 10]

    final_bounds = {}
    for p in param_names:
        if p in structured_bounds:
            final_bounds[p] = structured_bounds[p]
        elif p in docstring_bounds:
            final_bounds[p] = docstring_bounds[p]
        elif p.lower() in docstring_bounds:
            final_bounds[p] = docstring_bounds[p.lower()]
        elif "beta" in p.lower() or "temperature" in p.lower():
            final_bounds[p] = beta_bound
        else:
            final_bounds[p] = default_bound

    if not param_names:
        print(f"[⚠️ GeCCo] No parameters found in {expected_func_name}")
    if not structured_bounds and not docstring_bounds:
        print(f"[⚠️ GeCCo] No bounds found in {expected_func_name}; using defaults")

    try:
        return ModelSpec(
            name=expected_func_name,
            func=func,
            param_names=param_names,
            bounds=final_bounds,
        )
    except ValidationError as e:
        from gecco.offline_evaluation.exceptions import PydanticSchemaError

        raise PydanticSchemaError(e)


# Aliases for backward compatibility
def build_model_spec_from_llm_output(
    text: str, expected_func_name: str = "cognitive_model", cfg=None
) -> ModelSpec:
    """Alias for build_model_spec (backward compatibility)."""
    return build_model_spec(text, expected_func_name, cfg=cfg)


def build_model_spec_from_class_code(
    code: str, expected_func_name: str = "cognitive_model1", cfg=None
) -> ModelSpec:
    """Alias for build_model_spec (backward compatibility)."""
    return build_model_spec(code, expected_func_name, cfg=cfg)
