# Pydantic Schema Enforcement for Cognitive Models

## Context

GeCCo (Generate, Evaluate, Critique, Correct) is a cognitive model discovery system that iteratively generates candidate cognitive models from an LLM, fits them to behavioral data, and uses feedback to guide the search toward better models.

Currently, candidate models are parsed from LLM JSON responses and compiled into a `ModelSpec` dataclass without strict schema validation. This leads to issues like:
- Parameter names in code don't match declared bounds
- Invalid or unsafe Python code being executed
- Poor error messages when validation fails
- No structured feedback mechanism for the LLM to correct its errors

Additionally, the system is being updated to support Numba's `@njit` decorator for performance optimization (see `plans/add_njit_support.md`). This requires:
- `numba` and `njit` being injected into the execution namespace
- LLMs being required to use `@njit` on their generated models

## Goal

Implement Pydantic-based schema validation at two key stages:
1. **JSON Parsing Stage** - Validate LLM response structure and parameter declarations
2. **Code Execution Stage** - Validate code safety, `@njit` usage, and correctness before execution

All validation errors must be structured and actionable so they can be fed back to the LLM agent for correction.

## Prerequisites

**IMPORTANT:** The Numba support plan (`plans/add_njit_support.md`) must be implemented BEFORE this plan because:

1. `numba` and `njit` must be added to the execution namespace before validation can check for `@njit`
2. The namespace injection changes are a prerequisite for the validation logic

**Implementation Order:**
1. Implement `plans/add_njit_support.md` first (namespace setup)
2. Then implement this plan (schema validation)

---

## Architecture Overview

### Current Flow
```
LLM → JSON/Code → parse_model_response() → build_model_spec()
    → ModelSpec (dataclass) → FIT_ERROR/RECOVERY_FAILED → feedback → next LLM
```

### Enhanced Flow with Pydantic
```
LLM → JSON → LLMModelResponse (Pydantic validation)
    → Code → CodeValidationSchema (safety checks)
    → build_model_spec() → ModelSpec (Pydantic)
    → ValidationError → ModelValidationError → feedback → next LLM
```

---

## Files to Create/Modify

### 1. New File: `gecco/offline_evaluation/exceptions.py`

**Purpose:** Custom exception hierarchy for structured validation errors.

```python
from pydantic import ValidationError
from typing import Optional, Dict, Any

class ModelValidationError(Exception):
    """Base exception for model validation failures with structured details."""
    
    def __init__(
        self,
        message: str,
        error_type: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.details = details or {}

class CodeSafetyError(ModelValidationError):
    """Raised when code contains forbidden patterns or invalid syntax."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CodeSafetyError", details)

class ParameterMismatchError(ModelValidationError):
    """Raised when declared parameters don't match code extraction."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "ParameterMismatchError", details)

class PydanticSchemaError(ModelValidationError):
    """Raised when Pydantic schema validation fails."""
    def __init__(self, pydantic_error: ValidationError):
        errors = pydantic_error.errors()
        messages = []
        for err in errors:
            loc = '.'.join(str(x) for x in err['loc'])
            messages.append(f"{loc}: {err['msg']}")
        
        super().__init__(
            message="Schema validation failed",
            error_type="PydanticSchemaError",
            details={"validation_errors": messages, "raw_errors": errors}
        )
```

**Key Design Decisions:**
- All exceptions inherit from `ModelValidationError` for uniform handling
- `error_type` string enables easy filtering in feedback generation
- `details` dict provides structured data for LLM feedback prompts
- `PydanticSchemaError` extracts actionable messages from Pydantic's error objects

---

### 2. Modify: `gecco/structured_output.py`

**Add new Pydantic schemas after existing imports (around line 25):**

```python
from pydantic import BaseModel, Field, field_validator, model_validator
import re
from typing import List, Optional

class ModelParameter(BaseModel):
    """Schema for a single model parameter with bounds."""
    name: str = Field(
        ...,
        min_length=1,
        pattern=r'^[a-z_][a-z0-9_]*$',
        description="Parameter name in snake_case"
    )
    lower_bound: float = Field(
        ...,
        ge=-100,
        le=1000,
        description="Lower bound for optimization"
    )
    upper_bound: float = Field(
        ...,
        ge=-100,
        le=1000,
        description="Upper bound for optimization"
    )
    
    @model_validator(mode='after')
    def bounds_order_valid(self):
        """Ensure upper_bound >= lower_bound."""
        if self.upper_bound < self.lower_bound:
            raise ValueError(
                f'upper_bound ({self.upper_bound}) must be >= lower_bound ({self.lower_bound})'
            )
        return self

class LLMModelResponse(BaseModel):
    """Schema for a single model's JSON response from the LLM."""
    name: str = Field(
        ...,
        min_length=1,
        max_length=50,
        pattern=r'^[a-z_][a-z0-9_]*$',
        description="Snake_case model name (2-4 words)"
    )
    rationale: str = Field(
        ...,
        min_length=10,
        description="One sentence explaining the model's hypothesis"
    )
    parameters: List[ModelParameter] = Field(
        ...,
        min_length=1,
        description="List of model parameters with bounds"
    )
    code: str = Field(
        ...,
        min_length=10,
        description="Complete Python function definition"
    )
    analysis: Optional[str] = Field(
        default=None,
        description="LLM's reasoning scratchpad"
    )
    
    @model_validator(mode='after')
    def params_match_code(self):
        """Verify declared parameters match code unpacking pattern."""
        param_match = re.search(
            r'(?:^|\n)\s*([\w\s,]+?)\s*=\s*model_parameters',
            self.code
        )
        if param_match:
            code_params = {p.strip() for p in param_match.group(1).split(',')}
            declared_params = {p.name for p in self.parameters}
            if code_params != declared_params:
                raise ValueError(
                    f'Parameter mismatch: code unpacks {sorted(code_params)} '
                    f'but JSON declares {sorted(declared_params)}'
                )
        return self

class LLMResponseSchema(BaseModel):
    """Schema for the complete LLM response containing multiple models."""
    models: List[LLMModelResponse] = Field(
        ...,
        min_length=1,
        description="List of generated models"
    )
    
    @field_validator('models')
    def unique_model_names(cls, v):
        """Ensure all model names are unique."""
        names = [m.name for m in v]
        if len(names) != len(set(names)):
            raise ValueError(f'Duplicate model names: {names}')
        return v
```

**Modify `_validate_models()` function (around line 273):**

```python
def _validate_models(models: list) -> Optional[List[Dict]]:
    """Validate and normalize parsed model dicts using Pydantic.
    
    Returns:
        List of validated model dicts if validation succeeds, None otherwise.
        Returning None triggers the fallback regex extraction in parse_model_response().
    """
    try:
        validated = LLMResponseSchema(models=models)
        return [
            {
                "name": m.name,
                "rationale": m.rationale,
                "parameters": [
                    {
                        "name": p.name,
                        "lower_bound": p.lower_bound,
                        "upper_bound": p.upper_bound
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
        return None
```

**Important:** The original `_validate_models()` should be modified in-place, preserving its return type signature (`Optional[List[Dict]]`). The function should still return `None` on validation failure so the existing fallback logic in `parse_model_response()` continues to work.

#### 2.3 Add Validation Helper Functions (NEW)

Add these functions after `_validate_models()` in `gecco/structured_output.py`:

```python
from dataclasses import dataclass
from typing import List, Any, Optional, Dict

@dataclass
class ValidationResult:
    """Result of validating a single model."""
    is_valid: bool
    errors: List[str]
    spec: Any  # ModelSpec if valid, None otherwise


def validate_single_model(model: Dict[str, Any]) -> ValidationResult:
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
    
    # Stage 1: Check required fields exist
    required_fields = ["code"]
    for field in required_fields:
        if field not in model or not model[field]:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return ValidationResult(is_valid=False, errors=errors, spec=None)
    
    # Stage 2: Try to build ModelSpec (triggers Pydantic validation + code safety checks)
    try:
        func_name = model.get("name", "cognitive_model")
        spec = build_model_spec(
            code=model["code"],
            expected_func_name=func_name,
            structured_params=model.get("parameters"),
        )
        return ValidationResult(is_valid=True, errors=[], spec=spec)
    
    except ModelValidationError as e:
        # Full error trace from Pydantic
        errors.append(f"{e.error_type}: {e.message}")
        
        # Add detailed validation errors
        if "validation_errors" in e.details:
            for val_err in e.details["validation_errors"]:
                errors.append(f"  {val_err}")
        
        # Add Pydantic errors if available
        if "pydantic_errors" in e.details:
            for pyd_err in e.details["pydantic_errors"]:
                errors.append(f"  {pyd_err.get('msg', str(pyd_err))}")
        
        # Add forbidden patterns if available
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
    
    Uses the same schema instructions as the original generation to maintain
    consistency with the main prompt.
    
    Parameters
    ----------
    model : dict
        The model that failed validation.
    model_index : int
        Position of this model in the iteration (1-indexed).
    validation_errors : list of str
        Full error trace from validation.
    schema_instructions : str
        Schema instructions from get_schema_instructions() - same as used in main prompt.
    
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
        params_str = "Parameters:\n"
        for p in params:
            params_str += f"  - {p.get('name', '?')}: [{p.get('lower_bound', 0)}, {p.get('upper_bound', 1)}]\n"
    
    return f"""The following model has validation errors and needs to be corrected.

### Model {model_index}: {model.get('name', 'unnamed')}

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

{schema_instructions}

Respond with the CORRECTED model in the exact JSON format described above, containing exactly 1 model.
"""
```

---

### 3. Modify: `gecco/offline_evaluation/utils.py`

#### 3.1 Add imports (after existing imports)

```python
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Any, Dict, List, Callable, Optional
import ast
```

#### 3.2 Add `CodeValidationSchema` (before `ModelSpec` class)

```python
FORBIDDEN_PATTERNS = {
    # Dangerous/System access
    'import os', 'import sys', 'import subprocess',
    'import eval', 'import exec', 'import compile',
    '__import__', 'open(', 'file(', 'input(', 'raw_input(',
    '__builtins__', '__class__', '__bases__',
    'globals()', 'locals()', 'vars(', 'dir(',
    
    # Already-injected packages (from _safe_exec_user_code namespace)
    # LLMs should use these without importing
    'import numba', 'from numba', 'import njit',
    'import numpy', 'from numpy', 'import np',
    'import scipy', 'from scipy',
    'import json', 'from json',
    'import math', 'from math',
    'import itertools', 'from itertools',
}

class CodeValidationSchema(BaseModel):
    """Validates that model code is safe, well-formed, and uses @njit."""
    code: str
    
    @field_validator('code')
    def validate_safety(cls, v):
        """Check for forbidden patterns."""
        v_normalized = v.replace(' ', ' ')
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in v_normalized:
                raise ValueError(f'Forbidden pattern in code: {pattern}')
        return v
    
    @field_validator('code')
    def validate_syntax(cls, v):
        """Verify code has valid Python syntax."""
        try:
            ast.parse(v)
        except SyntaxError as e:
            raise ValueError(f'Invalid Python syntax: {e}')
        return v
    
    @field_validator('code')
    def validate_njit_decorator(cls, v):
        """Ensure @njit decorator is present for performance."""
        if '@njit' not in v and '@numba.njit' not in v:
            raise ValueError(
                'Model must use @njit decorator for performance. '
                'Add @njit above the function definition. '
                'Example: @njit\\ndef cognitive_model(...):'
            )
        return v
    
    @field_validator('code')
    def validate_function_signature(cls, v):
        """Ensure code defines a cognitive_model function."""
        func_match = re.search(r'def\s+cognitive_model\d*\s*\(', v)
        if not func_match:
            raise ValueError(
                'No cognitive_model function found. '
                'Define a function named cognitive_model (or cognitive_model1, etc.)'
            )
        return v
    
    @field_validator('code')
    def validate_parameter_unpacking(cls, v):
        """Ensure code unpacks from model_parameters."""
        if 'model_parameters' not in v:
            raise ValueError(
                'Code must unpack parameters from model_parameters. '
                'Add parameter unpacking like: alpha, beta = model_parameters'
            )
        return v
```

**Note on decorator ordering:** The `@njit` decorator must appear directly above the function definition. Valid patterns:
```python
@njit
def cognitive_model(...):
    ...

# OR
@numba.njit
def cognitive_model(...):
    ...
```

Invalid (whitespace between decorator and function):
```python
@njit
def cognitive_model(...):  # This is OK
```

#### 3.3 Convert `ModelSpec` from dataclass to Pydantic

**Original (lines 12-17):**
```python
@dataclass
class ModelSpec:
    name: str
    func: Any
    param_names: List[str]
    bounds: Dict[str, List[float]]
```

**Replace with:**
```python
class ParameterBounds(BaseModel):
    """Stores bounds for a single parameter."""
    lower: float = Field(..., ge=-1000, le=1000)
    upper: float = Field(..., ge=-1000, le=1000)
    
    @model_validator(mode='after')
    def bounds_order(self):
        if self.upper < self.lower:
            raise ValueError(f'upper ({self.upper}) must be >= lower ({self.lower})')
        return self

class ModelSpec(BaseModel):
    """Unified representation of a cognitive model specification.
    
    Attributes:
        name: Snake_case model name
        func: Compiled callable cognitive model function
        param_names: Ordered list of parameter names
        bounds: Dict mapping parameter names to [lower, upper] bounds
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = Field(..., min_length=1, pattern=r'^[a-z_][a-z0-9_]*$')
    func: Any = Field(..., description="Compiled cognitive model function")
    param_names: List[str] = Field(..., min_length=1)
    bounds: Dict[str, List[float]] = Field(...)
    
    @field_validator('bounds')
    @classmethod
    def bounds_match_params(cls, v, info):
        """Ensure all param_names have bounds defined."""
        param_names = info.data.get('param_names', [])
        missing = set(param_names) - set(v.keys())
        if missing:
            raise ValueError(f'Missing bounds for parameters: {sorted(missing)}')
        return v
    
    def model_post_init(self, __context):
        """Validate function is callable after initialization."""
        if not callable(self.func):
            raise ValueError(f'func must be callable, got {type(self.func)}')
```

**Note on backward compatibility:** The `bounds` field remains `Dict[str, List[float]]` for backward compatibility with existing code. The internal `ParameterBounds` class is for future use if needed.

#### 3.4 Modify `build_model_spec()` function (around line 235)

Add validation at the start of the function:

```python
def build_model_spec(
    code: str,
    expected_func_name: str = "cognitive_model",
    cfg = None,
    base_class_code: Optional[str] = None,
    structured_params: Optional[List[Dict[str, Any]]] = None,
) -> ModelSpec:
    """
    Build ModelSpec from LLM-generated code with validation.
    
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
    
    Raises:
        CodeSafetyError: If code fails safety validation
        ValueError: If code extraction fails
    """
    # --- Stage 1: Code safety validation ---
    try:
        CodeValidationSchema(code=code)
    except ValidationError as e:
        from gecco.offline_evaluation.exceptions import CodeSafetyError
        raise CodeSafetyError(
            "Code validation failed",
            details={"pydantic_errors": e.errors()}
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
    ns = _safe_exec_user_code(code, inject_base_class=is_class, base_class_code=base_class_code)
    
    # Get the function
    func = ns.get(expected_func_name)
    if func is None:
        func = _find_first_function(ns)
    if func is None:
        preview = code[:200].replace("\n", "\\n") if code else "(empty)"
        raise ValueError(
            f"Function '{expected_func_name}' not found in code. "
            f"Code preview: {preview}"
        )
    
    # Extract parameters and docstring
    if is_class:
        class_match = re.search(
            rf'{expected_func_name}\s*=\s*make_cognitive_model\s*\(\s*(\w+)\s*\)',
            code
        )
        if class_match:
            class_name = class_match.group(1)
        else:
            num_match = re.search(r'(\d+)$', expected_func_name)
            class_name = f"ParticipantModel{num_match.group(1)}" if num_match else "ParticipantModel"
        
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
        elif 'beta' in p.lower() or 'temperature' in p.lower():
            final_bounds[p] = beta_bound
        else:
            final_bounds[p] = default_bound

    if not param_names:
        print(f"[⚠️ GeCCo] No parameters found in {expected_func_name}")
    if not structured_bounds and not docstring_bounds:
        print(f"[⚠️ GeCCo] No bounds found in {expected_func_name}; using defaults")
    
    # Build ModelSpec - Pydantic validation happens at construction
    try:
        return ModelSpec(
            name=expected_func_name,
            func=func,
            param_names=param_names,
            bounds=final_bounds
        )
    except ValidationError as e:
        from gecco.offline_evaluation.exceptions import PydanticSchemaError
        raise PydanticSchemaError(e)
```

---

### 4. Modify: `gecco/run_gecco.py`

#### 4.1 Update error handling in the model fitting loop (around line 507)

**Find this section:**
```python
try:
    # --- Parameter recovery check (optional) ---
    if self.recovery_checker is not None:
        self._set_activity(f"parameter recovery {i+1}/{n_models}: {display_name} (iter {it})")
        from gecco.offline_evaluation.utils import build_model_spec
        try:
            spec = build_model_spec(
                func_code, expected_func_name=func_name, cfg=self.cfg,
                structured_params=structured_params,
            )
            ...
        except Exception as e:
            console.print(
                f"  [yellow]{display_name} recovery check error: {e}[/]"
            )
            iteration_results.append({
                "function_name": display_name,
                "metric_name": "FIT_ERROR",
                "metric_value": float("inf"),
                "param_names": [],
                "code": func_code,
                "error": str(e),
            })
            continue
```

**Replace with:**
```python
try:
    # --- Parameter recovery check (optional) ---
    if self.recovery_checker is not None:
        self._set_activity(f"parameter recovery {i+1}/{n_models}: {display_name} (iter {it})")
        from gecco.offline_evaluation.utils import build_model_spec
        from gecco.offline_evaluation.exceptions import ModelValidationError
        try:
            spec = build_model_spec(
                func_code, expected_func_name=func_name, cfg=self.cfg,
                structured_params=structured_params,
            )
            ...
        except ModelValidationError as e:
            console.print(
                f"  [yellow]{display_name} validation error ({e.error_type}): {e.message}[/]"
            )
            iteration_results.append({
                "function_name": display_name,
                "metric_name": "VALIDATION_ERROR",
                "metric_value": float("inf"),
                "param_names": [],
                "code": func_code,
                "error_type": e.error_type,
                "error_message": e.message,
                "error_details": e.details,
            })
            continue
        except Exception as e:
            console.print(
                f"  [yellow]{display_name} recovery check error: {e}[/]"
            )
            iteration_results.append({
                "function_name": display_name,
                "metric_name": "RECOVERY_ERROR",
                "metric_value": float("inf"),
                "param_names": [],
                "code": func_code,
                "error": str(e),
            })
            continue
```

**Note:** `ModelValidationError` is the base class that catches `CodeSafetyError`, `ParameterMismatchError`, and `PydanticSchemaError`.

#### 4.2 Add Validation Correction Loop in `generate_models()` (around line 300)

The correction loop validates each model and retries generation for failures BEFORE reflection.

**Find the `generate_models()` method and modify the initial generation section:**

```python
def generate_models(self, prompt):
    """
    Generate cognitive models with structured output, parsing, validation
    correction loop, and optional self-critique reflection.

    Returns
    -------
    tuple of (raw_text, list of model dicts)
        Each model dict has keys: name, rationale, code, analysis.
    """
    from gecco.structured_output import (
        parse_model_response,
        build_reflection_prompt,
        validate_single_model,      # NEW
        build_correction_prompt,    # NEW
        get_schema_instructions,    # NEW
    )

    structured = getattr(self.cfg.llm, "structured_output", True)
    n_models = self.cfg.llm.models_per_iteration
    max_retries = getattr(self.cfg.validation, {}).get("retry_limit", 3)

    # --- Initial generation ---
    raw_text = self.generate(self.model, self.tokenizer, prompt)
    models, json_ok = parse_model_response(raw_text, n_models, structured_output=structured)

    if not models:
        console.print("[yellow]No models extracted from LLM response[/]")
        return raw_text, []

    # --- NEW: Validation correction loop ---
    # Validate each model with Pydantic and retry on failures.
    # This happens BEFORE reflection to avoid wasting compute on invalid models.
    validated_models = []
    for i, model in enumerate(models):
        validated_model = model
        model_name = model.get("name", f"cognitive_model{i + 1}")
        
        for retry_attempt in range(max_retries):
            validation_result = validate_single_model(validated_model)
            
            if validation_result.is_valid:
                console.print(f"[dim]Model {i + 1} ({model_name}) passed validation[/]")
                break
            
            # Build correction prompt with full error trace
            error_trace = "\n".join(f"  {err}" for err in validation_result.errors)
            console.print(
                f"[yellow]Model {i + 1} ({model_name}) failed validation (attempt {retry_attempt + 1}/{max_retries}):[/]\n"
                f"{error_trace}"
            )
            
            # Build correction prompt using same schema instructions as original generation
            schema_instructions = get_schema_instructions(1, include_analysis=False)
            correction_prompt = build_correction_prompt(
                model=validated_model,
                model_index=i + 1,
                validation_errors=validation_result.errors,
                schema_instructions=schema_instructions,
            )
            
            # Regenerate just this model
            correction_text = self.generate(self.model, self.tokenizer, correction_prompt)
            corrected, _ = parse_model_response(correction_text, 1, structured_output=structured)
            
            if corrected:
                validated_model = corrected[0]
            else:
                console.print("[yellow]Failed to parse correction attempt[/]")
                break
        
        if validation_result.is_valid:
            validated_models.append(validated_model)
        else:
            console.print(
                f"[bold red]Model {i + 1} ({model_name}) failed validation after {max_retries} retries — skipping[/]"
            )
            # Record validation error for feedback
            validated_models.append({
                "name": model_name,
                "rationale": model.get("rationale", ""),
                "code": model["code"],
                "analysis": model.get("analysis", ""),
                "validation_failed": True,
                "validation_errors": validation_result.errors,
            })

    models = validated_models
    if not models:
        console.print("[yellow]All models failed validation — no models to process[/]")
        return raw_text, []

    # --- Reflection step (optional) ---
    # ... existing reflection code ...
```

#### 4.3 Update the general fitting error handler (around line 620)

**Find:**
```python
except Exception as e:
    console.print(f"  [bold red]Error fitting {display_name}:[/] {e}")
    iteration_results.append({
        "function_name": display_name,
        "metric_name": "FIT_ERROR",
        "metric_value": float("inf"),
        "param_names": [],
        "code": func_code,
        "error": str(e),
    })
```

**Replace with:**
```python
except ModelValidationError as e:
    console.print(f"  [bold red]Validation error in {display_name}:[/] {e.message}")
    iteration_results.append({
        "function_name": display_name,
        "metric_name": "VALIDATION_ERROR",
        "metric_value": float("inf"),
        "param_names": [],
        "code": func_code,
        "error_type": e.error_type,
        "error_message": e.message,
        "error_details": e.details,
    })
except Exception as e:
    console.print(f"  [bold red]Error fitting {display_name}:[/] {e}")
    iteration_results.append({
        "function_name": display_name,
        "metric_name": "FIT_ERROR",
        "metric_value": float("inf"),
        "param_names": [],
        "code": func_code,
        "error": str(e),
    })
```

---

### 5. Modify: `gecco/construct_feedback/feedback.py`

#### 5.1 Add import at top of file

```python
from gecco.offline_evaluation.exceptions import ModelValidationError
```

#### 5.2 Add new method to `FeedbackGenerator` class (after `_build_landscape_summary()` around line 240)

```python
def _build_validation_feedback(self) -> str:
    """
    Generate actionable feedback from validation errors.
    
    This feedback is used by the LLM to correct code generation
    in the next iteration.
    """
    validation_errors = []
    for entry in self.history:
        for r in entry.get("results", []):
            if r.get("metric_name") == "VALIDATION_ERROR":
                validation_errors.append({
                    "name": r.get("function_name"),
                    "error_type": r.get("error_type"),
                    "message": r.get("error_message"),
                    "details": r.get("error_details", {}),
                    "iter": entry.get("iteration"),
                })
    
    if not validation_errors:
        return ""
    
    lines = [
        f"\nValidation errors ({len(validation_errors)} model(s) rejected):",
        "These issues must be fixed in your next iteration:\n"
    ]
    
    for err in validation_errors[:5]:  # Limit to 5 most recent
        lines.append(f"  • {err['name']}: {err['message']}")
        
        if err["error_type"] == "CodeSafetyError":
            lines.append(
                "    Forbidden code patterns detected. "
                "Packages are already injected - do NOT write import statements."
            )
            lines.append(
                "    Allowed: @njit, np, json, math, scipy, itertools, numba (use directly, no import)."
            )
            for detail in err["details"].get("pydantic_errors", []):
                lines.append(f"    - {detail}")
        
        elif err["error_type"] == "ParameterMismatchError":
            lines.append(
                "    Parameters in code don't match JSON declaration. "
                "Ensure model_parameters unpacking matches declared parameters."
            )
        
        elif err["error_type"] == "PydanticSchemaError":
            for val_err in err["details"].get("validation_errors", []):
                lines.append(f"    - {val_err}")
    
    lines.append(
        "\n  All models MUST satisfy ALL of the following:\n"
        "    1. Use @njit decorator directly above function definition (e.g., @njit\\ndef cognitive_model(...))\n"
        "    2. Do NOT write import statements - packages are already injected and available\n"
        "    3. Define a function named 'cognitive_model' (or cognitive_model1, etc.)\n"
        "    4. Unpack parameters from model_parameters tuple\n"
        "    5. Match declared parameter names to model_parameters unpacking\n"
        "    6. Provide valid numeric bounds for each parameter\n"
        "    7. Use snake_case for all parameter and function names\n"
    )
    
    return "\n".join(lines)
```

#### 5.3 Integrate validation feedback into `get_feedback()` method

**Find the `get_feedback()` method and add the validation feedback call:**

```python
def get_feedback(self) -> str:
    """Get formatted feedback string for the next iteration."""
    parts = []
    
    # Add validation feedback FIRST since it's most actionable
    validation_feedback = self._build_validation_feedback()
    if validation_feedback:
        parts.append(validation_feedback)
    
    # Add other feedback levels...
    trajectory = self._build_trajectory_summary()
    if trajectory:
        parts.append(trajectory)
    
    landscape = self._build_landscape_summary()
    if landscape:
        parts.append(landscape)
    
    # ... rest of method
```

---

### 6. Update Tests: `tests/test_hbi.py`

The `MockModelSpec` class (line 52) needs to work with the new Pydantic `ModelSpec`. Since `ModelSpec` now uses `arbitrary_types_allowed=True`, existing code that creates `ModelSpec` instances should work, but test mocks may need updating.

**Check if MockModelSpec is used for testing only or if it mocks the interface:**

If `MockModelSpec` is just a simple mock for testing fitting algorithms, update it to inherit from `ModelSpec` or at least have the same interface:

```python
# Option A: Inherit from ModelSpec for proper interface
# Option B: Keep as-is if it's only used for attribute access
# Check test usage first before modifying
```

**To verify, check how `MockModelSpec` is used in the test file:**
```bash
grep -n "MockModelSpec" tests/test_hbi.py
```

If it's used for attribute access only (`.param_names`, `.bounds`, `.func`), the existing mock should work since Pydantic models support arbitrary attributes when configured. If it fails tests after changes, update it then.

---

### 7. Add Config Section for Validation Retry Limit

Add a new`validation` section to config YAML files:

```yaml
# In config/*.yaml files
llm:
  # ... existing llm config ...
  structured_output: true
  reflection: true

# NEW: Validation settings
validation:
  retry_limit: 3# Max retries for Pydantic validation errors per model
```

**Default behavior:** If`validation` section is missing, defaults to `retry_limit: 3`.

The code reads this as:
```python
max_retries = getattr(self.cfg.validation, "retry_limit", 3)
# or
max_retries = getattr(self.cfg, "validation", {}).get("retry_limit", 3)
```

---

## Implementation Order

**PREREQUISITE:** Implement `plans/add_njit_support.md` FIRST. This plan adds `numba` and `njit` to the execution namespace, which is required for the validation checks below.

1. **`gecco/offline_evaluation/exceptions.py`** - Create custom exception classes
2. **`gecco/structured_output.py`** - Add Pydantic schemas, update `_validate_models()`, add `validate_single_model()` and `build_correction_prompt()`
3. **`gecco/offline_evaluation/utils.py`** - Add `CodeValidationSchema` (with `@njit` check), convert `ModelSpec`, update `build_model_spec()`
4. **`gecco/run_gecco.py`** - Add validation correction loop in `generate_models()`, update error handling to catch `ModelValidationError`
5. **`gecco/construct_feedback/feedback.py`** - Add `_build_validation_feedback()`, integrate into `get_feedback()`
6. **Config files** - Add `validation.retry_limit` to example configs
7. **Tests** - Run existing tests and fix any issues with `MockModelSpec`

---

## Backward Compatibility

- `ModelSpec` remains constructible with the same arguments: `(name, func, param_names, bounds)`
- `build_model_spec()` returns the same type, just with validation on construction
- `parse_model_response()` still returns `Optional[List[Dict]]` - returning `None` triggers existing fallback
- All existing exception types (`ValueError`, `TypeError`) still propagate from `build_model_spec()` for non-validation errors

---

## Error Type Summary

| Error Type | Trigger | LLM Feedback Message |
|------------|---------|----------------------|
| `CodeSafetyError` | Forbidden patterns (imports, system access), invalid syntax, missing `@njit` | "Use @njit decorator, don't import packages" |
| `ParameterMismatchError` | Code/JSON param mismatch | "Match model_parameters unpacking to declarations" |
| `PydanticSchemaError` | Invalid bounds, naming violations | Specific field errors from Pydantic |
| `FIT_ERROR` | Fitting algorithm failure | Generic fitting error |
| `RECOVERY_FAILED` | Parameter recovery check failure | Recovery-specific feedback |

---

## Testing Checklist

After implementing BOTH plans (Numba + Pydantic), verify:

### Namespace and Basic Import
- [ ] `python -c "import numba; print(numba.__version__)"` works
- [ ] `python -c "from gecco.offline_evaluation.utils import _safe_exec_user_code; ns = _safe_exec_user_code('x=1'); print('njit' in ns and 'numba' in ns)"` returns `True`

### Forbidden Pattern Validation
- [ ] `python -c "from gecco.offline_evaluation.utils import CodeValidationSchema; CodeValidationSchema(code='import numba\\n@njit\\ndef f(): pass')"` **FAILS** (blocked import)
- [ ] `python -c "from gecco.offline_evaluation.utils import CodeValidationSchema; CodeValidationSchema(code='import numpy\\n@njit\\ndef f(): pass')"` **FAILS** (blocked import)
- [ ] `python -c "from gecco.offline_evaluation.utils import CodeValidationSchema; CodeValidationSchema(code='import os')"` **FAILS** (system access)

### @njit Requirement
- [ ] `python -c "from gecco.offline_evaluation.utils import CodeValidationSchema; CodeValidationSchema(code='@njit\\ndef cognitive_model(): pass')"` passes
- [ ] `python -c "from gecco.offline_evaluation.utils import CodeValidationSchema; CodeValidationSchema(code='def cognitive_model(): pass')"` **FAILS** (missing @njit)
- [ ] `python -c "from gecco.offline_evaluation.utils import CodeValidationSchema; CodeValidationSchema(code='@numba.njit\\ndef cognitive_model(): pass')"` passes

### Full Integration
- [ ] `python -c "from gecco.offline_evaluation.exceptions import ModelValidationError; print('OK')"` works
- [ ] `python -c "from gecco.offline_evaluation.utils import ModelSpec, build_model_spec; print('OK')"` works
- [ ] Existing tests pass: `pytest tests/test_hbi.py -v`
- [ ] Validation errors are caught and recorded with `error_type`, `error_message`, `error_details`
- [ ] Feedback includes validation errors when present
- [ ] Models that fail validation are NOT fitted (early rejection)

---

## Common Pitfalls to Avoid

1. **Don't forget to import `ValidationError`** from pydantic in each module
2. **Preserve return type signatures** - `build_model_spec()` must still return `ModelSpec`, `_validate_models()` must still return `Optional[List[Dict]]`
3. **Keep fallback logic intact** - when `_validate_models()` returns `None`, `parse_model_response()` falls back to regex extraction
4. **Use the base class `ModelValidationError`** when catching errors - don't catch specific subclasses unless needed
5. **Test with actual LLM output** - synthetic test cases may not catch JSON parsing edge cases
6. **Implement Numba plan first** - The namespace must have `njit` available before validation checks for `@njit`. If you see `NameError: name 'njit' is not defined` during validation, the Numba namespace wasn't set up.
7. **`@njit` decorator must match exactly** - Use `@njit` or `@numba.njit`. Other forms like `@njit()` (with parentheses but no arguments) may work but are less reliable.
