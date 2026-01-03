import re
import types
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import numpy as np

# ============================================================
# Dataclass for unified representation of model specifications
# ============================================================

@dataclass
class ModelSpec:
    name: str
    func: Any
    param_names: List[str]
    bounds: Dict[str, List[float]]


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
    if hasattr(cfg, 'llm') and hasattr(cfg.llm, 'abstract_base_model'):
        raw = cfg.llm.abstract_base_model
        return _extract_code_from_markdown(raw)
    
    raise ValueError("No abstract_base_model found in config.llm")


def _safe_exec_user_code(
    code: str, 
    inject_base_class: bool = False,
    base_class_code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Safely execute user code and return namespace.
    
    Args:
        code: User code to execute
        inject_base_class: Whether to inject base class first
        base_class_code: The base class code (required if inject_base_class=True)
    """
    ns: Dict[str, Any] = {"np": np}
    
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
        'beta', 'beta1', 'beta2', 'beta_1', 'beta_2', 
        'softmax', 'softmax_beta', 'theta', 'temperature',
        'inverse_temperature'
    }
    return [i for i, element in enumerate(target_list) if element.lower() in search_terms]


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
        if 'model_parameters' in stripped and 'self.' in stripped and '=' in stripped:
            match = re.match(r'^(self\.\w+(?:\s*,\s*self\.\w+)*)\s*=\s*model_parameters', stripped)
            if match:
                lhs = match.group(1)
                params = re.findall(r'self\.(\w+)', lhs)
                if params:
                    return params
    
    # Pattern 2: Function-based - look for var1, var2 = model_parameters
    for line in text.splitlines():
        stripped = line.strip()
        if 'model_parameters' in stripped and '=' in stripped and 'self.' not in stripped:
            match = re.match(r'^([\w\s,]+?)\s*=\s*model_parameters', stripped)
            if match:
                lhs = match.group(1)
                params = [p.strip() for p in lhs.split(',') if p.strip()]
                if len(params) >= 1:
                    return params
    
    # Pattern 3: Fallback - any multi-variable assignment
    for line in text.splitlines():
        # Strip leading/trailing whitespace to be robust to indentation
        stripped = line.strip()
        # Match lines like: a, b, c = something
        match = re.match(r'^([\w\s,]+?)\s*=\s*[A-Za-z_][A-Za-z0-9_]*$', stripped)
        if match:
            lhs = match.group(1)
            params = [p.strip() for p in lhs.split(',') if p.strip()]
            if len(params) > 1:
                return params
    return []


def extract_parameter_names_from_class(code: str, class_name: str) -> List[str]:
    """Extract parameter names from a specific class's unpack_parameters method."""
    class_pattern = rf'class\s+{class_name}\s*\([^)]*\):\s*(.*?)(?=\nclass\s+\w+\s*\(|cognitive_model\d*\s*=|$)'
    class_match = re.search(class_pattern, code, re.DOTALL)
    
    if not class_match:
        class_pattern = rf'class\s+{class_name}\s*\([^)]*\):(.*?)(?=\n\S)'
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
    explicit_pattern = re.compile(
        r"""
        (?:^|\n)\s*[-*]?\s*
        ([A-Za-z_][A-Za-z0-9_]*)
        [^()\[\]\n]*?
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
    cfg = None,
    base_class_code: Optional[str] = None
) -> ModelSpec:
    """
    Build ModelSpec from LLM-generated code.
    Automatically detects class-based vs function-based code.
    
    Args:
        code: Raw code string from LLM
        expected_func_name: Name of the function to extract
        cfg: Config object (used to get base class code if not provided)
        base_class_code: Base class code (optional, extracted from cfg if not provided)
    
    Returns:
        ModelSpec with compiled function, parameter names, and bounds
    """
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
        raise ValueError(f"Function '{expected_func_name}' not found in code")
    
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
    
    # Parse bounds
    bounds = parse_bounds_from_docstring(doc)

    # Fill in missing bounds with defaults
    default_bound = [0, 1]
    beta_bound = [0, 10]
    
    final_bounds = {}
    for p in param_names:
        if p in bounds:
            final_bounds[p] = bounds[p]
        elif p.lower() in bounds:
            final_bounds[p] = bounds[p.lower()]
        elif 'beta' in p.lower() or 'temperature' in p.lower():
            final_bounds[p] = beta_bound
        else:
            final_bounds[p] = default_bound
    
    if not param_names:
        print(f"[⚠️ GeCCo] No parameters found in {expected_func_name}")
    if not bounds:
        print(f"[⚠️ GeCCo] No bounds found in {expected_func_name}; using defaults")
    
    return ModelSpec(
        name=expected_func_name,
        func=func,
        param_names=param_names,
        bounds=final_bounds
    )


# Aliases for backward compatibility
def build_model_spec_from_llm_output(
    text: str, 
    expected_func_name: str = "cognitive_model",
    cfg = None
) -> ModelSpec:
    """Alias for build_model_spec (backward compatibility)."""
    return build_model_spec(text, expected_func_name, cfg=cfg)


def build_model_spec_from_class_code(
    code: str, 
    expected_func_name: str = "cognitive_model1",
    cfg = None
) -> ModelSpec:
    """Alias for build_model_spec (backward compatibility)."""
    return build_model_spec(code, expected_func_name, cfg=cfg)