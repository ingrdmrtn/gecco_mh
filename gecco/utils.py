import ast
import re
from datetime import datetime
from typing import Dict, List, Optional

import sentry_sdk


def log(msg: str, level: str = "info") -> None:
    """Print a timestamped log message and send to Sentry as a breadcrumb.

    Parameters
    ----------
    msg : str
        The message to log.
    level : str
        Log level: "debug", "info", "warning", "error". Defaults to "info".
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

    sentry_sdk.add_breadcrumb(
        message=msg,
        level=level,
        timestamp=datetime.now(),
    )


def extract_model_code(text: str, model_num: int) -> Optional[str]:
    """
    Unified extraction function that handles both class-based and function-based models.

    Args:
        text: Full LLM output
        model_num: Which model number (1, 2, 3, ...)

    Returns:
        Extracted code string, or None if not found
    """
    func_name = f"cognitive_model{model_num}"

    # First, try class-based extraction
    class_code = extract_class_with_wrapper(text, model_num)
    if class_code:
        return class_code

    # Fall back to function-based extraction
    func_code = extract_full_function(text, func_name)
    if func_code:
        return func_code

    return None


def extract_class_with_wrapper(code_text: str, model_num: int) -> str:
    """
    Extract ParticipantModelX class and its cognitive_modelX wrapper.

    Args:
        code_text: Full LLM output
        model_num: Which model number (1, 2, 3, ...)

    Returns:
        String containing the class definition + wrapper call
    """
    class_name = f"ParticipantModel{model_num}"
    wrapper_name = f"cognitive_model{model_num}"

    # Pattern to match the class definition
    # Matches from "class ParticipantModelX" to the next class or wrapper
    class_pattern = rf"(class\s+{class_name}\s*\(CognitiveModelBase\):.*?)(?=\nclass\s+|\n{wrapper_name}\s*=)"

    class_match = re.search(class_pattern, code_text, re.DOTALL)
    if not class_match:
        return None

    class_code = class_match.group(1).strip()

    # Pattern to match the wrapper assignment
    wrapper_pattern = (
        rf"({wrapper_name}\s*=\s*make_cognitive_model\s*\(\s*{class_name}\s*\))"
    )
    wrapper_match = re.search(wrapper_pattern, code_text)

    if not wrapper_match:
        return None

    wrapper_code = wrapper_match.group(1).strip()

    return f"{class_code}\n\n{wrapper_name} = make_cognitive_model({class_name})"


def extract_all_models(code_text: str, max_models: int = 10) -> dict:
    """
    Extract all ParticipantModel classes and their wrappers.

    Returns:
        Dict mapping function name to full code (class + wrapper)
    """
    models = {}
    for i in range(1, max_models + 1):
        code = extract_class_with_wrapper(code_text, i)
        if code:
            models[f"cognitive_model{i}"] = code
        else:
            break  # Stop when we don't find the next model
    return models


def extract_full_function(text: str, func_name: str) -> str:
    """
    Extract a full function definition for a given function name from the LLM output.
    Example:
        extract_full_function(llm_output, "cognitive_model1")
    will return:
        def cognitive_model1(...):
            ...
    """
    # extract all revelant blcoks
    blocks = re.findall(r"```(?:python)?(.*?)```", text, re.S)
    if blocks:
        text = "\n\n".join(block.strip() for block in blocks if block.strip())

    # Match the specific function by name, including any decorators above `def`
    pattern = rf"((?:@\w[\w.]*(?:\([^)]*\))?\s*\n\s*)*def\s+{func_name}\s*\([^)]*\)\s*:[\s\S]+?)(?=\n\s*(?:@\w[\w.]*\s*\n\s*)*def\s|\Z)"
    match = re.search(pattern, text, re.M)
    if match:
        func_block = match.group(1).strip()
    else:
        # Fallback: try to find any def block (with decorators) as a last resort
        match = re.search(
            r"((?:@\w[\w.]*(?:\([^)]*\))?\s*\n\s*)*def\s+\w+\s*\([^)]*\)\s*:[\s\S]+?)(?=\n\s*(?:@\w[\w.]*\s*\n\s*)*def\s|\Z)",
            text,
            re.M,
        )
        func_block = match.group(1).strip() if match else text.strip()

    # Clean up markdown or stray comments
    func_block = re.sub(r"^(\s*#+.*$)", "", func_block, flags=re.M)
    # Remove invalid line continuations (backslash followed by whitespace before newline)
    func_block = re.sub(r"\\\s+\n", "\n", func_block)
    # Convert literal \n and \t from JSON-encoded code to real newlines/tabs
    # (happens when regex fallback extracts code from inside a JSON string)
    # Use count comparison: JSON-encoded code has many \\n but few real \n
    n_escaped = func_block.count("\\n")
    n_real = func_block.count("\n")
    if n_escaped > n_real:
        func_block = (
            func_block.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
        )
    # Strip trailing JSON artifacts (e.g. closing `"`, `}`, `]` from the surrounding
    # JSON object) that the regex may have captured past the end of the function body.
    # This applies both after JSON unescaping and when the raw JSON uses real newlines.
    try:
        ast.parse(func_block)
    except SyntaxError:
        lines = func_block.split("\n")
        for end in range(len(lines), 0, -1):
            candidate = "\n".join(lines[:end]).strip()
            try:
                ast.parse(candidate)
                func_block = candidate
                break
            except SyntaxError:
                continue
    return func_block.strip()
