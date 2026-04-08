# Implementation Plan: Add `@njit` Support for Generated Cognitive Models

**Objective:** Update the GeCCo system to support LLM-generated cognitive models that use Numba's `@njit` decorator for performance optimization.

**Background:** The fitting process involves thousands of model evaluations per participant. Using `@njit` can significantly speed up these evaluations. The implementation approach is "Option A" - decorate the main `cognitive_model` function with `@njit`. This has been tested and works well with scipy's minimize function.

**Relationship to Other Plans:** This plan is a **prerequisite** for `plans/pydantic_schema_enforcement.md`. The Pydantic validation plan requires `@njit` to be available in the execution namespace before it can enforce `@njit` decorator usage. **Implement this plan FIRST**, then implement the Pydantic plan.

---

## Files to Modify

### 1. `requirements.txt`
**Purpose:** Add numba as a project dependency.

**Change:** Add `numba>=0.59` to the list of requirements.

**Location:** Anywhere in the list (preferably grouped with other scientific computing libraries like numpy, scipy).

**Example insertion:**
```text
numpy>=1.26
pandas>=2.2
numba>=0.59  # <-- ADD THIS LINE
bottleneck>=1.4
```

---

### 2. `gecco/offline_evaluation/utils.py`
**Purpose:** Make numba's `njit` decorator available in the namespace where LLM-generated code is executed.

**Changes needed:**

#### Change 2a: Add import statement
**Location:** At the top of the file, after existing imports (around line 5)

**Add:**
```python
import numba
```

#### Change 2b: Add numba to execution namespace
**Location:** In `_safe_exec_user_code()` function, in the namespace dictionary (around line 81-91)

**Current code:**
```python
    ns: Dict[str, Any] = {
        "np": np,
        "json": _json,
        "math": _math,
        "scipy": _scipy,
        "itertools": _itertools,
        # JavaScript-style literals (common from non-Python-native LLMs)
        "true": True,
        "false": False,
        "null": None,
    }
```

**Modified code:**
```python
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
```

**Note:** The `"njit"` entry allows LLMs to use `@njit` directly, or they can use `@numba.njit` if they prefer.

---

### 3. `config/two_step_factors_distributed.yaml`
**Purpose:** Update the prompts and examples to request and demonstrate `@njit` usage.

**Changes needed:**

#### Change 3a: Update guardrails section
**Location:** Under `llm.guardrails` (around line 72-80)

**Current:**
```yaml
  guardrails:
    - "Each model must be a standalone Python function."
    - "Function names must be `cognitive_model1`, `cognitive_model2`, etc."
    - "Take as input: `action_1, state, action_2, reward, model_parameters`."
    - "Return the **negative log-likelihood** of observed choices."
    - "Use all parameters meaningfully (no unused params)."
    - "Include a clear docstring for the model and each parameter."
    - "Parameter bounds: [0,1] for most; [0,10] for softmax `beta` (inverse temperature). Please refer to the template for how these are defined."
    - "Do NOT include any package imports inside the code you write. Assume all packages are already imported."
```

**Modified:**
```yaml
  guardrails:
    - "Each model must be a standalone Python function."
    - "Function names must be `cognitive_model1`, `cognitive_model2`, etc."
    - "Take as input: `action_1, state, action_2, reward, model_parameters`."
    - "Return the **negative log-likelihood** of observed choices."
    - "Use all parameters meaningfully (no unused params)."
    - "Include a clear docstring for the model and each parameter."
    - "Parameter bounds: [0,1] for most; [0,10] for softmax `beta` (inverse temperature). Please refer to the template for how these are defined."
    - "Do NOT include any package imports inside the code you write. Assume all packages are already imported."
    - "Use the `@njit` decorator on the cognitive_model function for improved performance. Numba is already imported and available."
```

#### Change 3b: Update template_model section
**Location:** Under `llm.template_model` (around line 82-115)

**Current:**
```yaml
  template_model: |
    def cognitive_model(action_1, state, action_2, reward, model_parameters):
        """Example model illustrating format only (do not reuse logic).
        Bounds:
        learning_rate: [0,1]
        beta: [0,10]
        """
        learning_rate, beta = model_parameters
```

**Modified:**
```yaml
  template_model: |
    @njit
    def cognitive_model(action_1, state, action_2, reward, model_parameters):
        """Example model illustrating format only (do not reuse logic).
        Bounds:
        learning_rate: [0,1]
        beta: [0,10]
        """
        learning_rate, beta = model_parameters
```

**Important:** Only add `@njit` before the function definition. Do NOT change any other code in the template.

#### Change 3c: Update baseline.model section
**Location:** Under `baseline.model` (around line 136-186)

**Current:**
```yaml
baseline:
  model: |
    def hybrid_model(action_1, state, action_2, reward, model_parameters):
      """
      Hybrid Model-Based and Model-Free Learning with Perseveration and Eligibility Traces.
```

**Modified:**
```yaml
baseline:
  model: |
    @njit
    def hybrid_model(action_1, state, action_2, reward, model_parameters):
      """
      Hybrid Model-Based and Model-Free Learning with Perseveration and Eligibility Traces.
```

**Important:** Only add `@njit` before the function definition. Do NOT change any other code in the baseline model.

---

## Implementation Notes

### Why These Changes Work

1. **Dynamic Execution:** The GeCCo system uses `exec()` to run LLM-generated code in a controlled namespace. By adding `numba` and `njit` to this namespace, the generated code can use `@njit` without needing to import anything.

2. **Scipy Compatibility:** The `@njit` decorator works with scipy's `minimize` because:
   - Numba-compiled functions accept numpy arrays as input
   - The first call triggers compilation, subsequent calls use the compiled version
   - The function returns a scalar float (negative log-likelihood), which scipy expects

3. **No Code Changes to Fitting Logic:** The existing fitting code in `fit_generated_models.py` doesn't need changes because:
   - `build_model_spec()` returns a callable function
   - Whether that function is jitted or not is transparent to the calling code
   - Scipy's minimize just sees a function that takes parameters and returns a scalar

### Testing Recommendations

After implementing these changes:

1. **Verify numba installation:**
   ```bash
   python -c "import numba; print(numba.__version__)"
   ```

2. **Test a simple model generation:**
   - Run a quick test with 1 iteration to ensure models are generated successfully
   - Check that the generated code includes `@njit` decorator

3. **Verify performance improvement:**
   - Compare fitting time with and without `@njit` on the same model
   - Expect significant speedup for models with many trials/participants

4. **Check for compilation warnings:**
   - Numba may issue warnings if the code can't be fully compiled
   - These are usually harmless but worth monitoring

### Potential Issues to Watch For

1. **First-call overhead:** The first evaluation will be slower due to compilation. This is expected and amortized over many evaluations.

2. **Numba limitations:** Some Python features aren't supported by Numba (e.g., arbitrary Python objects, some numpy advanced indexing). The LLM should stick to basic numpy operations which work well with `@njit`.

3. **Type inference:** Numba infers types on first call. If scipy passes different array shapes/types, this could cause issues, but this is rare with consistent data structures.

---

## Summary of Changes

| File | Lines | Change Type |
|------|-------|-------------|
| `requirements.txt` | +1 line | Add dependency |
| `gecco/offline_evaluation/utils.py` | +2 lines (import) +2 lines (namespace) | Add numba support |
| `config/two_step_factors_distributed.yaml` | +1 line (guardrails) +1 line (template) +1 line (baseline) | Update prompts |

**Total:** ~7 lines of changes across 3 files.

---

## Rollback Plan

If issues arise:

1. **Remove numba dependency:** Delete line from `requirements.txt`
2. **Revert utils.py:** Remove import and namespace entries
3. **Revert config:** Remove the guardrail line and `@njit` decorators from template and baseline

The changes are minimal enough that rollback is straightforward.

---

## Combined Testing with Pydantic Plan

After implementing both this plan AND `plans/pydantic_schema_enforcement.md`, run the combined tests:

```bash
# Verify numba works
python -c "import numba; print(numba.__version__)"

# Verify namespace has njit
python -c "from gecco.offline_evaluation.utils import _safe_exec_user_code; ns = _safe_exec_user_code('x=1'); print('njit' in ns)"

# Verify validation blocks imports
python -c "from gecco.offline_evaluation.utils import CodeValidationSchema; CodeValidationSchema(code='import numba')"  # Should fail

# Verify @njit requirement works
python -c "from gecco.offline_evaluation.utils import CodeValidationSchema; CodeValidationSchema(code='@njit\\ndef f(): pass')"  # Should pass
```

See `plans/pydantic_schema_enforcement.md` for the full combined testing checklist.
