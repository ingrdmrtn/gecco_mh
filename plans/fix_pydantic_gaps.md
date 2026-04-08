# Fix: Pydantic Schema Enforcement — Implementation Gaps

## Context

Review of `plans/pydantic_schema_enforcement.md` implementation found two gaps. Gap 1 is a real bug; Gap 2 is optional.

---

## Gap 1: LLMFeedbackGenerator doesn't include validation feedback

### Problem

`FeedbackGenerator.get_feedback()` correctly calls `_build_validation_feedback()` as the first feedback part (line 776). But `LLMFeedbackGenerator` overrides `get_feedback()` entirely (line 862) and never calls `_build_validation_feedback()`.

When `cfg.llm.use_llm_feedback` is true (which activates `LLMFeedbackGenerator`), validation errors recorded as `VALIDATION_ERROR` in the history are silently dropped from the feedback sent to the LLM in the next iteration.

### File to modify

`gecco/construct_feedback/feedback.py`

### Fix

In `LLMFeedbackGenerator.get_feedback()` (around line 862), add validation feedback to the context parts that get sent to the judge LLM:

```python
def get_feedback(self, best_model, tried_param_sets, id_results=None):
    # ... existing code builds context_parts ...

    # ADD after all context_parts are built, before search_context join:
    validation_feedback = self._build_validation_feedback()
    if validation_feedback:
        context_parts.insert(0, f"## Validation Errors\n{validation_feedback}")

    search_context = "\n\n".join(context_parts)
    # ... rest of method unchanged ...
```

The `_build_validation_feedback()` method is inherited from the parent class so it's already available.

### Why insert at position 0

Validation errors are the most actionable feedback — the LLM judge should see them first so it can incorporate "fix these issues" into its guidance for the next iteration.

---

## Gap 2 (optional): ModelSpec.name missing snake_case pattern

### Problem

The plan specified:
```python
name: str = Field(..., min_length=1, pattern=r'^[a-z_][a-z0-9_]*$')
```

The implementation has:
```python
name: str = Field(..., min_length=1)
```

### File to modify

`gecco/offline_evaluation/utils.py` line 95

### Assessment

This is likely fine to leave as-is because:
- `ModelSpec` is constructed internally by `build_model_spec()`, not from raw LLM input
- The LLM-facing `LLMModelResponse.name` in `structured_output.py` already enforces the snake_case pattern
- The `expected_func_name` passed to `build_model_spec()` is controlled by our code

### Fix (if desired)

```python
name: str = Field(..., min_length=1, pattern=r'^[a-z_][a-z0-9_]*$')
```

---

## Verification

After applying fixes:
1. `pytest tests/test_hbi.py -v` — ensure nothing breaks
2. Manual test: run a GeCCo iteration with `use_llm_feedback: true` that triggers a validation error, and confirm the error appears in the LLM judge's feedback output
