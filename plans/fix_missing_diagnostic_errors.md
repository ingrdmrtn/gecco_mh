# Fix Missing Diagnostic Data Errors for Judge Tools

## Overview

**Status:** Implementation Ready  
**Priority:** Medium  
**Impact:** Improves judge feedback quality and debugging experience  
**Estimated Effort:** 45-60 minutes

## Background and Problem Statement

### The Issue

The GeCCo judge uses diagnostic tools (`get_recovery`, `get_ppc`, `get_individual_differences`, `get_block_residuals`) to evaluate model quality during the model search process. However, these tools **silently return empty or null results** when the underlying diagnostic data doesn't exist in the database:

```python
# Current behavior (problematic)
▸ get_recovery(model_id=4)
  └─ None (4 chars)
▸ get_ppc(model_id=4)
  └─ 0 rows (2 chars)
```

This leads to confusing judge output where the LLM reports:
- "Parameter identifiability: **Confidence: N/A**"
- "Predictive adequacy: **Confidence: N/A**"

### Root Cause

The diagnostic data is only generated when specific configuration options are enabled:

1. **Parameter Recovery** requires:
   ```yaml
   parameter_recovery:
     enabled: true
   ```

2. **PPC (Posterior Predictive Checks)** requires:
   ```yaml
   judge:
     ppc:
       enabled: true
   ```

3. **Individual Differences** requires:
   ```yaml
   data:
     covariates: [...]  # covariates must be provided
   ```

4. **Block Residuals** requires:
   ```yaml
   judge:
     block_residuals:
       enabled: true
   ```

When these features aren't enabled, the database tables remain empty, but the tools return empty results without any indication of *why* the data is missing.

### Why This Matters

1. **Silent Failures**: The judge has no way to distinguish between "data not computed yet" vs "data doesn't exist because feature is disabled"
2. **Wasted Tool Calls**: The LLM wastes tool calls on diagnostics that will never return data
3. **Poor User Experience**: Users see "N/A" confidence and don't know how to fix it
4. **Configuration Discovery**: New users don't know which config options enable which diagnostics

## Solution: Explicit Error Messages

Instead of returning empty/null results, the tools should **raise clear errors** explaining:
1. What data is missing
2. Which configuration option enables it
3. Where to find documentation

The errors are caught by `dispatch_tool()` (lines 979-991), which already wraps all tool calls in a `try/except` and returns `{"error": str(exc)}`. This means both the OpenAI and Gemini tool loops will automatically propagate the error message back to the LLM — no changes needed in the tool loop logic.

## Implementation Plan

### Phase 0: Add helper exceptions and model existence check

Add a custom exception class and a helper function to the top of `gecco/diagnostic_store/tools.py`.

**Why a custom exception?** Using `DiagnosticNotAvailableError` (subclass of `RuntimeError`) provides a clear semantic distinction that `dispatch_tool` or future callers can handle specially if needed — e.g., they could choose to suppress diagnostic-not-available errors while still propagating genuine runtime errors. It also makes the intent explicit in the code.

**Why check model existence?** If a user passes a `model_id` that doesn't exist at all in the `models` table, we should say "model not found" rather than "diagnostic not available" — these are different failure modes requiring different responses. This is especially important because the judge receives model IDs from `get_best_models` and could theoretically reference a model that was in one client's store but not another's (in orchestrated mode).

```python
class DiagnosticNotAvailableError(RuntimeError):
    """Raised when a diagnostic tool is called but the data
    hasn't been computed (typically because the config option
    is disabled)."""


def _check_model_exists(store: DiagnosticStore, model_id: int) -> None:
    """Raise ValueError if model_id does not exist in the models table."""
    row = store.fetchone(
        "SELECT 1 FROM models WHERE model_id = ?", [model_id]
    )
    if row is None:
        raise ValueError(
            f"model_id={model_id} does not exist in the database. "
            "Ensure the model ID is valid."
        )
```

### Phase 1: Modify `get_recovery()`

**Location:** Lines 254-265

**Current code:**
```python
def get_recovery(store: DiagnosticStore, model_id: int) -> dict | None:
    """Return parameter recovery diagnostics for a model.

    ``per_param_r`` is summarised to ``{n_params, mean_r, min_r, max_r,
    worst_params}``.  ``per_param_detail`` (verbose coefficient tables) is
    omitted entirely.
    """
    sql = """
        SELECT * FROM parameter_recovery WHERE model_id = ?
    """
    row = store.fetchone(sql, [model_id])
    return _hydrate_for_judge(row) if row else None
```

**New code:**
```python
def get_recovery(store: DiagnosticStore, model_id: int) -> dict:
    """Return parameter recovery diagnostics for a model.

    ``per_param_r`` is summarised to ``{n_params, mean_r, min_r, max_r,
    worst_params}``.  ``per_param_detail`` (verbose coefficient tables) is
    omitted entirely.

    Raises:
        ValueError: If model_id does not exist.
        DiagnosticNotAvailableError: If parameter recovery data has not
            been computed for this model.  Enable via
            ``parameter_recovery.enabled: true`` in config.
    """
    _check_model_exists(store, model_id)
    sql = """
        SELECT * FROM parameter_recovery WHERE model_id = ?
    """
    row = store.fetchone(sql, [model_id])
    if row is None:
        raise DiagnosticNotAvailableError(
            f"Parameter recovery data not found for model_id={model_id}. "
            "This diagnostic requires 'parameter_recovery.enabled: true' in the "
            "config. Add this section to enable parameter identifiability checks."
        )
    return _hydrate_for_judge(row)
```

**Key changes:**
- Return type changed from `dict | None` → `dict` (function never returns `None` now)
- Model existence check first, with distinct `ValueError` for unknown model IDs
- `DiagnosticNotAvailableError` raised when model exists but has no recovery row
- Error message includes exact config key and actionable guidance

### Phase 2: Modify `get_ppc()`

**Location:** Lines 283-357

**Why `get_ppc` needs an explicit existence check:** Unlike `get_recovery` and `get_individual_differences` (which use `fetchone` that returns `None` for missing rows), `get_ppc` runs aggregate `GROUP BY` queries that return an **empty list** `[]` when no data exists. This means there's no single `None` check point — we need an explicit existence query before the main query.

**Current behavior:** Returns `[]` when no data exists.

**New code (insert at top of function, after docstring):**
```python
def get_ppc(store: DiagnosticStore, model_id: int,
            statistic: str | None = None,
            condition: str | None = None,
            participant_detail: bool = False) -> list[dict]:
    """Return PPC statistics for a model.

    By default returns one aggregated row per (statistic, condition) pair:
    ``n_participants``, ``n_outside_95ci``, ``frac_outside_95ci``,
    ``mean_observed``, ``mean_simulated_mean``, ``mean_abs_zscore``.
    Rows are sorted by ``frac_outside_95ci`` descending so the worst-fitting
    statistics appear first.

    Set ``participant_detail=True`` to instead get raw per-participant rows
    for the *single worst* (statistic, condition) combination (cap 50 rows).
    ``statistic`` and ``condition`` filters are applied before selecting the
    worst group, so you can use them to pin a specific group.

    Raises:
        ValueError: If model_id does not exist.
        DiagnosticNotAvailableError: If PPC data has not been computed for
            this model.  Enable via ``judge.ppc.enabled: true`` in config.
    """
    _check_model_exists(store, model_id)
    # Existence check — needed because aggregate queries silently return []
    # when no rows exist, giving no signal that the diagnostic wasn't run.
    check_sql = "SELECT 1 FROM ppc WHERE model_id = ? LIMIT 1"
    if store.fetchone(check_sql, [model_id]) is None:
        raise DiagnosticNotAvailableError(
            f"PPC data not found for model_id={model_id}. "
            "This diagnostic requires 'judge.ppc.enabled: true' in the config. "
            "Add this section under 'judge:' to enable posterior predictive checks."
        )
    
    # ... rest of existing implementation unchanged ...
```

### Phase 3: Modify `get_individual_differences()`

**Location:** Lines 268-279

**New code:**
```python
def get_individual_differences(store: DiagnosticStore,
                                model_id: int) -> dict:
    """Return individual differences R² and predictor coefficients.

    ``per_param_r2`` is summarised to ``{n_params, mean_r2, min_r2, max_r2,
    best_params}``.  ``per_param_detail`` (verbose coefficient tables) is
    omitted entirely.

    Raises:
        ValueError: If model_id does not exist.
        DiagnosticNotAvailableError: If individual-differences data has not
            been computed.  Requires covariates to be configured in the
            data section.
    """
    _check_model_exists(store, model_id)
    sql = """
        SELECT * FROM individual_differences WHERE model_id = ?
    """
    row = store.fetchone(sql, [model_id])
    if row is None:
        raise DiagnosticNotAvailableError(
            f"Individual differences data not found for model_id={model_id}. "
            "This diagnostic requires covariates to be configured in the data "
            "section (e.g., 'data.covariates: [anxiety_score, age]'). "
            "Without covariates, individual differences analysis cannot be "
            "performed."
        )
    return _hydrate_for_judge(row)
```

**Key change:** Return type `dict | None` → `dict`.

### Phase 4: Modify `get_block_residuals()`

**Location:** Lines 361-389

**Important:** Unlike the other three functions, `get_block_residuals` currently **never returns `None` or `[]`** — it returns a dict with `n_participants=0` and `blocks=[]`. This is actually more confusing than returning `None`, because it looks like valid data with zero participants. We need to add an existence check and raise, similar to `get_ppc`.

**Current signature:** `-> dict` (correct, remains `dict`)

**New code:**
```python
def get_block_residuals(store: DiagnosticStore, model_id: int) -> dict:
    """Return aggregated block-level NLL residual summaries for a model.

    Useful for identifying phases of the task where the model's NLL per
    trial is especially high.

    Raises:
        ValueError: If model_id does not exist.
        DiagnosticNotAvailableError: If block residual data has not been
            computed.  Enable via ``judge.block_residuals.enabled: true``
            in config (auto-enabled when PPC is enabled).
    """
    _check_model_exists(store, model_id)
    # Existence check — needed because the current implementation returns
    # {n_participants: 0, blocks: []} for missing data, which could be
    # confused with a legitimate empty result.
    check_sql = "SELECT 1 FROM block_residuals WHERE model_id = ? LIMIT 1"
    if store.fetchone(check_sql, [model_id]) is None:
        raise DiagnosticNotAvailableError(
            f"Block residual data not found for model_id={model_id}. "
            "This diagnostic requires 'judge.block_residuals.enabled: true' "
            "in the config (or it is auto-enabled when 'judge.ppc.enabled: true'). "
            "Add this section under 'judge:' to enable block-level residual "
            "analysis."
        )
    
    # ... existing implementation unchanged ...
    participant_row = store.fetchone(
        "SELECT COUNT(DISTINCT participant_id) AS n_participants "
        "FROM block_residuals WHERE model_id = ?",
        [model_id],
    )
    # ... etc. (rest of existing code)
```

### Phase 5: Update `_format_tool_result()` in `gecco/construct_feedback/tool_judge.py`

**Location:** Lines 72-89

Currently, error dicts like `{"error": "PPC data not found..."}` display as `1 keys: error (123 chars)` which is not informative. Add a special case:

**Current code:**
```python
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
```

**New code:**
```python
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
```

This ensures that when a `DiagnosticNotAvailableError` is caught by `dispatch_tool` and returned as `{"error": "PPC data not found..."}`, the verbose log will show something like:
```
▸ get_ppc(model_id=4)
  └─ Error: PPC data not found for model_id=4. This diagnostic requires... (142 chars)
```
instead of:
```
▸ get_ppc(model_id=4)
  └─ 1 keys: error (142 chars)
```

### Phase 6: Update Tool Schemas (Required)

**Location:** Lines 721-737 (get_recovery), 743-760 (get_individual_differences), 765-805 (get_ppc), 810-826 (get_block_residuals)

This phase is **required**, not optional. The tool descriptions are what the LLM sees in its schemas. Without noting that they can return errors, the LLM has no way to know these tools might fail with configuration errors. This is critical for the judge to understand and report the error correctly.

Update each tool's `description` field to mention the error condition:

**`get_recovery`** (around line 722):
```python
"description": (
    "Return parameter recovery diagnostics for a model: overall "
    "pass/fail, mean Pearson r, and a compact per-parameter r "
    "summary (n_params, mean_r, min_r, max_r, worst 5 params by r). "
    "Verbose coefficient tables are omitted. "
    "WARNING: Returns an error if parameter recovery is not enabled "
    "in the config (parameter_recovery.enabled). Only call this tool "
    "when you know recovery diagnostics have been computed."
),
```

**`get_individual_differences`** (around line 744):
```python
"description": (
    "Return individual-differences regression results for a model: "
    "mean and max R², best parameter, and a compact per-parameter "
    "R² summary (n_params, mean_r2, min_r2, max_r2, best 5 params "
    "by R²).  Verbose coefficient tables are omitted. "
    "WARNING: Returns an error if covariates are not configured "
    "in the data section. Only call this tool when you know "
    "individual differences have been computed."
),
```

**`get_ppc`** (around line 766):
```python
"description": (
    "Return posterior predictive check (PPC) statistics for a model. "
    "By default returns one aggregated row per (statistic, condition) "
    "pair: n_participants, n_outside_95ci, frac_outside_95ci, "
    "mean_observed, mean_simulated_mean, mean_abs_zscore.  Rows are "
    "sorted by frac_outside_95ci descending so the worst-fitting "
    "statistics appear first.  "
    "Set participant_detail=true to get raw per-participant rows "
    "for the single worst (statistic, condition) group (capped at 50 "
    "rows); combine with statistic/condition filters to target a "
    "specific group. "
    "WARNING: Returns an error if PPC is not enabled in the config "
    "(judge.ppc.enabled). Only call this tool when you know PPC "
    "diagnostics have been computed."
),
```

**`get_block_residuals`** (around line 811):
```python
"description": (
    "Return block-level residual summaries for a model, aggregated "
    "across participants. Useful for identifying phases of the task "
    "where the model's NLL per trial is especially high. "
    "WARNING: Returns an error if block residuals are not enabled "
    "in the config (judge.block_residuals.enabled). Only call this "
    "tool when you know block residuals have been computed."
),
```

### Phase 7: Tests

Create tests in `tests/test_diagnostic_store_tools.py` (new file — no existing test file for this module).

```python
import pytest
from gecco.diagnostic_store import DiagnosticStore
from gecco.diagnostic_store.tools import (
    DiagnosticNotAvailableError,
    get_recovery,
    get_ppc,
    get_individual_differences,
    get_block_residuals,
    dispatch_tool,
)


@pytest.fixture
def empty_store(tmp_path):
    """A DiagnosticStore with schema but no diagnostic data."""
    db_path = tmp_path / "test.duckdb"
    return DiagnosticStore(db_path)


class TestDiagnosticNotAvailableErrors:
    """All four diagnostic tools should raise DiagnosticNotAvailableError
    when the relevant table is empty, and ValueError for unknown model IDs."""

    def test_get_recovery_no_data(self, empty_store):
        with pytest.raises(DiagnosticNotAvailableError, match="parameter_recovery.enabled"):
            get_recovery(empty_store, model_id=999)

    def test_get_ppc_no_data(self, empty_store):
        with pytest.raises(DiagnosticNotAvailableError, match="judge.ppc.enabled"):
            get_ppc(empty_store, model_id=999)

    def test_get_individual_differences_no_data(self, empty_store):
        with pytest.raises(DiagnosticNotAvailableError, match="covariates"):
            get_individual_differences(empty_store, model_id=999)

    def test_get_block_residuals_no_data(self, empty_store):
        with pytest.raises(DiagnosticNotAvailableError, match="block_residuals.enabled"):
            get_block_residuals(empty_store, model_id=999)


class TestModelExistenceCheck:
    """Tools should raise ValueError for model IDs that don't exist
    in the models table, distinct from diagnostic-not-available errors."""

    def test_get_recovery_unknown_model(self, empty_store):
        with pytest.raises(ValueError, match="does not exist"):
            get_recovery(empty_store, model_id=999)

    def test_get_ppc_unknown_model(self, empty_store):
        with pytest.raises(ValueError, match="does not exist"):
            get_ppc(empty_store, model_id=999)


class TestDispatchToolIntegration:
    """dispatch_tool should catch DiagnosticNotAvailableError and
    return it as an error dict."""

    def test_dispatch_catches_diagnostic_error(self, empty_store):
        result = dispatch_tool(empty_store, "get_recovery", {"model_id": 999})
        assert "error" in result
        assert "parameter_recovery.enabled" in result["error"]

    def test_dispatch_catches_unknown_model_error(self, empty_store):
        result = dispatch_tool(empty_store, "get_recovery", {"model_id": 999})
        # ValueError for unknown model is caught too
        assert "error" in result
```

**Note:** The `TestModelExistenceCheck` tests above assume an empty store has no models table entries. Since `_check_model_exists` queries the `models` table, model ID 999 won't exist, so `ValueError("does not exist")` will be raised *before* `DiagnosticNotAvailableError`. If the store always creates a `models` table on init, we may need to insert a model first and then test the "model exists but no diagnostic data" case separately. The test implementation should verify this.

## Functions NOT Modified

### `compare_models()` — EXCLUDED from changes

The `compare_models` function (line 480) uses `LEFT JOIN parameter_recovery` and `LEFT JOIN individual_differences`, which returns `NULL` for recovery/ID columns when the data doesn't exist. This is **by design** — it's a comparison tool that should show missing fields as null rather than error. A side-by-side comparison should still display models that lack diagnostics; it just shows `NULL` for those columns. This is correct behavior and should not be changed to raise errors.

### `_try_shortcut_from_recovery_failure()` — Verified Safe

This method in `ToolUsingJudge` (line 843) handles recovery failures passed as an **argument** from the main loop (`recovery_failures` parameter), not from `get_recovery` tool calls. It constructs failure notes from the `recovery_failures` list of dicts, which are populated elsewhere. The proposed changes to `get_recovery` don't affect this code path.

## Error Propagation: Both LLM Backends

Both tool loop implementations correctly propagate errors:

1. **`_OpenAIToolLoop.run()`** (line 219): Calls `dispatch_tool()` → result stored as `{"role": "tool", "content": result_str}` → LLM sees the error message.

2. **`_GeminiToolLoop.run()`** (line 351): Calls `dispatch_tool()` → result stored as `{"function_response": {"name": ..., "response": {"result": result_str}}}` → LLM sees the error message.

No changes needed in either loop. The `dispatch_tool` exception handler already catches any `Exception` (including `ValueError` and `DiagnosticNotAvailableError`) and returns `{"error": str(exc)}`.

## Expected Outcome

After implementing these changes, the judge will see clear, actionable error messages:

```
▸ get_recovery(model_id=4)
  └─ Error: Parameter recovery data not found for model_id=4. This diagnostic
    requires 'parameter_recovery.enabled: true' in the config. Add this section
    to enable parameter identifiability checks. (164 chars)

▸ get_ppc(model_id=4)
  └─ Error: PPC data not found for model_id=4. This diagnostic requires
    'judge.ppc.enabled: true' in the config. Add this section under 'judge:'
    to enable posterior predictive checks. (152 chars)
```

The judge can then report meaningfully:
- "Parameter identifiability: data unavailable — requires `parameter_recovery.enabled: true` in config"
- "Predictive adequacy: data unavailable — requires `judge.ppc.enabled: true` in config"

And vs. an unknown model ID:
```
▸ get_recovery(model_id=999)
  └─ Error: model_id=999 does not exist in the database. Ensure the model ID
    is valid. (78 chars)
```

## Migration Guide for Users

### To Enable Parameter Recovery:

```yaml
parameter_recovery:
  enabled: true
  n_subjects: 50        # Number of synthetic subjects for recovery
  n_trials: 100         # Number of trials per subject
  threshold: 0.5        # Minimum acceptable Pearson r
  n_fitting_starts: 3   # Number of optimization restarts
  n_jobs: -1            # Parallel jobs (-1 = all cores)
```

### To Enable PPC:

```yaml
judge:
  mode: "tool_using"
  ppc:
    enabled: true
    n_sims: 100         # Number of simulations per participant
  block_residuals:
    enabled: true       # Optional: analyze per-block residuals
    n_blocks: 10
```

### To Enable Individual Differences:

```yaml
data:
  path: "data/my_data.csv"
  covariates:           # List of column names to use as covariates
    - anxiety_score
    - age
    - gender
```

## Files to Modify

1. **`gecco/diagnostic_store/tools.py`** — Main implementation:
   - Add `DiagnosticNotAvailableError` exception class
   - Add `_check_model_exists()` helper function
   - Modify 4 functions: `get_recovery`, `get_ppc`, `get_individual_differences`, `get_block_residuals`
   - Update 4 tool schema descriptions (in `TOOL_SCHEMAS` list)

2. **`gecco/construct_feedback/tool_judge.py`** — Improve error display:
   - Modify `_format_tool_result()` to special-case `{"error": ...}` dicts

3. **`tests/test_diagnostic_store_tools.py`** — New test file:
   - `TestDiagnosticNotAvailableErrors` — all 4 tools raise correctly
   - `TestModelExistenceCheck` — unknown model IDs raise `ValueError`
   - `TestDispatchToolIntegration` — `dispatch_tool` catches and formats errors

## Verification Checklist

- [ ] `DiagnosticNotAvailableError` exception class added
- [ ] `_check_model_exists()` helper added and used by all 4 functions
- [ ] `get_recovery()` raises `DiagnosticNotAvailableError` when data missing
- [ ] `get_ppc()` raises `DiagnosticNotAvailableError` when data missing (with explicit existence check)
- [ ] `get_individual_differences()` raises `DiagnosticNotAvailableError` when data missing
- [ ] `get_block_residuals()` raises `DiagnosticNotAvailableError` when data missing (with explicit existence check)
- [ ] All 4 functions raise `ValueError` for unknown model IDs (distinct from "data not available")
- [ ] Return type annotations updated: `dict | None` → `dict` for affected functions
- [ ] `_format_tool_result()` special-cases `{"error": ...}` dicts
- [ ] Tool schema descriptions updated with WARNING about config requirements
- [ ] `compare_models()` left unchanged (LEFT JOIN handles missing data gracefully)
- [ ] `_try_shortcut_from_recovery_failure()` verified unaffected
- [ ] All new tests pass
- [ ] Existing tests still pass
- [ ] Judge output shows actionable error messages instead of "N/A"

## Related Issues

- Judge showing "Confidence: N/A" for diagnostics
- Wasted tool calls when diagnostics not enabled
- Confusion about which config options enable which features
- No distinction between "model doesn't exist" vs "diagnostic not computed"

## References

- `gecco/diagnostic_store/tools.py` — Tool functions and `TOOL_SCHEMAS` (main changes)
- `gecco/diagnostic_store/schema.py` — Database table definitions
- `gecco/diagnostic_store/store.py` — Store implementation, PPC/recovery data ingestion
- `gecco/construct_feedback/tool_judge.py` — `_format_tool_result()`, `_OpenAIToolLoop`, `_GeminiToolLoop`
- `gecco/run_gecco.py` — Configuration loading and diagnostic initialization (lines 106-184)
- `gecco/parameter_recovery.py` — Parameter recovery implementation
- `gecco/offline_evaluation/ppc.py` — PPC computation implementation