# Fix CMG Parameter-Recovery Simulation Failures

## Goal

Fix the CMG runtime behavior where evaluator clients receive generated candidates that fail during parameter-recovery simulation with errors like:

```text
TypeError: bad operand type for unary -: 'NoneType'
```

This usually means a generated model returned `None` instead of a numeric negative log-likelihood. In CMG mode, these failures currently become `RECOVERY_FAILED` and are not repaired, even though they are code/runtime errors that the evaluator LLM should try to fix.

## Background

In CMG mode:

1. The generator proposes `n_models` candidates.
2. Each numeric evaluator client fits one candidate by index.
3. Before fitting, parameter recovery may simulate synthetic subjects.
4. Simulation calls the generated model repeatedly on partial trial arrays.
5. If the generated model returns `None`, parameter recovery fails with:

```python
log_likes[opt] = -nll
```

because `nll` is `None`.

The existing CMG repair loop repairs only:

```python
VALIDATION_ERROR
FIT_ERROR
```

But this failure is currently returned as:

```python
RECOVERY_FAILED
```

So the repair loop does not run.

## Non-Goals

Do not change CMG assignment logic.

Do not disable parameter recovery.

Do not treat scientifically poor recovery as a syntax/runtime error. Only simulation/runtime failures should become repairable.

Do not change non-CMG behavior unless necessary and covered by tests.

---

## Chunk 1: Improve CMG Evaluator Logging

### Goal

Make evaluator logs clearly show which candidate each evaluator is fitting.

### Files

- `gecco/run_gecco.py`

### Current Problem

Logs currently show only the display model name, for example:

```text
Running parameter recovery check for comprehensive_mbmf_pers_asym
```

If several candidates share the same name, it looks like every evaluator is fitting the same candidate.

### Implementation Steps

In `_run_cmg_evaluator_iteration()` after `candidate` is selected, add a log line containing:

- evaluator index
- candidate index
- expected function name
- display name

Example output:

```text
CMG evaluator 3 fitting candidate index 3: cognitive_model4 (comprehensive_mbmf_pers_asym)
```

Use `console.print()`.

### Acceptance Criteria

When a CMG evaluator starts processing a candidate, logs identify:

- evaluator ID/index
- candidate index
- function name
- display name

This chunk does not need to change behavior.

### Suggested Test

Add or update a unit test in `tests/test_cmg_runtime.py` if there is already a convenient CMG evaluator test fixture. If not, manual dry-run/log verification is acceptable for this chunk.

---

## Chunk 2: Classify Parameter-Recovery Simulation Errors As Repairable

### Goal

Make CMG repair models that fail parameter-recovery simulation because the generated code returned invalid values or raised runtime errors.

### Files

- `gecco/run_gecco.py`

### Current Problem

In `_run_cmg_evaluator_iteration()`, repairable errors are currently:

```python
is_repairable_error = (
    result is not None
    and result.get("metric_name") in ("VALIDATION_ERROR", "FIT_ERROR")
)
```

Parameter recovery simulation failures return:

```python
metric_name == "RECOVERY_FAILED"
simulation_error is not None
recovery_n_successful == 0
```

These should be repairable because they represent runtime/code failure, not merely poor recovery.

### Implementation Steps

Add a helper function near the CMG methods:

```python
def _is_cmg_repairable_error(self, result: dict | None) -> bool:
    if result is None:
        return False

    metric_name = result.get("metric_name")
    if metric_name in ("VALIDATION_ERROR", "FIT_ERROR"):
        return True

    if metric_name == "RECOVERY_FAILED":
        return bool(result.get("simulation_error")) and result.get("recovery_n_successful", 0) == 0

    return False
```

Then replace the inline repairability check in `_run_cmg_evaluator_iteration()` with:

```python
is_repairable_error = self._is_cmg_repairable_error(result)
```

### Important Distinction

Only repair `RECOVERY_FAILED` when simulation failed entirely:

```python
simulation_error is present
recovery_n_successful == 0
```

Do not repair ordinary poor recovery like:

```python
mean_r < threshold
n_successful > 0
```

Those are legitimate scientific failures.

### Acceptance Criteria

CMG evaluator attempts repair when parameter recovery fails with a simulation/runtime error.

CMG evaluator does not attempt repair when parameter recovery runs successfully but mean recovery is below threshold.

---

## Chunk 3: Include Recovery Simulation Errors In Repair Feedback

### Goal

Ensure the repair prompt explains parameter-recovery simulation failures clearly.

### Files

- `gecco/run_gecco.py`

### Current Problem

`_build_syntax_error_feedback()` currently includes details for:

- `VALIDATION_ERROR`
- `FIT_ERROR`

It does not include useful details for:

- `RECOVERY_FAILED`

So if Chunk 2 makes recovery simulation failures repairable, the LLM may receive vague or missing error context.

### Implementation Steps

Update `_build_syntax_error_feedback()`.

Add an `elif` branch:

```python
elif error_type == "RECOVERY_FAILED":
    sim_err = result.get("simulation_error")
    if sim_err:
        error_messages.append(
            f"- {model_name}: parameter recovery simulation failed: {sim_err}. "
            "The model must always return a finite numeric negative log-likelihood, "
            "including when called on short prefix trial arrays during simulation."
        )
    else:
        error_messages.append(
            f"- {model_name}: parameter recovery failed."
        )
```

Keep the existing `VALIDATION_ERROR` and `FIT_ERROR` branches unchanged.

### Acceptance Criteria

When a CMG candidate returns `None` during parameter recovery, the repair prompt includes:

- the simulation error
- the requirement to return a finite numeric NLL
- the reminder that simulation uses short prefix arrays

---

## Chunk 4: Add A Pre-Recovery Return-Value Smoke Test

### Goal

Catch generated functions that return `None`, `NaN`, or non-numeric values before running full parameter recovery.

### Files

- `gecco/run_gecco.py`

### Current Problem

A bad generated model can enter expensive parameter recovery and fail only during simulation.

We can cheaply detect many cases immediately after `build_model_spec()`.

### Implementation Steps

Add a helper method near `_fit_candidate_model()`:

```python
def _smoke_test_model_return_value(self, spec) -> str | None:
    n = 3
    dummy_arrays = []
    for _ in getattr(self.cfg.data, "input_columns", []):
        dummy_arrays.append(np.zeros(n, dtype=np.int64))

    params = []
    for p in spec.param_names:
        lb, ub = spec.bounds[p]
        params.append((float(lb) + float(ub)) / 2.0)
    params = np.asarray(params, dtype=float)

    try:
        value = spec.func(*dummy_arrays, params)
    except Exception as e:
        return f"{type(e).__name__}: {e}"

    if value is None:
        return "Model returned None instead of a numeric negative log-likelihood."

    try:
        value = float(value)
    except Exception:
        return f"Model returned non-numeric value of type {type(value).__name__}."

    if not np.isfinite(value):
        return f"Model returned non-finite value: {value}."

    return None
```

Call this immediately after `spec = build_model_spec(...)` and before parameter recovery:

```python
smoke_error = self._smoke_test_model_return_value(spec)
if smoke_error:
    return {
        "function_name": display_name,
        "metric_name": "FIT_ERROR",
        "metric_value": float("inf"),
        "param_names": spec.param_names,
        "code": func_code,
        "error": smoke_error,
    }, False
```

### Notes For Junior Dev

`spec.func` expects one array per configured data input column, plus `model_parameters` as the final argument.

For the current two-step task, that is:

```python
(action_1, state, action_2, reward, model_parameters)
```

The generic `dummy_arrays` approach above should also work for future tasks with a different number of input columns.

### Acceptance Criteria

A model returning `None` is caught before parameter recovery.

The result is `FIT_ERROR`, so existing repair logic handles it.

The error message tells the LLM exactly what went wrong.

---

## Chunk 5: Add Tests For Repairable Recovery Simulation Failures

### Goal

Prevent regression.

### Files

- `tests/test_cmg_runtime.py`

### Test Cases

#### Test 1: Recovery simulation failure is repairable

Create a fake result:

```python
result = {
    "metric_name": "RECOVERY_FAILED",
    "simulation_error": "TypeError: bad operand type for unary -: 'NoneType'",
    "recovery_n_successful": 0,
}
```

Assert:

```python
search._is_cmg_repairable_error(result) is True
```

#### Test 2: Poor recovery is not repairable

Create a fake result:

```python
result = {
    "metric_name": "RECOVERY_FAILED",
    "simulation_error": None,
    "recovery_n_successful": 50,
    "recovery_r": 0.1,
}
```

Assert:

```python
search._is_cmg_repairable_error(result) is False
```

#### Test 3: Existing repairable errors still repair

Assert `VALIDATION_ERROR` and `FIT_ERROR` still return `True`.

#### Test 4: Smoke test catches `None`

Define a dummy function:

```python
def bad_model(action_1, state, action_2, reward, model_parameters):
    return None
```

Wrap it in a small `SimpleNamespace` with:

```python
func=bad_model
param_names=["alpha"]
bounds={"alpha": [0, 1]}
```

Assert smoke test returns an error string containing:

```text
returned None
```

#### Test 5: Smoke test accepts numeric return

Define:

```python
def good_model(action_1, state, action_2, reward, model_parameters):
    return 1.23
```

Assert smoke test returns `None`.

### Acceptance Criteria

All tests pass with:

```bash
conda run -n gecco_mh pytest tests/test_cmg_runtime.py -v
```

---

## Chunk 6: Optional Diagnostic Artifact For Failed CMG Candidates

### Goal

Make debugging failed candidates easier by saving the exact candidate metadata.

### Files

- `gecco/run_gecco.py`

### Implementation Steps

When `_fit_candidate_model()` returns a failed result in CMG mode, ensure the result includes:

```python
"candidate_index": idx
"expected_func_name": func_name
"display_name": display_name
```

This can be added in `_run_cmg_evaluator_iteration()` after receiving `result`, before `_finalize_iteration_results()`.

### Acceptance Criteria

Registry entries for failed CMG candidates identify:

- candidate index
- expected function name
- display name
- metric/error type
- simulation error if present

This chunk is helpful but not required to fix the repair loop.

---

## Chunk 7: Keep Joblib/Loky Temporary Files In The Run Directory

### Goal

Prevent `No space left on device` errors caused by joblib/loky writing temporary files to generic `/tmp` locations on shared nodes.

### Files

- `scripts/run_gecco_distributed.py`
- `scripts/run_judge_orchestrator.py` if the orchestrator ever runs parallel diagnostics
- `bash/run_gecco_distributed.sh`
- `bash/run_cmg_generator.sh`
- `bash/run_judge_orchestrator.sh`

### Current Problem

Parallel simulation uses loky/joblib-style worker processes. On clusters, these libraries often use `/tmp`, `/dev/shm`, or other node-local temporary directories by default. Those can fill up and produce:

```text
No space left on device
```

### Implementation Steps

Create a run-local temporary directory under the submitted project directory, for example:

```text
tmp/joblib
```

Set these environment variables before Python starts:

```bash
export GECCO_TMPDIR="${SLURM_SUBMIT_DIR:-$PWD}/tmp"
export TMPDIR="$GECCO_TMPDIR"
export JOBLIB_TEMP_FOLDER="$GECCO_TMPDIR/joblib"
export LOKY_TEMP_FOLDER="$GECCO_TMPDIR/joblib"
mkdir -p "$JOBLIB_TEMP_FOLDER"
```

Add the same block to all relevant shell wrappers before the Python command runs.

For direct Python execution without shell wrappers, add a small helper in Python startup scripts that sets defaults only if the variables are not already set:

```python
def configure_temp_dirs(project_root):
    tmp_root = Path(os.environ.get("GECCO_TMPDIR", project_root / "tmp"))
    joblib_tmp = tmp_root / "joblib"
    joblib_tmp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GECCO_TMPDIR", str(tmp_root))
    os.environ.setdefault("TMPDIR", str(tmp_root))
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(joblib_tmp))
    os.environ.setdefault("LOKY_TEMP_FOLDER", str(joblib_tmp))
```

Call this before any joblib/loky imports or parallel workers are created.

### Acceptance Criteria

SLURM logs print the temp directory being used.

`JOBLIB_TEMP_FOLDER` and `LOKY_TEMP_FOLDER` point inside the submitted project directory.

No joblib/loky temporary files are written to `/tmp` unless the user explicitly overrides the environment.

---

## Recommended Implementation Order

1. Chunk 1: logging
2. Chunk 2: repairability helper
3. Chunk 3: recovery error feedback
4. Chunk 4: smoke test
5. Chunk 5: tests
6. Chunk 6: optional diagnostic metadata
7. Chunk 7: joblib/loky temp directory configuration

Chunks 1, 2, 3, 6, and 7 are mostly independent.

Chunk 5 depends on Chunks 2 and 4.

Chunk 4 is independent but improves the quality of Chunk 2 by catching bad return values earlier.

## Verification Commands

Run targeted tests:

```bash
conda run -n gecco_mh pytest tests/test_cmg_runtime.py -v
```

Run CMG launcher dry-run to ensure unrelated launcher behavior remains OK:

```bash
conda run -n gecco_mh python scripts/launch_cmg_distributed.py \
  --config two_step_factors_gemini3flash_generic_lesion_no_tools_cmg.yaml \
  --dry-run \
  --conda-env gecco_mh
```

If practical, run a small CMG smoke experiment with:

- `centralized_model_generation.n_models: 2`
- `loop.max_iterations: 3`
- `parameter_recovery.n_subjects: 5`
- `parameter_recovery.n_trials: 20`

Expected behavior:

- If a generated candidate returns `None`, the evaluator attempts repair.
- Logs clearly identify evaluator/candidate/function name.
- Registry records the failure and repaired attempt clearly.
- Joblib/loky temp files are created under `tmp/joblib` in the run directory, not generic `/tmp`.
