# Fix Plan: CMG Review Findings

## Purpose

The centralized model generation (CMG) implementation mostly matches the intended design, but the latest review found several issues that can break distributed runs or leave confusing traces. This plan fixes those issues in independent chunks that can be worked on in parallel.

Each chunk lists the problem, files to edit, implementation steps, and acceptance checks. A junior developer should be able to implement a chunk without needing context from the other chunks.

---

## Summary Of Issues

| Priority | Issue | Main Risk |
|---|---|---|
| High | Distributed resume uses the global max iteration across all clients. | CMG evaluators can skip iteration 0 and deadlock the judge. |
| Medium | CMG repair prompt depends on `llm.include_feedback`. | Repair may omit the failing code and error details. |
| Medium | Empty candidate code finalizes as an empty result without repair. | Judge sees no useful error and evaluator never repairs. |
| Low/Medium | CMG judge short-circuit feedback is keyed as `default`. | Feedback traces are inconsistent and not generator-targeted. |
| Low | CMG launcher SLURM dry-run does not show actual SLURM commands. | Launcher topology is harder to verify before submission. |
| Low | CMG tests are incomplete and contain a stale numeric-generator test. | Regressions can pass unnoticed. |

---

## Chunk 1: Fix Per-Client Resume Semantics

### Problem

`GeCCoModelSearch.run_n_shots()` currently resumes from the highest iteration found anywhere in `iteration_history`. In distributed runs, this is unsafe. If evaluator `0` finishes iteration `0` before evaluator `1` starts, evaluator `1` sees max iteration `0`, starts at iteration `1`, and never evaluates candidate `1` for iteration `0`. The judge then waits for evaluator `1` at iteration `0` until timeout.

### Files

- `gecco/coordination.py`
- `gecco/run_gecco.py`
- `tests/test_cmg_runtime.py` or a new focused test file

### Implementation Steps

1. Add a new registry helper in `SharedRegistry`:

```python
def get_max_iteration_for_client(self, client_id):
    """Return the highest iteration completed by this client, or -1 if none."""
```

2. Implement it by scanning `data.get("iteration_history", [])` and only considering entries where `str(entry.get("client_id")) == str(client_id)`.

3. In `GeCCoModelSearch.run_n_shots()`, replace the global resume logic with per-client resume when `self.client_id is not None`:

```python
if self.shared_registry is not None and self.client_id is not None:
    max_existing = self.shared_registry.get_max_iteration_for_client(self.client_id)
elif self.shared_registry is not None:
    max_existing = self.shared_registry.get_max_iteration()
else:
    max_existing = -1
```

4. Keep the old global helper in place for any non-client use. Do not remove `get_max_iteration()`.

5. Add tests covering:
   - Client `0` has completed iteration `0`; client `1` has not. `get_max_iteration_for_client(1)` returns `-1`.
   - Client `0` resumes from `1` after completing iteration `0`.
   - Client `1` starts at `0` when only client `0` has written results.

### Acceptance Checks

- A slow or restarted CMG evaluator does not skip an iteration completed by another evaluator.
- Existing single-client/local resume behavior remains unchanged.
- `conda run -n gecco_mh pytest tests/test_cmg_registry.py tests/test_cmg_runtime.py` passes.

### Notes

This chunk is the highest priority. It is independent of the repair, judge, launcher, and test-cleanup chunks.

---

## Chunk 2: Make CMG Repair Prompt Always Include Repair Details

### Problem

`_repair_cmg_candidate()` correctly builds a repair section containing the failing code, assigned function name, and error details. However, it passes that text as `feedback_text` into `build_input_prompt()`. `build_prompt()` only includes feedback when `cfg.llm.include_feedback` is true, so repair details can be silently omitted.

### Files

- `gecco/prompt_builder/prompt.py`
- `gecco/run_gecco.py`
- `tests/test_cmg_runtime.py` or a new focused test file

### Implementation Steps

1. Add an optional force flag to prompt building:

```python
def build_prompt(..., n_models: Optional[int] = None, force_include_feedback: bool = False):
```

2. Change the feedback section condition to include feedback when either normal config allows it or the caller forces it:

```python
feedback_enabled = include_feedback or force_include_feedback
feedback_section = (
    f"\n\n### Feedback from previous iterations\n{feedback_text.strip()}"
    if (feedback_text and feedback_enabled)
    else ""
)
```

3. Add the same optional argument to `PromptBuilderWrapper.build_input_prompt(...)` and pass it through to `build_prompt(...)`.

4. In `_repair_cmg_candidate()`, call:

```python
prompt = self.prompt_builder.build_input_prompt(
    feedback_text=repair_section,
    n_models=1,
    force_include_feedback=True,
)
```

5. If the naive repair path is enabled, ensure the second-phase translation prompt also gets `force_include_feedback=True`. This may require adding the same argument to `generate_models_naive(...)` and passing it through to `build_input_prompt(...)`.

6. Keep normal generation behavior unchanged. Calls that do not set `force_include_feedback=True` should still respect `cfg.llm.include_feedback`.

### Acceptance Checks

- A unit test with `llm.include_feedback = False` confirms the CMG repair prompt still contains:
  - the failing code
  - the assigned function name
  - the error text
- A normal non-repair prompt with `llm.include_feedback = False` still omits feedback.
- Existing CMG config with `include_feedback: true` still behaves the same.

### Notes

This chunk is independent of resume and judge changes.

---

## Chunk 3: Treat Empty Candidate Code As A Repairable Validation Error

### Problem

`_fit_candidate_model()` returns `None, False` when candidate code is empty. In CMG evaluator mode, this finalizes an empty result list and does not trigger repair. Empty generated code should produce a normal error result so the evaluator can repair it or report a clear failure.

### Files

- `gecco/run_gecco.py`
- `tests/test_cmg_runtime.py` or a new focused test file

### Implementation Steps

1. In `_fit_candidate_model()`, replace the empty-code early return with a `VALIDATION_ERROR` result.

Current behavior:

```python
if not func_code:
    return None, False
```

New behavior:

```python
if not func_code:
    return {
        "function_name": display_name,
        "metric_name": "VALIDATION_ERROR",
        "metric_value": float("inf"),
        "param_names": [],
        "code": func_code,
        "error_type": "empty_code",
        "error_message": f"No code provided for {func_name}",
        "error_details": {"expected_func_name": func_name},
    }, False
```

2. Confirm this does not break normal mode. In normal mode, empty generated candidates will now appear in the iteration result JSON as validation errors instead of being silently skipped.

3. In `_run_cmg_evaluator_iteration()`, no special case should be needed because `VALIDATION_ERROR` is already repairable.

4. Add a test that calls `_fit_candidate_model()` or a small extracted helper with empty `code` and confirms it returns a `VALIDATION_ERROR` result.

5. Add a CMG evaluator repair-loop test if practical: empty candidate code should call `_repair_cmg_candidate()` once when retries remain.

### Acceptance Checks

- Empty candidate code produces a normal result dictionary, not `None`.
- CMG evaluator attempts repair for empty candidate code when retries remain.
- Final exhausted failure is saved as `VALIDATION_ERROR`, not an empty result list.

### Notes

This chunk is independent of prompt-forcing. The repair may still be low quality until Chunk 2 is done, but the evaluator will at least enter the repair loop.

---

## Chunk 4: Key CMG Judge Short-Circuit Feedback By Generator

### Problem

The main CMG judge path writes feedback keyed by the generator profile. The no-runnable-model fallback also does this. However, the judge short-circuit path still writes `{"default": ...}`. The generator can fall back to `default`, but the trace and registry are inconsistent with the CMG design.

### Files

- `scripts/run_judge_orchestrator.py`
- `tests/test_cmg_runtime.py` or a new focused test file

### Implementation Steps

1. Find the short-circuit branch in `scripts/run_judge_orchestrator.py`:

```python
if analysis_data.get("short_circuit"):
    synthesized_feedback = {"default": analysis_data["analysis_text"]}
```

2. Change it to respect CMG mode:

```python
if analysis_data.get("short_circuit"):
    if cmg_enabled:
        generator_name = getattr(cmg_cfg, "generator_client", "generator")
        synthesized_feedback = {generator_name: analysis_data["analysis_text"]}
    else:
        synthesized_feedback = {"default": analysis_data["analysis_text"]}
```

3. Leave the rest of the judge synthesis logic unchanged.

4. Add a small test or mock-based check that when `cmg_enabled` and `short_circuit` are true, `synthesized_feedback` is keyed by the generator name.

### Acceptance Checks

- In CMG mode, all judge feedback paths write feedback keyed by `centralized_model_generation.generator_client`.
- In non-CMG mode, short-circuit feedback remains keyed by `default`.
- Judge trace `personas` contains the generator profile name in CMG short-circuit cases.

### Notes

This chunk is independent of launcher, resume, and repair changes.

---

## Chunk 5: Make CMG Launcher SLURM Dry-Run Show Actual SLURM Commands

### Problem

`scripts/launch_cmg_distributed.py --slurm --dry-run` currently returns after printing the local launch plan. It does not show the actual `sbatch` generator job, evaluator array, or orchestrator job. This makes the fixed SLURM topology harder to verify before submitting jobs.

### Files

- `scripts/launch_cmg_distributed.py`
- `tests/test_cmg_runtime.py` or a new launcher test file

### Implementation Steps

1. Do not return immediately from `if args.dry_run` before the `args.slurm` branch.

2. Restructure the control flow:
   - Always print the high-level launch plan.
   - If `args.slurm` is true, build and print the three `sbatch` commands.
   - Pass `dry_run=args.dry_run` into `run_cmd(...)` for each `sbatch` command.
   - If `args.slurm` is false and `args.dry_run` is true, print only the local commands and return.

3. Ensure `run_cmd(...)` is called like this in SLURM mode:

```python
gen_job_id = run_cmd(gen_job_cmd, dry_run=args.dry_run)
run_cmd(eval_job_cmd, dry_run=args.dry_run)
run_cmd(orch_job_cmd, dry_run=args.dry_run)
```

4. In dry-run mode, avoid printing misleading job IDs when no job was submitted.

5. Add a test that captures `stdout` from `main()` with `--slurm --dry-run` and asserts:
   - one generator `sbatch` command contains `--client-profile generator`
   - evaluator array contains `--array=0-<n_models - 1>`
   - evaluator array does not contain `--array=0-<n_models>`
   - no evaluator command uses the generator profile

### Acceptance Checks

- `conda run -n gecco_mh python scripts/launch_cmg_distributed.py --config two_step_factors_cmg.yaml --slurm --dry-run` prints actual `sbatch` commands.
- The dry-run output shows evaluator IDs `0..n_models-1` only.
- The generator is a non-array job using `--client-profile <generator_client>`.
- No jobs are submitted in dry-run mode.

### Notes

This chunk is independent of runtime behavior.

---

## Chunk 6: Clean Up And Expand CMG Tests

### Problem

The CMG tests do not cover all review findings, and one test still asserts that numeric generator string equality works even though numeric generator IDs are now invalid at runtime.

### Files

- `tests/test_cmg_registry.py`
- `tests/test_cmg_runtime.py`
- Optional: `tests/test_cmg_launcher.py`
- Optional: `tests/test_cmg_judge.py`

### Implementation Steps

1. Remove or rewrite the stale test:

```python
def test_cmg_is_generator_numeric_str():
```

Recommended replacement: assert that `_validate_cmg_runtime()` rejects numeric generator IDs. This test already exists, so deleting the stale equality test is acceptable.

2. Add tests for Chunk 1:
   - per-client max iteration helper
   - one client does not resume based on another client’s history

3. Add tests for Chunk 2:
   - forced repair feedback is included even when `llm.include_feedback` is false
   - normal feedback remains omitted when not forced

4. Add tests for Chunk 3:
   - empty candidate code returns `VALIDATION_ERROR`
   - CMG evaluator considers that error repairable

5. Add tests for Chunk 4:
   - CMG short-circuit judge feedback is keyed by generator
   - non-CMG short-circuit remains keyed by `default`

6. Add tests for Chunk 5:
   - `--slurm --dry-run` prints generator non-array job and evaluator `0..n_models-1` array

7. Keep tests small and mock-heavy. Do not load real LLMs or run model fitting.

### Acceptance Checks

- `conda run -n gecco_mh pytest tests/test_cmg_registry.py tests/test_cmg_runtime.py` passes.
- If new test files are added, include them in the command and verify they pass.
- Tests do not require external APIs, OpenRouter, vLLM, or GPU resources.

### Notes

This chunk can be done after or alongside the implementation chunks. Some tests will fail until their corresponding chunks are implemented.

---

## Suggested Parallel Work Split

These chunks can mostly be processed in parallel:

- Developer A: Chunk 1, because it is the highest-priority runtime coordination bug.
- Developer B: Chunks 2 and 3, because both affect evaluator repair behavior.
- Developer C: Chunk 4, because it only touches judge feedback keying.
- Developer D: Chunk 5, because it only touches launcher dry-run behavior.
- Developer E: Chunk 6, either after the other chunks or by adding expected-failing tests first.

---

## Final Verification

After all chunks are implemented, run:

```bash
conda run -n gecco_mh pytest tests/test_cmg_registry.py tests/test_cmg_runtime.py
```

If additional CMG test files were added, include them too:

```bash
conda run -n gecco_mh pytest tests/test_cmg_registry.py tests/test_cmg_runtime.py tests/test_cmg_launcher.py tests/test_cmg_judge.py
```

Then run launcher dry-runs:

```bash
conda run -n gecco_mh python scripts/launch_cmg_distributed.py --config two_step_factors_cmg.yaml --dry-run
conda run -n gecco_mh python scripts/launch_cmg_distributed.py --config two_step_factors_cmg.yaml --slurm --dry-run
```

Manual CMG smoke test, if resources are available:

1. Launch generator profile `generator`.
2. Launch evaluator clients `0` and `1`.
3. Launch judge orchestrator.
4. Verify `iteration_history` has evaluator clients only.
5. Verify no evaluator skips iteration `0` because another evaluator completed it.
6. Verify judge feedback is always keyed by `generator`.
7. Verify malformed or empty candidate code creates a repair attempt or a clear `VALIDATION_ERROR` result.
