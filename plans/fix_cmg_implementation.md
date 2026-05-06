# Fix Plan: Centralized Model Generation Implementation

## Purpose

The centralized model generation (CMG) implementation is close to the intended design, but the review found several issues that can break distributed runs or create confusing side effects. This plan describes the fixes in small, ordered, vertical chunks.

Each chunk should be implemented and verified before moving to the next. The goal is to leave little room for interpretation: follow the requirements, edit the named files, and run the listed checks.

---

## Current Intended CMG Behavior

CMG mode should work like this:

1. One generator client is launched with a named profile, for example `--client-profile generator`.
2. The generator receives judge feedback, generates exactly `centralized_model_generation.n_models` candidates, and publishes them to the shared registry.
3. The generator does not fit models and does not write normal evaluator results to `iteration_history`.
4. Evaluator clients are launched with numeric IDs `0..n_models-1`.
5. Evaluator `0` fits candidate `0`, evaluator `1` fits candidate `1`, and so on.
6. Evaluators may repair assigned code after validation/fitting failure, but they must preserve the assigned function name.
7. The judge waits for exactly `n_models` evaluator completions, not for the generator.
8. The judge writes feedback for the generator persona only.

---

## Issues To Fix

| Severity | Issue |
|---|---|
| High | SLURM CMG launcher assigns wrong evaluator IDs. |
| High | Evaluator repair prompt does not include the original candidate code or normal task context. |
| High | Normal non-CMG distributed retry behavior regressed because completion is published before retry decision. |
| High | CMG generator still runs post-search simulation/test-fitting code intended for evaluators. |
| Medium | Runtime validation allows numeric `generator_client`, which conflicts with evaluator IDs. |
| Medium | Repaired code updates registry but not the evaluator model artifact file. |
| Medium | Repair path bypasses naive/profile-specific repair behavior. |
| Medium | Function-name preservation check is only a substring check. |
| Medium | No checked-in CMG example config exists. |
| Low | Generator failures before publication do not record failed generator status. |
| Low | Judge no-runnable-model fallback writes `default` feedback instead of generator-keyed feedback. |

---

## Chunk 1: Fix Normal-Mode Retry Publication Ordering

### Rationale

This is the most important side effect outside CMG. The current refactor calls `_finalize_iteration_results()` before deciding whether all models failed and a syntax retry is needed. `_finalize_iteration_results()` writes `complete` or `complete_no_success` to the shared registry. In distributed normal mode, the orchestrator can see that completion and proceed before the client retries.

Normal-mode behavior must match the pre-CMG behavior: if all models fail with `VALIDATION_ERROR` or `FIT_ERROR` and retries remain, the client should mark itself `retrying`, regenerate, and only publish completion after final success or retry exhaustion.

### Files

- `gecco/run_gecco.py`

### Implementation

Do not call `_finalize_iteration_results()` before the syntax retry decision.

In the normal path inside `run_n_shots()`:

1. Generate and fit models into `iteration_results` exactly as now.
2. Compute `all_syntax_errors` before publishing final completion to the registry.
3. If `all_syntax_errors` and `syntax_retry_count < max_syntax_retries`:
   - Save local artifacts if needed, but do not publish `complete` or `complete_no_success`.
   - Set `feedback = self._build_syntax_error_feedback(iteration_results)`.
   - Increment `syntax_retry_count`.
   - Call `_update_registry(it, [], status="retrying")` if a shared registry exists.
   - `continue` the retry loop.
4. Only after deciding not to retry, call `_finalize_iteration_results(...)`.

Recommended low-risk change:

- Keep `_finalize_iteration_results()` as the final publish helper.
- Move the call to `_finalize_iteration_results()` below the retry check.
- If you still need to save failed attempt JSONs before retry, add a separate helper later. Do not publish completion for retry attempts.

### Acceptance Checks

- In non-CMG normal mode, a full syntax failure with retries remaining should write registry status `retrying`, not `complete_no_success`.
- The orchestrator should not count a retrying client as complete.
- Existing successful normal-mode iteration should still write result JSON, diagnostics, feedback history, and registry completion.

---

## Chunk 2: Fix CMG SLURM Launch Topology

### Rationale

The current CMG SLURM launcher submits a single array `0-{n_models}` where task `0` is the generator and task `1` becomes evaluator ID `1`. This leaves evaluator `0` missing and creates an out-of-range evaluator `n_models`.

CMG must launch exactly:

- One generator job with `--client-profile <generator_client>`.
- One evaluator array with IDs `0..n_models-1`.

### Files

- `scripts/launch_cmg_distributed.py`
- Possibly `bash/run_gecco_distributed.sh`, only if needed to support separate generator/evaluator submission cleanly.

### Implementation

Change SLURM mode in `scripts/launch_cmg_distributed.py`.

Do not submit one combined array containing both generator and evaluators.

Submit the generator as a single job. The command should ensure the Python client receives:

```bash
--client-profile <generator_client>
```

The generator should not receive a numeric evaluator client ID that causes assignment confusion. If the existing shell wrapper always passes `--client-id`, either:

1. Add a new wrapper mode for generator jobs, or
2. Submit the generator command directly with `sbatch --wrap`, or
3. Add a role/profile argument to the wrapper so it can omit numeric evaluator behavior for generator jobs.

Then submit evaluators as an array:

```bash
--array=0-<n_models - 1>
```

Each evaluator task should run:

```bash
python scripts/run_gecco_distributed.py --config <config> --client-id "$SLURM_ARRAY_TASK_ID"
```

Do not pass evaluator profiles unless explicitly desired later. Numeric IDs are the assignment mechanism.

The orchestrator job may be submitted separately as it is now, but it should depend on the same registry/results directory and expect `n_models` evaluator completions.

### Acceptance Checks

- Dry run prints exactly one generator command and evaluator commands/array covering `0..n_models-1`.
- In SLURM mode, evaluator `0` exists.
- In SLURM mode, no evaluator with ID `n_models` exists.
- The generator is launched with `--client-profile <generator_client>`.
- The generator is not counted as evaluator `0`.

---

## Chunk 3: Skip Evaluator-Only Post-Processing for the CMG Generator

### Rationale

The generator does not fit candidates, so after `run_n_shots()` it may have no local `best_model`, or it may have a synced global best with an invalid `best_iter` for test fitting. The distributed script currently runs simulation/test fitting for all clients, including the CMG generator.

This causes noisy errors and confusing artifacts.

### Files

- `scripts/run_gecco_distributed.py`

### Implementation

Add a small helper or inline check after config/profile resolution:

```python
cmg_cfg = getattr(cfg, "centralized_model_generation", None)
cmg_enabled = cmg_cfg is not None and getattr(cmg_cfg, "enabled", False)
is_cmg_generator = (
    cmg_enabled
    and args.client_profile is not None
    and str(args.client_profile) == str(getattr(cmg_cfg, "generator_client", ""))
)
```

After `search.run_n_shots(...)`, if `is_cmg_generator` is true:

- Print a clear message such as `CMG generator complete; skipping evaluator-only simulation and test fitting.`
- Skip simulation generation.
- Skip best-model test fitting.
- Do not print misleading `Best BIC`/`Parameters` panel if no best model exists.

Implementation pattern:

```python
best_model, best_bic, best_params = search.run_n_shots(...)

if is_cmg_generator:
    console.print("[green]CMG generator run complete; skipping evaluator-only post-processing.[/]")
    continue
```

Make sure `continue` targets the independent-run loop, not the whole script in a way that skips final summary unexpectedly. If final summary relies on `global_best_bic`, print a generator-specific final summary when `is_cmg_generator` is true.

### Acceptance Checks

- Running the generator does not attempt `run_fit(df_test, best_model, ...)`.
- Generator logs do not contain `Error fitting cognitive_model-1`.
- Evaluator clients still run normal post-processing if appropriate.

---

## Chunk 4: Reject Numeric Generator IDs

### Rationale

The design reserves numeric client IDs for evaluators. A numeric `generator_client`, such as `"0"`, can cause evaluator `0` to become the generator and leave candidate `0` without an evaluator.

### Files

- `gecco/run_gecco.py`
- `scripts/run_judge_orchestrator.py`
- `scripts/launch_cmg_distributed.py`

### Implementation

Add validation wherever CMG config is validated.

Create a simple local check:

```python
generator_client = str(getattr(cmg_cfg, "generator_client", ""))
if generator_client.isdigit():
    raise ValueError(
        "centralized_model_generation.generator_client must be a named profile, "
        "not a numeric evaluator ID"
    )
```

In scripts that use `sys.exit(1)`, print the same message and exit.

### Acceptance Checks

- Config with `generator_client: "generator"` passes validation.
- Config with `generator_client: "0"` fails before generation or orchestration starts.
- Evaluators with numeric `--client-id 0` still work.

---

## Chunk 5: Make Evaluator Repair a Real Candidate Repair

### Rationale

The repair prompt currently includes only the error text. It does not include the candidate code that failed, the model name, rationale, parameters, task context, template, or guardrails. This makes it unlikely to repair the assigned candidate and risks generating a different model.

Evaluator repair should keep using the existing generation pipeline, but the prompt must include the assigned candidate and clear repair constraints.

### Files

- `gecco/run_gecco.py`
- Possibly `gecco/prompt_builder/prompt.py` if you decide to add a dedicated repair prompt builder.

### Implementation

Update `_repair_cmg_candidate(...)` so the repair prompt includes:

- Candidate name.
- Candidate rationale if available.
- Candidate parameters if available.
- Current failing code.
- Error feedback from `_build_syntax_error_feedback([error_result])`.
- Explicit required function name.
- Explicit instruction to repair the same conceptual model.

Minimum prompt content:

```python
current_code = current_model_dict.get("code", "")
candidate_name = current_model_dict.get("name", expected_func_name)
candidate_params = current_model_dict.get("parameters", [])
candidate_rationale = candidate.get("rationale", "")

repair_feedback = f"""
The assigned candidate model failed validation or fitting.

You must repair this exact candidate. Do not propose a new model idea.

Assigned function name: `{expected_func_name}`
Candidate name: {candidate_name}
Candidate rationale: {candidate_rationale}
Candidate parameters: {candidate_params}

Current code:
```python
{current_code}
```

Error:
{error_feedback}

Requirements:
- Return exactly one repaired model.
- The repaired code must define `{expected_func_name}` exactly.
- Keep the same conceptual mechanism unless a small change is necessary to make it runnable.
- Keep parameter declarations consistent with the repaired code.
"""
```

Then build the normal task prompt around this repair feedback:

```python
prompt = self.prompt_builder.build_input_prompt(
    feedback_text=repair_feedback,
    n_models=1,
)
_, new_models = self.generate_models(prompt, n_models=1)
```

This gives the LLM the task context, template, guardrails, and structured-output instructions.

If the evaluator has naive ideation enabled, use `generate_models_naive(repair_feedback, n_models=1)` to preserve existing per-client behavior. Otherwise use the normal prompt path above.

### Acceptance Checks

- Repair prompt contains the original failing code.
- Repair prompt contains the assigned function name.
- Repair prompt contains normal task/template/guardrail context through `build_input_prompt()`.
- Evaluator repair still returns exactly one parsed model.

---

## Chunk 6: Structurally Validate Repaired Function Name Before Registry Update

### Rationale

The current check only tests whether the expected function name appears as a substring in repaired code. That can pass for comments or strings. The registry should only be overwritten when the repaired code actually defines the assigned function.

### Files

- `gecco/run_gecco.py`

### Implementation

Before calling `update_candidate_model(...)`, validate the repaired code structurally.

Preferred approach:

Use existing model validation utilities that already know how to build model specs:

```python
from gecco.offline_evaluation.utils import build_model_spec

try:
    build_model_spec(
        repaired_code,
        expected_func_name=expected_func_name,
        cfg=self.cfg,
        structured_params=repaired.get("parameters", []),
    )
except Exception as e:
    console.print(
        f"[yellow]Repaired code does not define a valid `{expected_func_name}`: {e}[/]"
    )
    return current_model_dict
```

Do not update the registry if this validation fails.

Do not rely on:

```python
if expected_func_name not in repaired_code:
```

That is not sufficient.

### Acceptance Checks

- Code with `expected_func_name` only in a comment does not update the registry.
- Code defining `cognitive_model1` when evaluator index requires `cognitive_model2` does not update the registry.
- Valid repaired code defining the assigned function does update the registry.

---

## Chunk 7: Keep Repaired Registry Code and Artifact File Consistent

### Rationale

Evaluator artifacts currently write the original assigned code before repair. If repair succeeds, fitting uses the repaired code, but the `code_file` artifact still contains the old code. This is confusing and can mislead debugging.

### Files

- `gecco/run_gecco.py`

### Implementation

When `_repair_cmg_candidate(...)` returns repaired code, ensure the evaluator model file is updated before the next fit.

Simple approach:

- Pass `model_file` into `_repair_cmg_candidate(...)`, or write the model file immediately after repair in `_run_cmg_evaluator_iteration(...)`.

Recommended change in `_run_cmg_evaluator_iteration(...)` after repair returns:

```python
current_model_dict = self._repair_cmg_candidate(...)
with open(model_file, "w") as f:
    f.write(current_model_dict.get("code", ""))
```

This means `code_file` always points to the latest code attempted.

### Acceptance Checks

- After successful repair, the evaluator model `.txt` file contains repaired code.
- Result JSON `code` field and `code_file` contents agree.

---

## Chunk 8: Record Generator Failure Status

### Rationale

If the generator fails before publishing candidates, evaluators wait until timeout. That may be acceptable operationally, but the registry should show that the generator failed.

### Files

- `gecco/run_gecco.py`
- `gecco/coordination.py` only if `set_generator_status()` needs an optional error field.

### Implementation

Extend `set_generator_status(...)` to accept an optional `error` argument:

```python
def set_generator_status(self, iteration, client_id, status, n_candidates=None, error=None):
```

Store `error` when provided.

In `_run_cmg_generator_iteration(...)`, wrap generation/publication in `try/except`:

```python
try:
    ... generate and publish ...
    self.shared_registry.set_generator_status(..., status="complete", ...)
except Exception as e:
    self.shared_registry.set_generator_status(
        iteration=it,
        client_id=self.client_id,
        status="failed",
        n_candidates=0,
        error=str(e),
    )
    raise
```

Do not swallow the exception. The generator should still fail loudly.

### Acceptance Checks

- If generation produces the wrong number of candidates, `generator_status[str(it)].status == "failed"`.
- The original exception still reaches logs/Sentry.

---

## Chunk 9: Write Generator-Keyed Feedback for CMG No-Model Fallback

### Rationale

The normal CMG judge path writes feedback keyed by the generator profile. The no-runnable-model fallback currently writes `default`. The generator can consume `default`, but this is inconsistent and makes traces less clear.

### Files

- `scripts/run_judge_orchestrator.py`

### Implementation

In the `clients_with_models == 0` branch, if `cmg_enabled` is true, write synthesized feedback keyed by `generator_client`:

```python
if cmg_enabled:
    generator_name = getattr(cmg_cfg, "generator_client", "generator")
    synthesized_feedback = {
        generator_name: "All evaluator-assigned models failed syntax validation after retries. Review error messages and try a different approach."
    }
else:
    synthesized_feedback = {
        "default": "All models failed syntax validation after retries. Review error messages and try a different approach."
    }
```

Then pass `synthesized_feedback` to `registry.set_judge_feedback(...)`.

### Acceptance Checks

- In CMG mode with zero runnable evaluator models, registry feedback is keyed by generator profile.
- Non-CMG behavior still writes `default`.

---

## Chunk 10: Add a Checked-In CMG Example Config

### Rationale

The launcher requires a config with `centralized_model_generation`, but no current config contains that section. A minimal example reduces launch mistakes.

### Files

- Add a config file under `config/`, for example `config/two_step_factors_cmg.yaml`.

### Implementation

Create a config by copying the existing distributed config or using YAML anchors if the project prefers that style.

At minimum, it must include:

```yaml
judge:
  orchestrated: true
  mode: tool_using

centralized_model_generation:
  enabled: true
  generator_client: "generator"
  n_models: 4

clients:
  generator:
    llm:
      system_prompt_suffix: |
        Focus on proposing diverse, creative candidate models for parallel evaluation.
```

Make sure the generator profile exists under `clients:` and is not numeric.

Make sure the config has all existing required task/data/llm/evaluation sections.

### Acceptance Checks

- `python scripts/launch_cmg_distributed.py --config two_step_factors_cmg.yaml --dry-run` succeeds.
- Dry run prints one generator and exactly `n_models` evaluators.
- `python scripts/run_judge_orchestrator.py --config two_step_factors_cmg.yaml --help` still works.

---

## Chunk 11: Add Targeted Tests

### Rationale

The existing registry tests are useful but do not cover the most failure-prone CMG behavior.

### Files

- `tests/test_cmg_registry.py`
- Add a new test file if easier, for example `tests/test_cmg_runtime.py`.

### Tests To Add

Add tests for:

1. Numeric generator rejected.
2. Evaluator index mapping:
   - `client_id=0`, `n_models=2` returns `0`.
   - `client_id=1`, `n_models=2` returns `1`.
   - `client_id=2`, `n_models=2` returns `None`.
3. Repaired function-name validation rejects wrong function names.
4. CMG judge expected client count uses `n_models`, not `loop.n_clients`.
5. Launcher dry-run output includes evaluator IDs `0..n_models-1` and does not include evaluator `n_models`.

Keep tests small. Use mocks/stubs instead of loading real LLMs.

### Acceptance Checks

- `conda run -n gecco_mh pytest tests/test_cmg_registry.py` passes.
- New targeted tests pass under `conda run -n gecco_mh pytest <new-test-file>`.

---

## Final Manual Smoke Test

After all chunks are complete, run a small CMG smoke test.

### Setup

Use a config with:

```yaml
centralized_model_generation:
  enabled: true
  generator_client: "generator"
  n_models: 2
```

### Launch

Use dry run first:

```bash
conda run -n gecco_mh python scripts/launch_cmg_distributed.py --config <cmg-config>.yaml --dry-run
```

Then launch one generator, two evaluators, and one judge using the printed commands or SLURM mode.

### Verify Registry

Check `results/<task>/shared_registry.json`:

- `candidate_generations["0"].candidates` has exactly 2 candidates.
- Candidate indexes are `0` and `1`.
- Function names are `cognitive_model1` and `cognitive_model2`.
- `generator_status["0"].status == "complete"`.
- `iteration_history` has evaluator clients `0` and `1` only.
- No generator entry appears in `iteration_history`.
- Judge feedback is keyed by `generator`.

### Verify Logs

- Generator logs do not show fitting or test fitting.
- Evaluator `0` fits candidate `0`.
- Evaluator `1` fits candidate `1`.
- Judge waits for exactly 2 clients.

---

## Do Not Do These In This Fix Pass

- Do not add dashboards/monitoring for CMG.
- Do not add claim leases or reclaiming.
- Do not preserve original generated code history in the registry.
- Do not add explicit evaluator lists.
- Do not support CMG without orchestrated judge.
- Do not make the generator fit candidates.
- Do not add a separate CMG repair retry config.
