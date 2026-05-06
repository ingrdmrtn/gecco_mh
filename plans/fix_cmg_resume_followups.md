# Fix Plan: CMG Resume Follow-Ups

## Purpose

The latest CMG review found a few remaining issues after the previous fixes. The main risk is resume behavior: clients can skip unfinished retrying iterations, and the generator does not resume from its CMG-specific status because it intentionally does not write evaluator `iteration_history` entries.

This plan is split into independent chunks so multiple developers can work in parallel. Each chunk includes the problem, files to edit, detailed implementation steps, and acceptance checks.

---

## Summary Of Issues

| Priority | Issue | Main Risk |
|---|---|---|
| High | Per-client resume treats `retrying` registry entries as completed. | A crashed client can skip an unfinished iteration after restart. |
| Medium | CMG generator resume ignores `generator_status` and `candidate_generations`. | Restarted generator repeats iteration 0 and makes duplicate LLM calls. |
| Low | CMG SLURM dry-run prints a Rich `Panel` object repr. | Launch plan output is confusing. |
| Low | Tests do not cover retrying resume or generator resume. | Remaining coordination regressions can pass unnoticed. |

---

## Chunk 1: Make Per-Client Resume Ignore Retrying Iterations

### Problem

`SharedRegistry.get_max_iteration_for_client()` currently scans `iteration_history` and returns the highest iteration for the requested client. However, `_update_registry(..., status="retrying")` also writes an `iteration_history` entry. That means a client that crashes while retrying can restart at the next iteration even though it never completed the current one.

### Files

- `gecco/coordination.py`
- `tests/test_cmg_registry.py`

### Implementation Steps

1. Update `SharedRegistry.get_max_iteration_for_client(client_id)` so it only counts iterations whose client entry status is final.

2. Treat these statuses as final:

```python
FINAL_STATUSES = {"complete", "complete_no_success"}
```

3. Inside `get_max_iteration_for_client()`, for each matching `iteration_history` entry:
   - get the matching `client_entries[str(client_id)]`
   - read its `status`
   - only consider the iteration if status is in `FINAL_STATUSES`

4. Be careful: `client_entries` stores only the latest status for a client, not status per iteration. This is acceptable for the current use case because resume only needs to decide whether the latest attempted iteration completed. Do not try to infer historical statuses for older iterations unless the registry schema is changed.

5. Suggested implementation:

```python
def get_max_iteration_for_client(self, client_id):
    """Return highest fully completed iteration for this client, or -1."""
    data = self.read()
    target = str(client_id)
    client_entry = data.get("client_entries", {}).get(target, {})
    if client_entry.get("status") not in ("complete", "complete_no_success"):
        return -1

    max_iter = -1
    for entry in data.get("iteration_history", []):
        it = entry.get("iteration")
        if it is not None and str(entry.get("client_id")) == target and it > max_iter:
            max_iter = it
    return max_iter
```

6. Add tests:
   - client `0` has a retrying entry for iteration `0`; helper returns `-1`
   - client `0` has a complete entry for iteration `0`; helper returns `0`
   - client `0` complete and client `1` retrying; helper returns `0` for client `0` and `-1` for client `1`

### Acceptance Checks

- A client with latest status `retrying` resumes the same iteration instead of skipping ahead.
- A client with latest status `complete` resumes after its completed iteration.
- Existing tests in `tests/test_cmg_registry.py` pass.

### Notes

This chunk is independent of generator resume. It fixes evaluator and normal distributed retry-resume behavior.

---

## Chunk 2: Add CMG Generator Resume Support

### Problem

The CMG generator intentionally does not write normal evaluator `iteration_history`. It writes `candidate_generations` and `generator_status` instead. The current `run_n_shots()` resume logic only checks `iteration_history`, so a restarted generator starts from iteration `0` again, repeats LLM calls, and overwrites local artifacts even though registry candidate publication is idempotent.

### Files

- `gecco/coordination.py`
- `gecco/run_gecco.py`
- `tests/test_cmg_registry.py`
- `tests/test_cmg_runtime.py`

### Implementation Steps

1. Add a registry helper for generator progress:

```python
def get_max_generator_iteration(self, client_id):
    """Return highest completed CMG generator iteration for this client, or -1."""
```

2. Implement it by reading `data.get("generator_status", {})` and scanning entries where:
   - `str(entry.get("client_id")) == str(client_id)`
   - `entry.get("status") == "complete"`

3. Convert iteration keys from strings to integers safely. Ignore keys that cannot be parsed as integers.

4. Optional additional safety: only count an iteration as complete if `candidate_generations[str(iteration)]` also exists. This prevents resuming past a status entry that somehow exists without candidates.

5. Suggested implementation:

```python
def get_max_generator_iteration(self, client_id):
    data = self.read()
    target = str(client_id)
    generations = data.get("candidate_generations", {})
    max_iter = -1
    for key, entry in data.get("generator_status", {}).items():
        try:
            it = int(key)
        except (TypeError, ValueError):
            continue
        if str(entry.get("client_id")) != target:
            continue
        if entry.get("status") != "complete":
            continue
        if key not in generations:
            continue
        if it > max_iter:
            max_iter = it
    return max_iter
```

6. In `GeCCoModelSearch.run_n_shots()`, compute `cmg_cfg = self._cmg_config()` before resume logic.

7. If CMG is enabled and `self._cmg_is_generator(cmg_cfg)` is true, use the new generator helper for resume:

```python
cmg_cfg = self._cmg_config()
if (
    self.shared_registry is not None
    and self.client_id is not None
    and cmg_cfg is not None
    and self._cmg_is_generator(cmg_cfg)
):
    max_existing = self.shared_registry.get_max_generator_iteration(self.client_id)
elif self.shared_registry is not None and self.client_id is not None:
    max_existing = self.shared_registry.get_max_iteration_for_client(self.client_id)
...
```

8. Avoid validating CMG config before checking `_cmg_is_generator()` if validation requires shared registry. In `run_n_shots()`, `self.shared_registry` already exists when this matters, so this should be safe.

9. Add tests:
   - generator status complete for iteration `0` and candidates exist; helper returns `0`
   - generator status failed for iteration `0`; helper returns `-1`
   - generator status complete but candidate generation missing; helper returns `-1`
   - `run_n_shots()` resume logic uses generator helper for generator client and evaluator helper for numeric evaluator client. Use mocks; do not run fitting or LLM calls.

### Acceptance Checks

- A restarted CMG generator with completed iteration `0` starts at iteration `1`.
- A failed generator iteration does not get skipped on restart.
- A generator completion without candidate data does not get skipped.
- Evaluator resume behavior still uses `get_max_iteration_for_client()`.

### Notes

This chunk is independent of Chunk 1, but the two should both be completed before relying on distributed resume in production.

---

## Chunk 3: Fix CMG Launcher Rich Panel Output

### Problem

`scripts/launch_cmg_distributed.py --slurm --dry-run` prints:

```text
<rich.panel.Panel object at ...>
```

This happens because the launcher calls Python `print(Panel(...))` instead of `console.print(Panel(...))`.

### Files

- `scripts/launch_cmg_distributed.py`
- `tests/test_cmg_launcher.py`

### Implementation Steps

1. In `scripts/launch_cmg_distributed.py`, replace:

```python
print(
    Panel(...)
)
```

with:

```python
console.print(
    Panel(...)
)
```

2. Leave ordinary command output as `print(...)` unless you want to convert the whole script consistently.

3. Update the launcher test to assert the output does not contain a Rich object repr:

```python
assert "<rich.panel.Panel object" not in output
```

4. Also assert that readable fields are present:

```python
assert "Generator Client" in output
assert "Evaluators" in output
```

### Acceptance Checks

- `conda run -n gecco_mh python scripts/launch_cmg_distributed.py --config two_step_factors_cmg.yaml --slurm --dry-run` prints a readable launch panel.
- The dry-run output no longer contains `<rich.panel.Panel object`.
- `tests/test_cmg_launcher.py` passes.

### Notes

This chunk is independent and safe to do in parallel with runtime fixes.

---

## Chunk 4: Add Targeted Resume Tests

### Problem

The current tests pass but do not cover the most important remaining edge cases:

- retrying entries should not count as completed for resume
- generator resume should use `generator_status` and `candidate_generations`

### Files

- `tests/test_cmg_registry.py`
- `tests/test_cmg_runtime.py`

### Implementation Steps

1. Add registry tests for retrying resume behavior:

```python
def test_get_max_iteration_for_client_ignores_retrying(registry):
    registry.update(client_id=0, iteration=0, results=[], status="retrying")
    assert registry.get_max_iteration_for_client(0) == -1
```

2. Add registry tests for completed resume behavior:

```python
def test_get_max_iteration_for_client_counts_complete(registry):
    registry.update(client_id=0, iteration=0, results=[], status="complete")
    assert registry.get_max_iteration_for_client(0) == 0
```

3. Add registry tests for generator resume helper:

```python
def test_get_max_generator_iteration_counts_complete_with_candidates(registry):
    registry.set_candidate_models(0, [{"index": 0, "func_name": "cognitive_model1", "code": "..."}], "generator")
    registry.set_generator_status(0, "generator", "complete", n_candidates=1)
    assert registry.get_max_generator_iteration("generator") == 0
```

4. Add generator failed/missing-candidate cases:
   - status `failed` returns `-1`
   - status `complete` but no `candidate_generations` entry returns `-1`

5. Add a lightweight runtime test for `run_n_shots()` resume branch if practical:
   - create a mock `GeCCoModelSearch` with `client_id="generator"`
   - attach `_cmg_config()` and `_cmg_is_generator()` real methods
   - mock `shared_registry.get_max_generator_iteration.return_value = 0`
   - avoid running the loop by setting `cfg.loop.max_iterations = 0`
   - assert `get_max_generator_iteration()` was called

6. Add evaluator branch equivalent:
   - `client_id=0`
   - assert `get_max_iteration_for_client(0)` was called

### Acceptance Checks

- Tests fail before Chunks 1 and 2, then pass after those chunks.
- Tests do not load real LLMs or fit models.
- `conda run -n gecco_mh pytest tests/test_cmg_registry.py tests/test_cmg_runtime.py` passes.

### Notes

This chunk can be done by a separate developer. It may be easiest to add expected tests first, then let Chunk 1 and Chunk 2 implementations make them pass.

---

## Suggested Parallel Work Split

- Developer A: Chunk 1, evaluator and normal distributed retry resume.
- Developer B: Chunk 2, CMG generator resume.
- Developer C: Chunk 3, launcher display cleanup.
- Developer D: Chunk 4, tests for resume edge cases.

Chunks 1 and 2 touch nearby code in `run_n_shots()` and `coordination.py`, so coordinate if editing at the same time. Chunk 3 is completely independent. Chunk 4 can be implemented first as failing tests or after the implementation chunks as regression tests.

---

## Final Verification

Run targeted tests:

```bash
conda run -n gecco_mh pytest tests/test_cmg_registry.py tests/test_cmg_runtime.py tests/test_cmg_launcher.py tests/test_cmg_judge.py
```

Run launcher dry-run checks:

```bash
conda run -n gecco_mh python scripts/launch_cmg_distributed.py --config two_step_factors_cmg.yaml --dry-run
conda run -n gecco_mh python scripts/launch_cmg_distributed.py --config two_step_factors_cmg.yaml --slurm --dry-run
```

Manual restart smoke checks, if resources are available:

1. Start evaluator `0`, let it enter `retrying`, then stop it and restart it. It should resume the same iteration, not skip ahead.
2. Start the CMG generator and let it publish iteration `0`, then restart it. It should resume at iteration `1`.
3. Force a generator failure before candidate publication, then restart it. It should retry the failed iteration, not skip ahead.
4. Confirm the launcher dry-run has readable Rich output and no `<rich.panel.Panel object ...>` text.
