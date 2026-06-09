# Phase 6 Review Fixes Plan

This is **not a new phase**. Keep this follow-up narrow and limited to the review fixes below.

## Purpose / Target final state

Make the Phase 6 review follow-up easy to execute for a junior developer: only the reviewed regressions are fixed, the tests clearly describe the intended behavior, and the final diff stays free of dashboard scope creep.

## Non-goals

- No dashboard/provider/export/report work.
- No broad lifecycle abstractions.
- No JSON runtime truth source.
- No expensive or flaky tests.

## TDD rules

- Start each chunk with tests/guards before production edits.
- Keep edits minimal and local to the reviewed regression.
- Use type annotations and Google-style docstrings for any new helper functions.
- Format Python changes with Black.

## Chunk 0: Revert/split dashboard scope creep

### Goal

Remove dashboard changes from this status-ownership follow-up.

### Required actions

- Revert or split out any Phase 6 edits in:
  - `gecco-mh-dashboard/app.py`
  - `gecco-mh-dashboard/dashboard/config.py`
- Leave dashboard DuckDB support for a separate approved plan/phase.

### Verification

- Run `git diff --name-only` for this follow-up and verify it does **not** list `gecco-mh-dashboard/app.py` or `gecco-mh-dashboard/dashboard/config.py`.
- Do **not** replace those changes with alternative dashboard edits.
- Confirm the final diff for this phase does not include dashboard files.

## Chunk 1: Add `run_n_shots` CMG `complete_no_success` preservation regression

### Test-first target

Add a regression test in `tests/test_cmg_runtime.py`.

### Setup

- Use a real `SharedRegistry(tmp_path / "shared_registry.duckdb")`.
- Build a minimal `SimpleNamespace`/`MagicMock` search object with:
  - `cfg`
  - `client_id=0`
  - `shared_registry`
  - `distributed_coordinator.start_iteration`
  - `_cmg_config`
  - `_cmg_is_generator`
  - `_cmg_evaluator_index`
  - `_validate_cmg_runtime`
  - `_require_distributed_coordinator`
  - `_sync_from_registry`
  - `_set_activity`
  - best attrs, `feedback.record_iteration`, `tried_param_sets`, and `results_dir`
- Bind `GeCCoModelSearch.run_n_shots` onto the fake search object.
- Fake `_run_cmg_evaluator_iteration` should call `shared_registry.update(..., status="complete_no_success", had_runnable_model=False, results=[...])`.
- Let `run_n_shots()` finish normally.

### Assertion

- Final registry snapshot still has:
  - `status == "complete_no_success"`
  - `had_runnable_model is False`
- Also assert the final snapshot entry for client `"0"` remains terminal no-success after the runner exits.
- The test should fail if a final `mark_complete()` is reintroduced.

### Purpose

Protect against runner-exit overwrites of terminal CMG status.

## Chunk 2: Add exact `feedback_record=lambda` source guard

### Test-first target

Strengthen `tests/test_phase6_parallel_extraction_subtracks.py`.

### Assertion

- Use `inspect.getsource(CandidateEvaluator.evaluate_iteration)` and assert `"feedback_record=lambda" not in source`.

### Purpose

Lock the source shape expected by the approved plan.

## Chunk 3: Lock finalization ordering for feedback vs registry publication

### Test-first target

Add or adjust tests in `tests/test_phase6_candidate_evaluator.py`.

### Preferred behavior

Keep canonical persistence first, then call `feedback_record` if provided, then publish registry status.

### Required test coverage

- Use an `events: list[str]` trace.
- Patch/spy `artifact_store.write_iteration_results`, `feedback_record`, and `shared_registry.update` (or use a fake registry).
- Success order must be `write`, `feedback`, `publish`.
- Failure test must patch `write_iteration_results` to raise and assert neither feedback nor registry publish occurs.

### CMG invariant

- Do not break the invariant that terminal registry status is published only after persistence succeeds.

### If ordering stays changed

- Preferred production fix: reorder `CandidateEvaluator.finalize_iteration_results` so feedback happens after successful persistence and before `_publish_registry_status`.
- CMG calls with `feedback_record=None`, so this does not reintroduce callback-shaped CMG finalization.
- Add a dedicated test proving the new ordering only if the code path must remain different.
- Include implementation notes with a strong rationale for any deviation.

## Chunk 4: Final verification

Run these commands from the repo root:

```bash
conda run -n gecco_mh pytest tests/test_cmg_runtime.py -q
conda run -n gecco_mh pytest tests/test_phase6_parallel_extraction_subtracks.py -q
conda run -n gecco_mh pytest tests/test_phase6_candidate_evaluator.py -q
conda run -n gecco_mh pytest tests/test_phase6_candidate_generator.py -q
conda run -n gecco_mh pytest tests/test_phase6_run_n_shots_non_cmg.py -q
conda run -n gecco_mh git diff --check
```

## Acceptance checklist

- [ ] Dashboard files are excluded from this follow-up.
- [ ] `run_n_shots()` preserves CMG `complete_no_success` and `had_runnable_model=False`.
- [ ] `CandidateEvaluator.evaluate_iteration` has no `feedback_record=lambda` source pattern.
- [ ] Finalization ordering is covered with failure/success tests.
- [ ] Black formatting passes.
- [ ] `git diff --check` passes.
