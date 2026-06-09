# Phase 6 Review Findings Fix Plan

This is **not a new phase**. It is a bounded remediation plan for review findings discovered after completing `docs/codebase_cleanup_plan/phase_6_service_extraction_completion_plan.md`, and it must stay aligned with:

- `docs/codebase_cleanup_plan.md`
- `docs/codebase_cleanup_plan/README.md`
- `docs/codebase_cleanup_plan/phase_6_parallel_extraction_subtracks.md`
- `docs/codebase_cleanup_plan/phase_6_service_extraction_completion_plan.md`

## Scope and hard constraints

- **Canonical runtime state is DuckDB.**
- **JSON is not runtime source of truth.** Runtime JSON outputs are not the target architecture.
- **Dashboard/provider registry/export/report work is out of scope.** Dashboard-related workspace churn may exist, but do not make dashboard changes a required action here.
- **No compatibility wrappers or hidden fallbacks.** Fix the wiring and ownership directly.
- **No expensive fitting, HBI, or PPC in default unit tests.**
- **Use the `gecco_mh` conda environment** for Python and test commands.
- New Python code must use **type annotations**, **Google-style docstrings** for public functions/classes, and **Black** formatting.

## Review blockers to fix

1. **CMG evaluator finalization is stale.** Finalization currently runs with a no-op registry and feedback-history callbacks, so CMG completion/status/history can lag or remain stale.
2. **Phase 4 judge regression command fails in single-worker tests.** Explicit service wiring is not reflected by the test fixture/coordinator setup.
3. **`tried_param_sets` updates are lost** in extracted evaluator paths, which can let registry/runtime state drift.
4. **Duplicate monolith methods remain in `gecco/run_gecco.py`:** `generate_models`, `generate_models_naive`, `_fit_candidate_model`, `_repair_cmg_candidate`, `_validate_repaired_func_name`, `_finalize_iteration_results`, `_save_review`. Reduce/delete them after service coverage lands.
5. **Type annotation/docstring gaps** remain in new/changed Phase 6 code.
6. **Dashboard files touched in the workspace are out of scope** for this fix plan; mention them only.

## Chunk 0 — Regression baseline and focused characterization

### Tests to write/adjust first

- Add/adjust tests that reproduce:
  - CMG finalization stale status/history behavior.
  - `tried_param_sets` loss in extracted evaluator paths.
  - duplicate-source guard expectations for `gecco/run_gecco.py`.
  - Phase 4 single-worker judge fixture/coordinator failure.

### Implementation steps

- Capture current failures with focused tests only.
- Keep production code unchanged in this chunk.

### Files to touch

- `tests/test_phase6_feedback_coordinator.py`
- `tests/test_phase6_candidate_evaluator.py`
- `tests/test_phase4_orchestrated_judge.py`
- `tests/test_judge_orchestration.py`
- `tests/test_cmg_judge.py`
- `tests/test_cmg_registry.py`
- `tests/test_phase6_parallel_extraction_subtracks.py`

### Files to avoid

- `gecco/run_gecco.py`
- any dashboard/provider/export/report files

### Acceptance criteria

- Tests fail for the expected reasons and document the blockers clearly.

### What not to do

- Do not add fallback code or compatibility shims.
- Do not change runtime architecture yet.

## Chunk 1 — Fix Phase 4 single-worker judge wiring

### Tests to write/adjust first

- Update the Phase 4 regression test so it asserts explicit service wiring in the fixture/coordinator path.

### Implementation steps

- Fix the test fixture and/or coordinator setup so the single-worker regression uses the explicit wired services.
- Keep the fix direct; do not restore any implicit fallback path.

### Files to touch

- `tests/test_phase4_orchestrated_judge.py`
- `tests/test_judge_orchestration.py`
- coordinator/service wiring files used by those tests

### Files to avoid

- `gecco/run_gecco.py` unless the wiring lives there and must be narrowed
- dashboard/provider registry/export/report files

### Acceptance criteria

- The Phase 4 judge regression command passes in single-worker mode.
- No hidden fallback or compatibility branch is introduced.

### What not to do

- Do not widen the change into unrelated orchestration paths.

## Chunk 2 — Fix CMG evaluator finalization callbacks

### Tests to write/adjust first

- Add a regression test proving CMG finalization publishes completion/status/history through the extracted boundary.

### Implementation steps

- Route finalization through the real registry and feedback-history publication path.
- Remove the no-op finalization behavior.
- Ensure CMG completion/status/history are updated before final return.

### Files to touch

- CMG evaluator/coordinator/service files involved in finalization
- `tests/test_cmg_judge.py`
- `tests/test_cmg_registry.py`
- `tests/test_phase6_feedback_coordinator.py`

### Files to avoid

- dashboard/provider/export/report files

### Acceptance criteria

- CMG finalization updates registry/status/history deterministically.
- No stale completion state remains after finalization.

### What not to do

- Do not reintroduce a no-op registry.
- Do not make JSON the authoritative runtime state.

## Chunk 3 — Preserve `tried_param_sets` in extracted evaluator workflows

### Tests to write/adjust first

- Add a regression test that fails when `tried_param_sets` is not synchronized out of the evaluator path.

### Implementation steps

- Carry `tried_param_sets` through the simplest explicit state/callback/return object.
- Update registry/runtime state only after the evaluator state is synchronized.
- Keep the flow explicit and narrow.

### Files to touch

- evaluator/service boundary files for extracted Phase 6 workflows
- `tests/test_phase6_candidate_evaluator.py`
- `tests/test_phase6_candidate_generator.py`

### Files to avoid

- dashboard/provider/export/report files

### Acceptance criteria

- `tried_param_sets` is preserved across extracted evaluator paths.
- Registry/runtime state no longer drifts from evaluator state.

### What not to do

- Do not add hidden backfills or recovery fallbacks.
- Do not update registry state before the evaluator state is known.

## Chunk 4 — Remove or reduce duplicate monolith methods in `run_gecco.py`

### Tests to write/adjust first

- Add source-guard tests or search-based assertions that duplicate monolith behavior is no longer the active implementation path.

### Implementation steps

- Delete or sharply reduce the duplicate methods in `gecco/run_gecco.py`:
  - `generate_models`
  - `generate_models_naive`
  - `_fit_candidate_model`
  - `_repair_cmg_candidate`
  - `_validate_repaired_func_name`
  - `_finalize_iteration_results`
  - `_save_review`
- Leave `run_gecco.py` as orchestration-only wherever service coverage already exists.

### Files to touch

- `gecco/run_gecco.py`
- tests that guard the service coverage and source layout

### Files to avoid

- dashboard/provider/export/report files

### Acceptance criteria

- Duplicate monolith implementations are removed or reduced to thin orchestration only.
- Service-backed paths remain the single source of behavior.

### What not to do

- Do not leave parallel monolith/service logic in place.

## Chunk 5 — Type/docstring/formatting cleanup and scope audit

### Tests to write/adjust first

- Add/adjust lightweight lint-style or review tests only if needed to catch missing annotations/docstrings in changed public code.

### Implementation steps

- Add missing type annotations.
- Add Google-style docstrings to public classes/functions.
- Format with Black.
- Note dashboard file churn as out of scope; do not require a revert.

### Files to touch

- new/changed Phase 6 Python files
- relevant tests if they need updated signatures or fixtures

### Files to avoid

- dashboard/provider/export/report files beyond mention-only audit notes

### Acceptance criteria

- Public API additions/changes are annotated and documented.
- Files are Black-formatted.
- Scope remains limited to review blockers.

### What not to do

- Do not expand scope into dashboard cleanup.
- Do not add default-test-heavy fitting/HBI/PPC work.

## Chunk 6 — Final verification

### Verification commands

```bash
conda run -n gecco_mh pytest tests/test_phase6_parallel_extraction_subtracks.py -q
conda run -n gecco_mh pytest tests/test_phase6_candidate_generator.py tests/test_phase6_candidate_evaluator.py -q
conda run -n gecco_mh pytest tests/test_phase6_feedback_coordinator.py tests/test_phase6_run_n_shots_non_cmg.py -q
conda run -n gecco_mh pytest tests/test_phase4_orchestrated_judge.py tests/test_judge_orchestration.py tests/test_cmg_judge.py tests/test_cmg_registry.py -q
```

### Source searches to run

- Search for duplicate methods in `gecco/run_gecco.py`.
- Search for fallback/compatibility paths that bypass explicit service wiring.
- Search for runtime JSON being treated as authoritative state.

### Acceptance criteria

- All targeted tests pass in `gecco_mh`.
- Duplicate monolith paths are removed or reduced.
- CMG finalization, Phase 4 wiring, and `tried_param_sets` state are all stable.
- No new dashboard work is required.

## Final acceptance checklist

- [ ] Phase 4 single-worker judge regression passes.
- [ ] CMG finalization publishes correct completion/status/history.
- [ ] `tried_param_sets` survives extracted evaluator paths.
- [ ] Duplicate monolith methods are removed or reduced.
- [ ] No hidden fallbacks or compatibility wrappers were added.
- [ ] DuckDB remains the canonical runtime state.
- [ ] JSON remains non-authoritative runtime output only.
- [ ] New/changed Python code has type annotations and Google-style docstrings.
- [ ] Black formatting is clean.
- [ ] Dashboard churn is noted only as out of scope.

## What not to do

- Do not widen this into a new cleanup phase.
- Do not touch dashboard/provider/export/report work as a required deliverable.
- Do not use JSON as runtime source of truth.
- Do not add compatibility wrappers, hidden fallbacks, or expensive default tests.
- Do not change architecture beyond the listed review fixes.
