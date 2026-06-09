# Phase 6 Status Ownership Follow-up Fix Plan

This is **not a new phase**. It is a narrow review-follow-up for `docs/codebase_cleanup_plan/phase_6_runner_helper_split_fix_plan.md`. Stay aligned with the implementation guardrails, the Phase 6 runner/helper split plan, DuckDB canonical state, and **no** dashboard/provider/export/report work.

## 1. Purpose

Goal: finish the status-ownership and persistence-order guardrails after review.

Target final state:

- the runner does **not** overwrite evaluator-owned CMG completion
- CMG completion is published only after canonical persistence succeeds
- there is no no-op callback-shaped finalization path
- review files are deterministic and addressed by explicit iteration

## 2. Current findings to fix

Fix these exact points:

- `gecco/run_gecco.py::GeCCoModelSearch.run_n_shots` currently calls `self.shared_registry.mark_complete(self.client_id)` near the end of the run, which can overwrite `complete_no_success`.
- `gecco/coordination.py::SharedRegistry.mark_complete` unconditionally sets status to `complete`.
- `gecco/candidate_evaluation.py::CandidateEvaluator.finalize_iteration_results` persists before publishing, but the existing write-failure test is non-CMG only.
- `gecco/candidate_evaluation.py::CandidateEvaluator.evaluate_iteration` passes `feedback_record=lambda it, results: None`.
- `gecco/artifacts.py::ArtifactStore.write_review` takes `iteration` but writes `iter{len(existing)+1}{tag}.json`.

## 3. Non-goals / do not do

- Do not add dashboard work.
- Do not add provider/export/report work.
- Do not reintroduce runner-owned generation/evaluation/repair/finalization methods.
- Do not use JSON as runtime truth.
- Do not add expensive fitting/HBI/PPC to default tests.
- Do not introduce compatibility wrappers or fallback constructors.
- Do not implement broad lifecycle abstractions unless a test proves they are needed.

## 4. Desired final-state invariants

- `GeCCoModelSearch.run_n_shots()` must not overwrite evaluator-owned `complete_no_success` or `had_runnable_model=False`.
- CMG evaluator status/history publication remains in `CandidateEvaluator`.
- `ArtifactStore.write_iteration_results()` / DuckDB canonical persistence happens before registry `complete` or `complete_no_success` publication.
- If canonical persistence fails in CMG evaluation, an exception surfaces and no terminal registry status is published for that iteration.
- No `feedback_record=lambda ...` no-op hook remains in CMG finalization.
- Review persistence path uses explicit `iteration` and `tag`, not count of existing review files.
- Existing dashboard edits are untouched.

## 5. Chunk 0 — Add failing guards first

### TDD goal and tests

- Add/strengthen source guards in `tests/test_phase6_parallel_extraction_subtracks.py`.
- Guard that `GeCCoModelSearch.run_n_shots` does not call `shared_registry.mark_complete` directly.
- Guard that `CandidateEvaluator.evaluate_iteration` source does not contain `feedback_record=lambda`.
- Guard that `ArtifactStore.write_review` source does not use `len(existing)` or count-based `glob("iter*.json")` naming.

### Implementation checklist

- Tests only.
- No production changes.
- No xfails in final state.

### Acceptance criteria

- Guards fail before fixes and pass after.

## 6. Chunk 1 — Preserve evaluator-owned CMG final status at runner exit

### Tests first

- Add a full or minimal `run_n_shots()` CMG evaluator regression test, preferably in `tests/test_cmg_runtime.py` or `tests/test_phase6_parallel_extraction_subtracks.py`.
- Use a real `SharedRegistry` temp DuckDB and a fake/minimal `GeCCoModelSearch` object with real `run_n_shots` bound.
- Stub generator/evaluator route so the CMG evaluator path publishes `complete_no_success` with `had_runnable_model=False`; then let `run_n_shots` finish.
- Assert final registry snapshot still has `status == "complete_no_success"` and `had_runnable_model is False`.

### Implementation

- Preferred fix: remove the final `self.shared_registry.mark_complete(self.client_id)` call from `GeCCoModelSearch.run_n_shots` if tests show per-iteration services already publish terminal status.
- If a generator-specific lifecycle signal is genuinely needed, add a separate non-status lifecycle/activity update that cannot overwrite evaluator status; only do this if a failing test proves it is needed.
- Do not change `SharedRegistry.mark_complete` to hide the problem unless tests require preserving legacy callers; if changed, it must never overwrite `complete_no_success`/`had_runnable_model=False`.

### Acceptance

- Runner no longer overwrites evaluator-owned terminal statuses.

## 7. Chunk 2 — Add CMG canonical write-failure coverage

### Tests first

- Add `test_candidate_evaluator_cmg_write_failure_does_not_publish_completion` in `tests/test_phase6_candidate_evaluator.py`.
- Use real `RunContext`, real `ArtifactStore`, real `SharedRegistry`, and a fake fitting backend (mock `fit_candidate_model`).
- Seed `shared_registry.set_candidate_models(...)` for one CMG candidate.
- Patch `artifact_store.write_iteration_results` to raise `RuntimeError("duckdb unavailable")`, or use a failing diagnostic store at the artifact boundary if practical.
- Call `CandidateEvaluator.evaluate_iteration(...)` and assert it raises.
- Assert registry does not contain terminal `complete` or `complete_no_success` for the evaluator iteration/client. It may be absent or non-terminal, but must not be terminal.

### Implementation

- Keep `finalize_iteration_results` ordering: write canonical artifacts first, then `_publish_registry_status`.
- Do not catch/suppress the persistence exception.
- Do not publish completion in `finally` blocks.

### Acceptance

- CMG write failure is fail-fast and registry remains uncompleted.

## 8. Chunk 3 — Remove no-op callback-shaped CMG finalization hook

### Tests first

- Strengthen the source guard from Chunk 0.
- Optionally assert `CandidateEvaluator.finalize_iteration_results` can be called from CMG with `feedback_record=None` and still publishes registry status correctly.

### Implementation

- In `CandidateEvaluator.evaluate_iteration`, pass `feedback_record=None` or omit the argument when calling `finalize_iteration_results`.
- Do not add a new no-op function, lambda, or callback wrapper.
- Keep non-CMG `feedback_record=self.feedback.record_iteration` behavior unchanged unless a separate test requires moving it.

### Acceptance

- No `feedback_record=lambda ...` remains; CMG finalization is not callback-shaped.

## 9. Chunk 4 — Make review persistence deterministic by iteration/tag

### Tests first

- Update/add a test in `tests/test_phase6_candidate_generator.py` for a non-sequential iteration, e.g. `iteration=7`, `tag="_abc"`, and optionally pre-seed `reviews/iter1.json`.
- Assert the review file is exactly `reviews/iter7_abc.json` (or the project’s agreed deterministic equivalent using the explicit iteration and tag).
- Assert no count-based `iter2_abc.json` style file is created.

### Implementation

- Change `ArtifactStore.write_review` to write `review_dir / f"iter{iteration}{tag}.json"`.
- Use the explicit `iteration` argument; do not count existing files.
- Keep `inspection_output_enabled` behavior unchanged: return `None` and write nothing when disabled.

### Acceptance

- Review persistence is deterministic and owned by the generator/artifact boundary.

## 10. Chunk 5 — Final verification

Run all commands with `conda run -n gecco_mh`:

- `pytest tests/test_phase6_candidate_generator.py -q`
- `pytest tests/test_phase6_candidate_evaluator.py -q`
- `pytest tests/test_phase6_parallel_extraction_subtracks.py -q`
- `pytest tests/test_phase6_run_n_shots_non_cmg.py -q`
- `pytest tests/test_cmg_runtime.py -q`
- `pytest tests/test_cmg_registry.py -q`
- `pytest tests/test_cmg_judge.py -q`
- `pytest tests/test_phase6_feedback_coordinator.py -q`
- `pytest tests/test_phase4_orchestrated_judge.py -q`
- `git diff --check`

Note: keep Python typed, Google-docstringed for new/changed functions, commented where helpful, and Black-formatted.

## 11. Final acceptance checklist

- [ ] `run_n_shots()` does not overwrite evaluator-owned `complete_no_success`
- [ ] CMG canonical write failure leaves the registry incomplete
- [ ] `feedback_record=lambda ...` is gone from CMG finalization
- [ ] review file naming uses explicit `iteration` and `tag`
- [ ] no dashboard/provider/export/report scope creep

## 12. Reviewer checklist

- Can `run_n_shots` overwrite final evaluator status?
- Does CMG write failure leave the registry incomplete?
- Is the `feedback_record` lambda gone?
- Are reviews deterministic?
- Did the targeted tests pass?
- Were there no dashboard/provider/export/report changes?
