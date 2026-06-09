# Phase 6 Runner/Helper Split Review Fix Plan

This is **not a new phase**. It is a targeted Phase 6 runner/helper split fix for review findings on `docs/codebase_cleanup_plan/phase_6_runner_orchestration_completion_plan.md`.

This plan must stay aligned with:

- `docs/codebase_cleanup_plan/implementation_guardrails.md`
- `docs/codebase_cleanup_plan/phase_6_runner_orchestration_completion_plan.md`
- `docs/codebase_cleanup_plan/phase_6_parallel_extraction_subtracks.md`
- `docs/codebase_cleanup_plan.md`
- `docs/codebase_cleanup_plan/README.md`
- `docs/phase6_revision_difficulty_deck.html`

## Purpose

This fix closes the review finding repeated in the Phase 6 difficulty deck: the helpers existed, but the runner/callbacks still owned real behavior.

The goal is to finish the ownership split so `GeCCoModelSearch` becomes orchestration only, while the helper services own the actual generation, evaluation, repair, review persistence, and CMG status/history publication responsibilities.

User decisions for this fix:

- Keep the current dashboard edits, but do **not** add any further dashboard work in this fix.
- `GeCCoModelSearch._run_cmg_generator_iteration` and `GeCCoModelSearch._run_cmg_evaluator_iteration` may remain as orchestration helpers only.
- Use **TDD**: tests first, then route production through helper, then delete old/callback path.

This plan is narrow on purpose. It fixes the runner/helper split and the remaining callback ownership leaks. It does **not** restart Phase 6 and does **not** add provider/export/report work.

## Mental model for junior developers

Read this before coding.

### Runner (`GeCCoModelSearch`)

Runner is orchestration only. It may do:

- iteration loop
- CMG vs non-CMG route selection
- lifecycle setup/teardown
- explicit service wiring
- top-level return value assembly

Runner must **not** own the real generation/evaluation logic.

### CandidateGenerator

CandidateGenerator owns:

- generation
- naive generation
- review/fix loop
- candidate artifacts
- candidate publication

### CandidateEvaluator

CandidateEvaluator owns:

- fitting
- repairability checks
- syntax/recovery error feedback
- smoke tests
- CMG repair validation
- best state
- finalization
- status/history publication boundary

### ArtifactStore / DuckDB

ArtifactStore and DuckDB-backed stores own persistence.

- DuckDB is canonical.
- JSON is inspection-only.
- DuckDB / DiagnosticStore write failures must fail fast.

### FeedbackCoordinator

FeedbackCoordinator owns the judge flow.

### Thin route helpers

`GeCCoModelSearch._run_cmg_generator_iteration` and `GeCCoModelSearch._run_cmg_evaluator_iteration` may remain only as thin route helpers.

They may:

- gather and pass explicit inputs
- call helper services
- route CMG vs non-CMG execution

They must **not** contain generation, evaluation, repair, finalization, or status business logic.

### Runner/helper split invariant

Keep this invariant in mind while editing code or tests:

- OK: runner passes `cfg`/`model`/`tokenizer`/`generate_text`/`iteration`/`tag`/`shared_registry` into a service.
- Not OK: runner passes private methods or callbacks that perform generation/evaluation/review persistence/final completion.
- OK: `_run_cmg_generator_iteration` and `_run_cmg_evaluator_iteration` gather inputs and call services.
- Not OK: these helpers implement business logic or own status/finalization.

## Hard forbidden patterns / final-state checks

At final acceptance, confirm these patterns do not remain:

- [ ] `GeCCoModelSearch` does not define `generate_models`
- [ ] `GeCCoModelSearch` does not define `generate_models_naive`
- [ ] `GeCCoModelSearch` does not define `_save_review`
- [ ] `GeCCoModelSearch` does not define `_fit_candidate_model`
- [ ] `GeCCoModelSearch` does not define `_repair_cmg_candidate`
- [ ] `GeCCoModelSearch` does not define `_validate_repaired_func_name`
- [ ] `GeCCoModelSearch` does not define `_finalize_iteration_results`
- [ ] `GeCCoModelSearch` does not define `_is_cmg_repairable_error`
- [ ] `GeCCoModelSearch` does not define `_build_syntax_error_feedback`
- [ ] `GeCCoModelSearch` does not define `_smoke_test_model_return_value`
- [ ] Services do not import or call `GeCCoModelSearch`
- [ ] Services do not depend on private runner methods as collaborators
- [ ] Final CMG completion is not published via runner callback
- [ ] Generator review persistence does not use a generic runner-supplied `save_review` callback
- [ ] Fallback constructors or compatibility wrappers do not remain
- [ ] JSON is not treated as runtime source of truth
- [ ] Default tests do not rely on expensive fitting/HBI/PPC

Special note:

- `GeCCoModelSearch._run_cmg_generator_iteration` and `GeCCoModelSearch._run_cmg_evaluator_iteration` are allowed to exist only as thin orchestration helpers.
- If they contain real business logic, the fix is incomplete.

## Chunks with detailed checklists

## Chunk 0 — Source guards and characterization first

### Goal

Lock in the current behavior with tests before changing production code.

The guard tests you add here may fail initially while production code still contains the ownership leaks. That is expected. The final state must have all guards strict, green, and free of xfails.

### Files likely touched

- `tests/test_phase6_parallel_extraction_subtracks.py`
- `tests/test_cmg_runtime.py`
- `tests/test_cmg_registry.py`

### Files to avoid

- further dashboard files
- provider/export/report files
- unrelated CLI/script files unless a test requires touching them

### Tests to write first

Add or extend source-guard and characterization tests in `tests/test_phase6_parallel_extraction_subtracks.py` or a new focused test file.

Check for:

- [ ] forbidden runner-owned methods still present in `GeCCoModelSearch`
- [ ] `save_review=lambda ...` style plumbing from runner into `CandidateGenerator`
- [ ] final-completion callback / no-op `_update_registry_from_evaluator` patterns in `CandidateEvaluator`
- [ ] `GeCCoModelSearch` still owning real CMG finalization business logic

Use source-level or introspection checks that are narrow enough to avoid matching legitimate service-owned methods with the same names.

### Implementation checklist

- [ ] Keep production code unchanged in this chunk
- [ ] Add guard tests that fail if the runner still owns the listed methods
- [ ] Add guard tests that fail if generator review persistence still depends on a runner callback
- [ ] Add guard tests that fail if evaluator final completion still routes through a runner callback or no-op registry update

### Acceptance criteria

- [ ] Tests describe the current ownership problem clearly
- [ ] Guards are strict enough to catch regressions
- [ ] Guards may fail at first, but there are no xfails in the final state
- [ ] Production behavior is unchanged in this chunk

### What not to do

- [ ] Do not delete code yet
- [ ] Do not add fallback wrappers
- [ ] Do not make the guards so broad that they match service code or comments

## Chunk 1 — CandidateEvaluator owns CMG final status/history publication

### Goal

Move CMG final status/history publication out of the runner callback path and into the evaluator-owned boundary.

### Files likely touched

- `gecco/candidate_evaluation.py`
- `gecco/artifacts.py` only if needed
- `gecco/coordination.py` only if needed
- `tests/test_phase6_candidate_evaluator.py`
- `tests/test_cmg_registry.py`

### Files to avoid

- further dashboard files
- provider/export/report files
- unrelated CLI/script files unless a test requires touching them

### Tests to write first

Add or strengthen tests in `tests/test_phase6_candidate_evaluator.py` and/or `tests/test_cmg_registry.py`.

Use:

- a real `ArtifactStore`
- a real temporary `RunContext`
- a temporary DuckDB registry/state store
- a fake fitting backend
- no expensive fit

Cover these cases:

- [ ] CMG evaluator success writes artifacts through `ArtifactStore` and marks the registry complete with results/history
- [ ] CMG evaluator with all candidates failed / no runnable candidate ends in `complete_no_success`
- [ ] canonical write failure surfaces an exception and the registry is **not** marked complete
- [ ] repair retry status remains deterministic, but final completion happens only after persistence succeeds

### Implementation checklist

- [ ] Remove the no-op `_update_registry_from_evaluator` pattern if present
- [ ] Make evaluator finalization publish through explicit `shared_registry` wiring or a minimal evaluator-owned status helper after `ArtifactStore` succeeds
- [ ] Ensure final status/history publication happens **after** the canonical store write succeeds
- [ ] If using `SharedRegistry` directly, preserve the `tried_param_sets` and `best` state fields expected by registry update
- [ ] Do **not** call back into `GeCCoModelSearch._update_registry` for final completion
- [ ] Keep DuckDB canonical and fail fast on write failure

### Acceptance criteria

- [ ] CMG success updates artifact store and registry deterministically
- [ ] CMG no-success paths produce the correct final status
- [ ] A canonical write failure stops completion and raises
- [ ] Final completion is not owned by the runner callback path

### What not to do

- [ ] Do not mark completion before persistence succeeds
- [ ] Do not reintroduce a no-op registry update
- [ ] Do not make JSON authoritative

## Chunk 2 — Delete residual runner-owned evaluator helpers

### Goal

Remove the remaining runner-owned helper logic that actually belongs to `CandidateEvaluator`.

### Files likely touched

- `gecco/run_gecco.py`
- `gecco/candidate_evaluation.py`
- `tests/test_cmg_runtime.py`
- `tests/test_phase6_candidate_evaluator.py`

### Files to avoid

- further dashboard files
- provider/export/report files
- unrelated CLI/script files unless a test requires touching them

### Tests to write first

Migrate the relevant tests in `tests/test_cmg_runtime.py` so they target `CandidateEvaluator` directly instead of `GeCCoModelSearch` private methods.

Cover service ownership for:

- [ ] repairability checks
- [ ] syntax/recovery feedback construction
- [ ] smoke-test logic for model return values

Also keep or add a source-guard assertion that `GeCCoModelSearch` no longer defines those methods.

### Implementation checklist

- [ ] Delete `GeCCoModelSearch._is_cmg_repairable_error`
- [ ] Delete `GeCCoModelSearch._build_syntax_error_feedback`
- [ ] Delete `GeCCoModelSearch._smoke_test_model_return_value`
- [ ] Keep `_run_cmg_*` route helpers only if they stay thin
- [ ] Ensure `CandidateEvaluator` owns the behavior directly

### Acceptance criteria

- [ ] Tests exercise the evaluator service directly
- [ ] The runner no longer defines the residual evaluator helpers
- [ ] CMG route helpers, if present, are orchestration-only

### What not to do

- [ ] Do not move the helper bodies into another runner private method
- [ ] Do not leave a delegator in place as a “temporary” step
- [ ] Do not hide the logic behind a compatibility wrapper

## Chunk 3 — Fix CMG repair validation context for class-based candidates

### Goal

Make repaired class-based candidates validate with the right context instead of relying on `cfg=None` assumptions.

### Files likely touched

- `gecco/candidate_evaluation.py`
- `tests/test_phase6_candidate_evaluator.py`

### Files to avoid

- further dashboard files
- provider/export/report files
- unrelated CLI/script files unless a test requires touching them

### Tests to write first

Add a regression test in `tests/test_phase6_candidate_evaluator.py` for a class-based repaired candidate.

The test should prove:

- [ ] `_validate_repaired_func_name` gets the needed config / structured params / base-class context
- [ ] class-based repaired candidates validate successfully with real context
- [ ] direct repair remains a direct repair prompt, not naive ideation

### Implementation checklist

- [ ] Thread explicit config or base-class context into the repair validation path
- [ ] Do not pass `cfg=None` blindly when class-based code needs configuration
- [ ] Keep the repair prompt direct and candidate-specific
- [ ] Avoid restoring any two-phase naive ideation flow for repair

### Acceptance criteria

- [ ] Class-based repaired candidates validate correctly
- [ ] Repair prompt semantics remain direct repair
- [ ] The fix does not broaden into unrelated generation changes

### What not to do

- [ ] Do not turn repair into a new candidate ideation step
- [ ] Do not add broad abstractions just to avoid threading context

## Chunk 4 — CandidateGenerator owns review persistence

### Goal

Remove review persistence from runner callback ownership.

### Files likely touched

- `gecco/candidate_generation.py`
- `gecco/run_gecco.py`
- `gecco/artifacts.py` only if needed
- `tests/test_phase6_candidate_generator.py`

### Files to avoid

- further dashboard files
- provider/export/report files
- unrelated CLI/script files unless a test requires touching them

### Tests to write first

Update or add tests in `tests/test_phase6_candidate_generator.py`.

The test should:

- [ ] construct `CandidateGenerator` with a real `ArtifactStore`
- [ ] call the generator method directly
- [ ] avoid passing a `save_review` callback
- [ ] verify the service writes the review through `ArtifactStore`

### Implementation checklist

- [ ] Remove generic `save_review` from generator APIs if it can be removed cleanly
- [ ] Prefer explicit `iteration` and `tag` arguments on the generator method(s) that need review persistence so the service can call `self.artifact_store.write_review(review, iteration=iteration, tag=tag)` directly
- [ ] Make the service call `self.artifact_store.write_review(...)` directly
- [ ] Keep runner wiring limited to `iteration`, `tag`, `cfg`, `model`, `tokenizer`, `generate_text`, and `shared_registry` where needed
- [ ] Do **not** keep a generic `save_review` escape hatch in production
- [ ] Do **not** pass a review-save callback from the runner

### Acceptance criteria

- [ ] Review persistence belongs to the generator boundary
- [ ] No runner callback is required for review writes
- [ ] Existing dashboard edits are left alone and no new dashboard work is added

### What not to do

- [ ] Do not leave a `save_review=lambda ...` bridge in production
- [ ] Do not route review persistence through `GeCCoModelSearch._save_review`
- [ ] Do not add new dashboard files or dashboard features

## Chunk 5 — Final runner/helper split cleanup and verification

### Goal

Remove the remaining runner ownership leaks and verify the split end to end.

### Files likely touched

- `gecco/run_gecco.py`
- `gecco/candidate_generation.py`
- `gecco/candidate_evaluation.py`
- `gecco/coordination.py` only if needed
- `tests/test_phase6_parallel_extraction_subtracks.py`
- `tests/test_phase6_run_n_shots_non_cmg.py`
- `tests/test_cmg_runtime.py`
- `tests/test_cmg_registry.py`
- `tests/test_phase6_candidate_generator.py`
- `tests/test_phase6_candidate_evaluator.py`

### Files to avoid

- further dashboard files
- provider/export/report files
- unrelated CLI/script files unless a test requires touching them

### Tests and verification checklist

- [ ] Make all source guards strict; no xfails remain for the target split
- [ ] Confirm `run_n_shots()` is orchestration only
- [ ] Confirm `_run_cmg_generator_iteration` and `_run_cmg_evaluator_iteration` are orchestration only
- [ ] Confirm tests no longer bind deleted runner private methods
- [ ] Confirm dashboard edits remain untouched, but no additional dashboard files are changed for this fix

### Verification commands

Use `conda run -n gecco_mh` for all Python/test commands.

Run the targeted checks:

```bash
conda run -n gecco_mh pytest tests/test_phase6_candidate_generator.py -q
conda run -n gecco_mh pytest tests/test_phase6_candidate_evaluator.py -q
conda run -n gecco_mh pytest tests/test_phase6_parallel_extraction_subtracks.py -q
conda run -n gecco_mh pytest tests/test_phase6_run_n_shots_non_cmg.py -q
conda run -n gecco_mh pytest tests/test_cmg_runtime.py -q
conda run -n gecco_mh pytest tests/test_cmg_registry.py -q
conda run -n gecco_mh pytest tests/test_cmg_judge.py -q
conda run -n gecco_mh pytest tests/test_phase6_feedback_coordinator.py -q
conda run -n gecco_mh pytest tests/test_phase4_orchestrated_judge.py -q
git diff --check
```

### Implementation checklist

- [ ] Delete any remaining runner-owned helper bodies that survived the earlier chunks
- [ ] Remove stale test bindings to deleted runner private methods
- [ ] Keep the route helpers thin if they remain
- [ ] Verify source guards catch the repeated ownership bug from the difficulty deck

### Acceptance criteria

- [ ] The runner is orchestration only
- [ ] The helpers own the real behavior
- [ ] The callback path is gone
- [ ] The targeted verification commands pass

### What not to do

- [ ] Do not leave “temporary” wrappers in place
- [ ] Do not add new dashboard work
- [ ] Do not expand scope into provider/export/report features

## Final acceptance checklist

- [ ] `GeCCoModelSearch` only orchestrates
- [ ] `CandidateGenerator` owns generation and review persistence
- [ ] `CandidateEvaluator` owns fitting, repair, smoke tests, and finalization boundary work
- [ ] CMG completion/status/history are published only after canonical persistence succeeds
- [ ] Runner callback ownership has been removed
- [ ] DuckDB remains canonical
- [ ] JSON remains inspection-only
- [ ] No expensive default fitting/HBI/PPC was added
- [ ] No new dashboard work was added
- [ ] New/changed Python code is typed, documented with Google docstrings, and Black-formatted

## Reviewer checklist

Use this when reviewing the implementation.

- [ ] Does the runner still contain any real generation, evaluation, repair, or finalization logic?
- [ ] Do the tests prove the new ownership without constructing the monolith for the service cases?
- [ ] Are there any callback bridges left that secretly keep the runner in charge?
- [ ] Is final CMG completion published only after ArtifactStore / DuckDB persistence succeeds?
- [ ] Are DuckDB write failures still fail-fast?
- [ ] Does `CandidateGenerator` write reviews directly instead of through a runner-supplied callback?
- [ ] Are the thin `_run_cmg_*` helpers really thin?
- [ ] Did the fix stay inside Phase 6 and avoid new dashboard/provider/export/report work?

## Common mistakes to avoid

These are the repeated problems from `docs/phase6_revision_difficulty_deck.html`.

- [ ] **Thin wrapper trap:** adding a helper service while the runner/callback still owns the real behavior
- [ ] **Callback trap:** passing runner private methods into services as the main implementation
- [ ] **No-op completion trap:** updating registry/status with a stub or no-op path that looks wired but does nothing
- [ ] **Early completion trap:** marking CMG completion before canonical persistence succeeds
- [ ] **Fallback trap:** keeping compatibility constructors or hidden old-path fallbacks “just in case”
- [ ] **Repair drift trap:** letting class-based repair validation lose its config/base-class context
- [ ] **JSON trap:** treating JSON as runtime truth instead of inspection-only output
- [ ] **Scope creep trap:** adding dashboard, provider, export, or report work during this fix

If you see one of these patterns, stop and fix the ownership boundary before moving on.
