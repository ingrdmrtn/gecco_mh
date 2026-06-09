# Phase 6 Runner Orchestration Completion Plan

This is **not a new phase**. It is the final ownership/deletion pass for Phase 6, and it must stay aligned with:

- `docs/codebase_cleanup_plan/implementation_guardrails.md`
- `docs/codebase_cleanup_plan/phase_6_parallel_extraction_subtracks.md`
- `docs/codebase_cleanup_plan/phase_6_parallel_extraction_subtracks_fix_plan.md`
- `docs/codebase_cleanup_plan/phase_6_parallel_extraction_subtracks_review_findings_fix_plan.md`
- `docs/codebase_cleanup_plan/phase_6_service_extraction_completion_plan.md`
- `docs/codebase_cleanup_plan/phase_6_review_fix_plan.md`
- `docs/phase6_revision_difficulty_deck.html`

## Purpose

Finish the Phase 6 runner/service split so `GeCCoModelSearch` / `gecco/run_gecco.py` becomes orchestration-only. The remaining duplicate runner methods must be removed after the new service paths are protected by tests.

User-confirmed decisions:

- Scope is the full Phase 6 finish.
- Use a **test-then-delete migration style**.
- Duplicate runner methods should be **fully removed**, not left as thin delegators.
- Service interfaces should use **minimal typed dataclasses where helpful**, not broad abstractions.
- Include **source-level guard tests**.

## Hard constraints and non-goals

- **DuckDB is the canonical runtime state.**
- **JSON is not runtime source of truth.** Any remaining JSON must be inspection-only.
- **DuckDB write failures must fail fast.** Do not log-and-continue after a canonical runtime write fails.
- **No dashboard work.**
- **No provider registry/export/report work.**
- **No compatibility wrappers or hidden fallbacks.**
- **No expensive fitting, HBI, or PPC in default unit tests.**
- Use `conda run -n gecco_mh ...` for Python/test commands.
- New Python code must use **type annotations**, **Google-style docstrings** for public functions/classes, and **Black** formatting.

## Implementation clarifications

These clarifications are part of the existing Phase 6 finish. They are not permission to broaden scope.

- Source guards should be precise. Prefer AST/introspection checks scoped to `GeCCoModelSearch` over broad string checks that also match docs, comments, or service methods with legitimate names.
- It is acceptable for `CandidateGenerator` to define service-owned generation methods such as `generate_models`; it is not acceptable for `GeCCoModelSearch` to keep duplicate generation methods.
- Review persistence must not be routed through a runner callback such as `GeCCoModelSearch._save_review`. Persist reviews through the service boundary and `ArtifactStore`.
- CMG repair must use a direct repair prompt. Do not preserve or reintroduce two-phase naive ideation for repair, because repair should fix the assigned candidate rather than ideate a new one.
- Registry/status updates must be ordered after evaluator and canonical-store state are known. Do not mark completion before persistence succeeds.
- Each deletion chunk should include a quick reference audit for the runner method being removed, including tests.

## Target ownership

After completion, ownership should look like this:

- `GeCCoModelSearch`: orchestration only; iteration loop, CMG/non-CMG route selection, high-level lifecycle, service wiring, final top-level return.
- `CandidateGenerator`: standard generation, naive generation, correction/review loop, candidate artifact persistence, candidate publication.
- `CandidateEvaluator`: candidate fitting, validation/recovery handling, CMG repair loop, best-model state updates, finalization, status/history update boundary.
- `ArtifactStore` / DuckDB stores: persistence boundary; DuckDB canonical, JSON inspection-only if retained.
- `FeedbackCoordinator`: orchestrated judge flow with explicit collaborators only.
- `DistributedCoordinator` / `SharedRegistry`: distributed state/status coordination.

## Ownership flow diagram

```text
GeCCoModelSearch
  ├── CandidateGenerator
  │     ├── candidate artifacts
  │     └── publication / review persistence
  ├── CandidateEvaluator
  │     ├── fitting backend
  │     ├── CMG repair loop
  │     ├── best-model state
  │     └── finalization / status updates
  ├── FeedbackCoordinator
  │     └── explicit judge collaborators
  └── ArtifactStore / DuckDB
        └── canonical runtime state
```

## TDD rule for this plan

Follow this order for every chunk:

1. Write characterization / guard tests first.
2. Write direct contract tests for the extracted helper or boundary.
3. Route production code through the new owner.
4. Delete the old duplicate runner path only after the tests prove the new path is active.

Do not start deletions before the tests that protect the replacement path exist.

## Forbidden patterns

These must not remain at completion:

- `GeCCoModelSearch` still defining core generation/evaluation/repair/finalization methods.
- `run_gecco.py` still containing the duplicate methods:
  - `generate_models`
  - `generate_models_naive`
  - `_save_review`
  - `_fit_candidate_model`
  - `_repair_cmg_candidate`
  - `_validate_repaired_func_name`
  - `_finalize_iteration_results`
- Helper services calling back into `GeCCoModelSearch.generate_models` or `GeCCoModelSearch.generate_models_naive`.
- Fallback collaborator construction such as `getattr(self, ..., None) or CandidateGenerator(...)`.
- Hidden compatibility wrappers or “temporary” delegators left in place.
- JSON used as runtime source of truth.
- Default tests that perform expensive fitting/HBI/PPC.
- Monolith as an injected substitute for an external backend.

## Chunk 0 — Baseline audit and source guards

### Goal

Freeze the current Phase 6 behavior and add/adjust guard tests that describe what must disappear by the end.

### Dependencies

- Existing Phase 6 tests and docs.

### Tests to write first

- Add or adjust source-guard tests in `tests/test_phase6_parallel_extraction_subtracks.py` or a new focused file.
- Guard against:
  - service code importing/calling monolith private methods for core behavior;
  - `run_gecco.py` still defining the duplicate methods after later deletion chunks;
  - fallback constructors such as `getattr(self, ..., None) or CandidateGenerator(...)`;
  - helper calls back to `GeCCoModelSearch.generate_models` or `_fit_candidate_model`.
- Keep guards scoped so they do not reject legitimate service-owned methods with the same names.

### Implementation steps

- Run focused existing tests to establish the current baseline.
- Keep production code unchanged in this chunk.
- If a guard must be temporarily soft, use a TODO or xfail only as an intermediate step, and plan to make it strict in the final deletion chunk.
- Prefer AST/introspection guards over brittle whole-file string assertions where practical.

### Files likely touched

- `tests/test_phase6_parallel_extraction_subtracks.py`
- possibly a new focused source-guard test file under `tests/`

### Files to avoid

- `gecco/run_gecco.py`
- `gecco/candidate_generation.py`
- `gecco/candidate_evaluation.py`
- any dashboard/provider/export/report files

### Deletion tasks

- None yet.

### Acceptance criteria

- Baseline behavior is documented by tests.
- Guard tests exist for the forbidden patterns.

### What not to do

- Do not refactor production code yet.
- Do not write brittle tests that only prove a mock was called.

## Chunk 1 — CandidateGenerator owns all generation behavior

### Goal

Move all generation behavior out of the runner and into `CandidateGenerator`.

### Dependencies

- Chunk 0 source guards.
- Existing `CandidateGenerator` implementation.

### Tests to write first

Strengthen direct tests in `tests/test_phase6_candidate_generator.py` and related Phase 6 tests to cover:

- standard generation;
- naive generation;
- review/fix persistence;
- parse/validation behavior;
- candidate artifact write;
- registry publication for CMG;
- non-CMG generation result shape.

Use:

- a fake deterministic LLM text-generation backend;
- a real `ArtifactStore` with a temp directory;
- no `GeCCoModelSearch` construction for service tests.

### Implementation steps

- Ensure CMG and non-CMG routes call `CandidateGenerator` methods with explicit inputs.
- Remove callbacks from helper paths that point back to `GeCCoModelSearch.generate_models` or `generate_models_naive`.
- Remove review persistence callbacks that point back to `GeCCoModelSearch._save_review`; route review writes through `ArtifactStore` or an explicit service-owned boundary.
- Keep generation inputs/outputs narrow and explicit.

### Files likely touched

- `tests/test_phase6_candidate_generator.py`
- `tests/test_phase6_parallel_extraction_subtracks.py`
- `gecco/run_gecco.py`
- `gecco/candidate_generation.py`

### Files to avoid

- `gecco/candidate_evaluation.py` unless a test needs a shared fixture update
- dashboard/provider/export/report files

### Deletion tasks

- Remove `GeCCoModelSearch.generate_models`.
- Remove `GeCCoModelSearch.generate_models_naive`.
- Remove `GeCCoModelSearch._save_review`.

### Acceptance criteria

- Generation behavior is fully covered through the helper service.
- The runner no longer owns generation logic.
- No monolith-private generation callback remains in service wiring.

### What not to do

- Do not keep the runner as a thin generation delegator.
- Do not introduce a broad abstraction just to hide method calls.
- Do not keep a generic review-save callback if it only preserves the old runner ownership path.

## Chunk 2 — CandidateEvaluator owns CMG repair behavior

### Goal

Move CMG repair/validation behavior out of the runner and into `CandidateEvaluator`.

### Dependencies

- Chunk 1.
- Real or fake registry/store collaborators available for tests.

### Tests to write first

Add direct CMG repair-path tests for `CandidateEvaluator.evaluate_iteration` and/or `_repair_candidate` using:

- fake registry;
- fake text generator;
- fake fitting backend;
- real `ArtifactStore`.

Cover:

- validation/fitting failure -> repair prompt -> repaired candidate parsed -> AST/function-name validation -> `registry.update_candidate_model` -> refit -> final result;
- repair failure or invalid function name keeps the original candidate.
- naive-enabled evaluator clients still use the direct repair prompt, not two-phase naive ideation.

### Implementation steps

- Ensure repair uses low-level `generate_text` backend plus prompt builder.
- Do **not** use `self.generate_models` on the runner for repair.
- Do **not** use naive ideation for CMG repair.
- Keep validation boundaries explicit and local to the evaluator.

### Files likely touched

- `tests/test_phase6_candidate_evaluator.py`
- `tests/test_phase6_parallel_extraction_subtracks.py`
- `gecco/run_gecco.py`
- `gecco/candidate_evaluation.py`

### Files to avoid

- dashboard/provider/export/report files

### Deletion tasks

- Remove `GeCCoModelSearch._repair_cmg_candidate`.
- Remove `GeCCoModelSearch._validate_repaired_func_name`.

### Acceptance criteria

- CMG repair is owned by the evaluator service.
- The runner no longer contains repair-specific business logic.

### What not to do

- Do not keep a repair path that delegates back into the monolith.
- Do not hide invalid repairs behind a fallback retry loop.
- Do not turn repair into new candidate ideation.

## Chunk 3 — CandidateEvaluator owns fitting, best state, tried_param_sets, and finalization

### Goal

Move fitting and finalization ownership fully into `CandidateEvaluator`.

### Dependencies

- Chunk 2.

### Tests to write first

Strengthen tests in:

- `tests/test_phase6_candidate_evaluator.py`
- `tests/test_phase6_parallel_extraction_subtracks.py`
- CMG registry/status tests as needed

Cover:

- successful fit;
- validation error;
- fit error;
- recovery simulation error;
- best-model state update;
- finalization writes through `ArtifactStore`;
- canonical DuckDB/`DiagnosticStore` write failure is surfaced as a failure, not logged and ignored;
- status/history publication;
- `tried_param_sets` preservation.

Use a fake fitting backend or monkeypatch the expensive fit function. Do not add expensive fitting/HBI/PPC to default unit tests.

### Implementation steps

- Keep the evaluator as the owner of fit/finalize boundaries.
- Preserve `tried_param_sets` through explicit state or request/response objects.
- Ensure finalization only reports success after canonical runtime writes have succeeded.
- Introduce minimal typed dataclasses only if they simplify the boundary, for example:
  - `EvaluationRequest`
  - `EvaluationOutcome`
  - `RepairRequest`
- Keep them small and typed; do not overdesign.

### Files likely touched

- `tests/test_phase6_candidate_evaluator.py`
- `tests/test_cmg_registry.py`
- `tests/test_cmg_runtime.py`
- `gecco/run_gecco.py`
- `gecco/candidate_evaluation.py`

### Files to avoid

- dashboard/provider/export/report files

### Deletion tasks

- Remove `GeCCoModelSearch._fit_candidate_model`.
- Remove `GeCCoModelSearch._finalize_iteration_results`.

### Acceptance criteria

- Best-model and finalization state lives in the evaluator boundary, not the runner.
- `tried_param_sets` is not lost.
- DuckDB/store writes happen through the extracted boundary.

### What not to do

- Do not update registry state before evaluator state is known.
- Do not update registry state before canonical persistence succeeds.
- Do not preserve JSON as runtime truth.

## Chunk 4 — `run_n_shots()` becomes orchestration-only

### Goal

Reduce `run_n_shots()` to coordination, route selection, wiring, and top-level return only.

### Dependencies

- Chunks 1–3.

### Tests to write first

Strengthen orchestration tests in:

- `tests/test_phase6_run_n_shots_non_cmg.py`
- `tests/test_cmg_runtime.py`

Prove that:

- the CMG generator path delegates to `CandidateGenerator.generate_iteration`;
- the CMG evaluator path delegates to `CandidateEvaluator.evaluate_iteration`;
- the non-CMG loop delegates generation/evaluation/finalization to helper services;
- `run_n_shots()` only handles retry decision, feedback, iteration control, and the final top-level return.

### Implementation steps

- Shrink inline logic in `run_n_shots()`.
- Keep only orchestration concerns in the runner.
- Keep route selection and service wiring explicit.
- Do not introduce a new workflow engine or broad abstraction.

Allowed responsibilities for `run_n_shots()` after this chunk:

- iteration loop and resume start point;
- CMG vs non-CMG route selection;
- high-level feedback retrieval/coordination;
- service invocation and state synchronization;
- final top-level return and lifecycle completion.

Disallowed responsibilities for `run_n_shots()` after this chunk:

- parsing model responses;
- fitting candidate models;
- repairing candidate code;
- writing iteration results directly;
- manually publishing evaluator results that belong to the evaluator/status boundary.

### Files likely touched

- `gecco/run_gecco.py`
- `tests/test_phase6_run_n_shots_non_cmg.py`
- `tests/test_cmg_runtime.py`

### Files to avoid

- dashboard/provider/export/report files

### Deletion tasks

- Delete any remaining runner-owned core generation/evaluation/finalization code paths already covered by the service tests.

### Acceptance criteria

- `run_n_shots()` reads like orchestration.
- Helper services own the actual work.
- No hidden fallback construction remains.

### What not to do

- Do not reintroduce monolith-owned business logic in the loop.
- Do not add a wrapper layer just to preserve old call sites.

## Chunk 5 — Feedback, status, and runtime state stay explicit and canonical

### Goal

Finish the review blockers around coordinated judge flow, status updates, and canonical state.

### Dependencies

- Chunks 1–4.

### Tests to write first

Use or extend:

- `tests/test_phase6_feedback_coordinator.py`
- `tests/test_phase4_orchestrated_judge.py`
- `tests/test_cmg_registry.py`
- `tests/test_cmg_judge.py`

Cover:

- stale CMG finalization callbacks are fixed;
- Phase 4 single-worker judge wiring uses explicit collaborators;
- `tried_param_sets` is preserved if not already covered;
- feedback/status updates are published through the real boundary;
- status outcomes are deterministic for success, no-success completion, repair failure, candidate wait timeout, and canonical-store write failure.

### Implementation steps

- Ensure `FeedbackCoordinator` uses explicit orchestrated judge collaborators only.
- Remove any fallback to `gecco.run_gecco` imports.
- Keep DuckDB / registry state as the source of truth.
- Keep registry completion/status updates after evaluator and canonical-store state are known.
- Treat JSON as inspection-only if it still exists.

### Files likely touched

- `gecco/feedback_coordinator.py`
- `gecco/construct_feedback/orchestrated.py` (optional, if the orchestrated judge helper itself needs wiring adjustments)
- `tests/test_phase6_feedback_coordinator.py`
- `tests/test_phase4_orchestrated_judge.py`
- `tests/test_cmg_registry.py`
- `tests/test_cmg_judge.py`

### Files to avoid

- dashboard/provider/export/report files

### Deletion tasks

- Remove stale finalization callbacks or no-op publication paths.

### Acceptance criteria

- Feedback/status publication is explicit and deterministic.
- No fallback import path remains.
- Canonical-store failures cannot be reported as successful completion.

### What not to do

- Do not make JSON the runtime authority.
- Do not broaden the coordinator into a new architecture layer.
- Do not add new dashboard, provider registry, export, or report behavior while fixing status publication.

## Chunk 6 — Final deletion gate and verification

### Goal

Make the source guards strict, confirm the duplicate runner methods are gone, and verify helpers do not call back into the monolith.

### Dependencies

- Chunks 1–5.

### Tests to write first

- Turn all temporary source guards strict.
- Verify no forbidden patterns remain.
- Verify no hidden fallbacks or compatibility wrappers remain.

### Implementation steps

- Remove the duplicate runner method definitions.
- Confirm the helpers only depend on explicit collaborators and external backends.
- Confirm tests no longer bind or call the deleted runner private methods.
- Run the targeted test set, then broader relevant tests if time permits.

### Files likely touched

- `tests/test_phase6_parallel_extraction_subtracks.py`
- `tests/test_phase6_candidate_generator.py`
- `tests/test_phase6_candidate_evaluator.py`
- `tests/test_phase6_run_n_shots_non_cmg.py`
- `tests/test_cmg_runtime.py`
- `tests/test_cmg_registry.py`
- `tests/test_cmg_judge.py`
- `tests/test_phase6_feedback_coordinator.py`
- `tests/test_phase4_orchestrated_judge.py`
- `gecco/run_gecco.py`

### Files to avoid

- dashboard/provider/export/report files

### Deletion tasks

- Remove any remaining duplicate runner implementations and wrapper-like paths.

### Acceptance criteria

- The duplicate runner methods are gone.
- The source guards pass.
- No helper imports or calls back into monolith private methods remain.
- Existing tests have been migrated away from deleted runner methods.

### What not to do

- Do not leave any xfails in the final state.
- Do not leave the monolith as a hidden dependency.

## Implementation notes for junior developers

- **Use fake external backends, not the monolith, for tests.** Faking LLM text generation, fitting, network APIs, or long diagnostics is fine. Passing `GeCCoModelSearch` methods into a service is not.
- **Keep service interfaces small.** If a typed dataclass helps, make it minimal and explicit. Prefer a few fields that clearly describe the request/response boundary over a broad “context” object.
- **Keep runtime state canonical in DuckDB.** If JSON still exists, treat it as inspection-only. Do not write tests that depend on JSON for correctness.
- **Fail fast on canonical runtime write errors.** Do not hide DuckDB/`DiagnosticStore` failures behind logging-only behavior.
- **Keep CMG repair direct.** Repair prompts should fix the assigned candidate; do not route repair through naive ideation.
- **Follow Black, type hints, and Google docstrings** for any new/changed public Python code.
- **Update source guards as you go.** It is okay to start with a temporary guard that documents the current shape, but the final chunk must make it strict and delete the forbidden code.
- **If a test needs a collaborator, inject the real collaborator boundary or a deterministic fake.** Do not inject the old monolith as a service dependency.

## Verification commands

Use these commands during and after implementation:

```bash
conda run -n gecco_mh pytest tests/test_phase6_candidate_generator.py -q
conda run -n gecco_mh pytest tests/test_phase6_candidate_evaluator.py -q
conda run -n gecco_mh pytest tests/test_phase6_parallel_extraction_subtracks.py -q
conda run -n gecco_mh pytest tests/test_phase6_run_n_shots_non_cmg.py -q
conda run -n gecco_mh pytest tests/test_cmg_runtime.py tests/test_cmg_registry.py tests/test_cmg_judge.py -q
conda run -n gecco_mh pytest tests/test_phase6_feedback_coordinator.py tests/test_phase4_orchestrated_judge.py -q
```

If time permits, run broader related tests as well, but do **not** require expensive diagnostics by default.

## Final acceptance checklist

- [ ] `GeCCoModelSearch` is orchestration-only.
- [ ] `CandidateGenerator` owns generation, naive generation, review persistence, and candidate publication.
- [ ] `CandidateEvaluator` owns fitting, CMG repair, best-model state, finalization, and status/history boundaries.
- [ ] `run_n_shots()` no longer contains core business logic.
- [ ] Duplicate runner methods are deleted, not delegated.
- [ ] No helper calls back into `GeCCoModelSearch` for core behavior.
- [ ] No fallback constructors or hidden compatibility wrappers remain.
- [ ] DuckDB is still the canonical runtime state.
- [ ] DuckDB/`DiagnosticStore` write failures fail fast.
- [ ] JSON is inspection-only, not runtime truth.
- [ ] CMG repair uses direct repair prompting, including for naive-enabled clients.
- [ ] Source-guard tests pass and are strict.
- [ ] Default tests avoid expensive fitting/HBI/PPC.
- [ ] New/changed public Python code has type hints, Google docstrings, and Black formatting.
