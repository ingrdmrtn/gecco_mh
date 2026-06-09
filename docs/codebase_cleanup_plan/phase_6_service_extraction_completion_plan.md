# Phase 6 Service Extraction Completion Plan

## Purpose

This is the final Phase 6 completion plan. It is **not** a new phase.

Its job is to close the remaining review gaps after the parallel extraction subtracks work. The scope is strictly the completion of service extraction for `GeCCoModelSearch` / `gecco/run_gecco.py`, plus the related tests and boundary cleanup needed to finish Phase 6.

## Source scope and non-goals

Source documents in scope:

- `docs/codebase_cleanup_plan.md`
- `docs/codebase_cleanup_plan/README.md`
- `docs/codebase_cleanup_plan/phase_6_parallel_extraction_subtracks.md`
- `docs/codebase_cleanup_plan/phase_6_parallel_extraction_subtracks_review_findings_fix_plan.md`

Hard constraints:

- DuckDB is the canonical runtime state store.
- JSON is not the runtime source of truth.
- Runtime JSON outputs are not part of the target architecture.
- No dashboard work.
- No provider registry/export/report work.
- No compatibility wrappers or hidden fallbacks.
- No expensive fitting, HBI, or PPC in default unit tests.
- Use the `gecco_mh` conda environment for Python/test commands.
- New Python code must use type annotations, Google-style docstrings for public functions/classes, and Black formatting.

## Current gaps to close

The code still has the following Phase 6 gaps:

- `gecco/run_gecco.py` still contains duplicate or monolith-owned logic for:
  - `generate_models`
  - `generate_models_naive`
  - `_fit_candidate_model`
  - `_finalize_iteration_results`
  - inline non-CMG loop logic inside `run_n_shots()`
  - best-model tracking and JSON writing for best-BIC output
  - syntax retry / repair handling
  - fallback collaborator construction through `getattr(self, ..., None) or Constructor(...)`
- `gecco/artifacts.py` still writes JSON routinely in runtime paths.
- `CandidateGenerator` and `CandidateEvaluator` exist, but the monolith still duplicates parts of their logic instead of fully delegating.
- Tests in `tests/test_phase6_parallel_extraction_subtracks.py` are still too mock-heavy in critical places.
- Dedicated service tests still need stronger coverage, especially for non-CMG routing and JSON non-canonicity.

## Target design

The target ownership is simple and explicit:

- `GeCCoModelSearch` becomes an orchestration shell only.
- `RunContext` owns paths and tempdirs.
- `ArtifactStore` owns the persistence boundary and writes DuckDB canonical runtime data.
  - If any JSON remains, it must be explicit inspection-only output.
  - JSON must not be used for runtime correctness.
- `CandidateGenerator` owns standard generation, naive generation, correction/review loop, and candidate persistence.
- `CandidateEvaluator` owns fitting, validation, recovery error handling, best-model state updates, repair loop, finalization, and DuckDB writes.
- `FeedbackCoordinator` owns explicit judge orchestration with injected collaborators only.
- `DistributedCoordinator` owns registry/status coordination.
- `run_gecco.py` must not hide fallback construction for collaborators.

## TDD workflow

Follow this order strictly:

1. Write characterization tests for the remaining current behavior.
2. Write contract tests for the extracted services.
3. Delete monolith-owned duplicate code only after the replacement path is protected by tests.

For Phase 6, do not begin deletions before the tests that prove the new boundary exist.

## Vertical implementation chunks

### Chunk 0: Safety baseline and characterization

**Goal:** Freeze the remaining behavior before changing production code.

**Dependencies:** none.

**Tests to write first:**

- Extend `tests/test_phase6_parallel_extraction_subtracks.py` with characterization coverage for current non-CMG boundary behavior.
- Add a characterization test for current JSON-writing behavior so the later removal is intentional.
- Add a test that captures the current fallback constructor behavior before replacing it.

**Implementation steps:**

- Do not change production code in this chunk.
- Keep tests lightweight and narrow.
- Prefer direct assertions on observable behavior over broad full-stack mocks.

**Files to touch:**

- `tests/test_phase6_parallel_extraction_subtracks.py`

**Files to avoid:**

- `gecco/run_gecco.py`
- `gecco/artifacts.py`
- any dashboard files

**Acceptance criteria:**

- The tests clearly document what the monolith currently still owns.
- JSON-writing behavior is captured before removal.
- No production code changes are needed yet.

**What not to do:**

- Do not refactor while characterizing.
- Do not over-mock the entire system.
- Do not touch dashboard code.

### Chunk 1: Explicit service wiring, no fallback constructors

**Goal:** Remove hidden collaborator construction.

**Dependencies:** Chunk 0.

**Tests to write first:**

- Add tests proving missing required collaborators fail fast.
- Add tests proving no `getattr(..., None) or Constructor(...)` fallback path is used.
- Keep the test targeted to wiring and construction, not fitting behavior.

**Implementation steps:**

- Construct services once in `__init__`.
- If needed, add small private accessors that raise a clear `RuntimeError` when a collaborator is missing.
- Replace fallback patterns in CMG, non-CMG, feedback, and finalize paths.

**Files to touch:**

- `gecco/run_gecco.py`
- `tests/test_phase6_parallel_extraction_subtracks.py`

**Files to avoid:**

- dashboard files
- provider registry files
- export/report files

**Deletion tasks:**

- Remove fallback instantiations.

**Acceptance criteria:**

- Missing collaborators fail fast with a clear error.
- No hidden constructor fallback remains in the Phase 6 path.

**What not to do:**

- Do not add a compatibility wrapper.
- Do not keep a hidden default for safety.

### Chunk 2: Move generation ownership fully into `CandidateGenerator`

**Goal:** Make generation a real service boundary.

**Dependencies:** Chunk 1.

**Tests to write first:**

- Extend `tests/test_phase6_candidate_generator.py`.
- Add contract tests for standard generation.
- Add contract tests for naive generation fallback.
- Add contract tests for review/fix persistence.
- Add a test proving the monolith no longer needs private generation methods.

**Implementation steps:**

- Make CMG and non-CMG code use `CandidateGenerator` APIs.
- Avoid passing `GeCCoModelSearch.generate_models` or `generate_models_naive` callbacks around.
- Keep the generator focused on explicit inputs and outputs.

**Files to touch:**

- `gecco/run_gecco.py`
- `gecco/candidate_generation.py` or the existing generator module used by the project
- `tests/test_phase6_candidate_generator.py` (extend existing file)

**Files to avoid:**

- `gecco/artifacts.py` except for any minimal store wiring needed by the generator
- dashboard files

**Deletion tasks:**

- Remove or reduce `GeCCoModelSearch.generate_models`.
- Remove or reduce `GeCCoModelSearch.generate_models_naive`.
- Remove or reduce `_save_review` once callers are migrated.

**Acceptance criteria:**

- Generator behavior is testable without monolith private methods.
- The monolith no longer owns generation logic directly.

**What not to do:**

- Do not keep the generator as a pass-through.
- Do not add extra layers just to preserve old structure.

### Chunk 3: Move evaluation, best-model state, repair, and finalization ownership into `CandidateEvaluator`

**Goal:** Make evaluation/finalization a service boundary.

**Dependencies:** Chunk 2.

**Tests to write first:**

- Extend `tests/test_phase6_candidate_evaluator.py`.
- Add tests for successful fit.
- Add tests for validation/recovery errors.
- Add tests for best-model updates.
- Add tests for finalization writes to DuckDB.
- Add a CMG repair-loop test using fakes.

**Implementation steps:**

- Introduce the smallest possible explicit state object or callback needed for best-model updates.
- Move best-model file handling into `ArtifactStore` only if it must remain, and keep it non-canonical.
- Make the evaluator own syntax retry and finalization contracts.
- Keep default tests free of expensive fitting.

**Files to touch:**

- `gecco/run_gecco.py`
- `gecco/candidate_evaluation.py` or the existing evaluator module used by the project
- `gecco/artifacts.py`
- `tests/test_phase6_candidate_evaluator.py` (extend existing file)

**Files to avoid:**

- dashboard files
- provider registry files
- export/report files

**Deletion tasks:**

- Remove or reduce `_fit_candidate_model`.
- Remove or reduce `_finalize_iteration_results`.
- Remove or reduce `_repair_cmg_candidate`.
- Remove or reduce `_validate_repaired_func_name` from the monolith after service coverage exists.

**Acceptance criteria:**

- Evaluation and finalization are testable as a standalone service.
- Best-model state is no longer owned by the monolith.

**What not to do:**

- Do not bury repair-loop details in the monolith.
- Do not add expensive default fitting to unit tests.

### Chunk 4: Replace inline non-CMG `run_n_shots()` loop with service workflow

**Goal:** Make `run_n_shots()` a thin orchestrator.

**Dependencies:** Chunks 1, 2, and 3.

**Tests to write first:**

- Create `tests/test_phase6_run_n_shots_non_cmg.py`.
- Use real services with lightweight fakes/stubs for LLM and fitting.
- Assert monolith methods are not called for non-CMG work.
- Assert finalization happens through evaluator/store.
- Cover syntax retry once.

**Implementation steps:**

- Extract a small service-level method only if needed, such as `CandidateEvaluator.run_non_cmg_iteration(...)`.
- If a new helper service is necessary, keep it tiny and internal to Phase 6.
- Keep `run_n_shots()` as loop + feedback + branch selection only.

**Files to touch:**

- `gecco/run_gecco.py`
- `gecco/candidate_evaluation.py` or a small iteration helper if needed
- `tests/test_phase6_run_n_shots_non_cmg.py`

**Files to avoid:**

- dashboard files
- provider registry files
- export/report files

**Deletion tasks:**

- Remove inline generation/evaluation/best-model/finalization blocks from `run_n_shots()`.

**Acceptance criteria:**

- `run_n_shots()` no longer owns substantial non-CMG iteration logic.
- The route is visibly service-based.

**What not to do:**

- Do not create a second parallel implementation.
- Do not keep old monolith blocks as a hidden fallback.

### Chunk 5: Make JSON explicitly non-canonical or remove it from runtime path

**Goal:** Stop treating JSON as runtime state.

**Dependencies:** Chunks 3 and 4.

**Tests to write first:**

- Replace JSON-existence assertions with DuckDB assertions where possible.
- Add a test proving runtime succeeds when inspection JSON is disabled or not written.
- Add an `ArtifactStore` contract test proving DuckDB is the source of truth.

**Implementation steps:**

- Add an explicit optional inspection-output flag only if needed.
- Default inspection output must be off for runtime correctness.
- Write DuckDB first.
- Never read JSON for runtime state.
- Remove routine best-BIC JSON write paths or mark them as clearly inspection-only with tests.

**Files to touch:**

- `gecco/artifacts.py`
- `gecco/run_gecco.py`
- `tests/test_phase6_parallel_extraction_subtracks.py`
- `tests/test_phase6_candidate_evaluator.py`

**Files to avoid:**

- dashboard files
- provider registry files
- export/report files

**Deletion tasks:**

- Remove routine JSON write paths.
- Remove tests that require JSON files as runtime truth.

**Acceptance criteria:**

- Runtime correctness does not depend on JSON files.
- Any remaining JSON output is explicitly non-canonical inspection output.

**What not to do:**

- Do not make JSON a new source of truth.
- Do not leave silent duplicate writes just for convenience.

### Chunk 6: FeedbackCoordinator and DistributedCoordinator boundary hardening

**Goal:** Finish explicit dependency injection and remove hidden coordinator construction.

**Dependencies:** Chunks 1 and 4.

**Tests to write first:**

- Create `tests/test_phase6_feedback_coordinator.py`.
- Keep an import guard proving there is no `gecco.run_gecco` fallback.
- Add a minimal distributed coordinator test only if this boundary is still involved in the Phase 6 path.

**Implementation steps:**

- Keep explicit dependency injection.
- Ensure `run_n_shots()` uses the existing coordinator attribute only.
- Remove any remaining hidden coordinator construction.

**Files to touch:**

- `gecco/feedback_coordinator.py`
- `gecco/run_gecco.py`
- `gecco/coordination.py` only if the distributed boundary is still part of the extraction path
- `gecco/construct_feedback/...` only if required by the existing orchestrated judge adapter, and not for legacy/manual judge work
- `tests/test_phase6_feedback_coordinator.py`

**Files to avoid:**

- dashboard files
- provider registry files
- export/report files

**Deletion tasks:**

- Remove hidden coordinator construction.

**Acceptance criteria:**

- The coordinator depends only on injected collaborators.
- No monolith fallback path remains.

**What not to do:**

- Do not widen this into provider registry or dashboard work.

### Chunk 7: Delete monolith duplicates and tighten regression suite

**Goal:** Remove leftover duplicate logic and make the service tests the real protection.

**Dependencies:** Chunks 2 through 6.

**Tests to write first:**

- Add simple no-duplicate guard tests or source inspections only where useful.
- Keep these checks focused and easy for a junior developer to understand.
- Run focused tests before deleting the final duplicate blocks.

**Implementation steps:**

- Delete duplicate monolith methods and blocks after tests pass.
- Update imports and call sites.
- Make sure `run_gecco.py` reads like orchestration shell code.

**Files to touch:**

- `gecco/run_gecco.py`
- `tests/test_phase6_parallel_extraction_subtracks.py`
- the new service test files from chunks 2 through 6

**Files to avoid:**

- dashboard files
- provider registry files
- export/report files

**Acceptance criteria:**

- The monolith no longer duplicates the same responsibilities.
- The extracted services carry the real logic.
- `run_gecco.py` is clearly orchestration only.

**What not to do:**

- Do not leave duplicate methods behind “just in case.”
- Do not broaden this into new architecture work.

## Verification commands

Run these focused commands in the `gecco_mh` environment:

```bash
conda run -n gecco_mh pytest tests/test_phase6_parallel_extraction_subtracks.py -q
conda run -n gecco_mh pytest tests/test_phase6_candidate_generator.py tests/test_phase6_candidate_evaluator.py -q
conda run -n gecco_mh pytest tests/test_phase6_feedback_coordinator.py tests/test_phase6_run_n_shots_non_cmg.py -q
conda run -n gecco_mh pytest tests/test_phase4_orchestrated_judge.py tests/test_judge_orchestration.py tests/test_cmg_judge.py tests/test_cmg_registry.py -q
```

Also run source searches for forbidden patterns:

- `getattr(self, ..., None) or Constructor(...)`
- runtime JSON reliance
- duplicate monolith generation/evaluation methods
- JSON writes used as canonical state

## Completion checklist

- [ ] Characterization tests freeze the remaining monolith behavior.
- [ ] Missing collaborators fail fast; no fallback constructors remain.
- [ ] `CandidateGenerator` owns generation.
- [ ] `CandidateEvaluator` owns evaluation, repair, best-model updates, and finalization.
- [ ] Non-CMG `run_n_shots()` routes through extracted services.
- [ ] JSON is not treated as routine canonical runtime output.
- [ ] `FeedbackCoordinator` has no hidden fallback to `gecco.run_gecco`.
- [ ] Service tests are stronger and less mock-heavy.
- [ ] Monolith duplicate logic has been deleted only after tests protected the replacement path.
- [ ] `run_gecco.py` reads as orchestration-only shell code.

## Reviewer notes

Expected final review outcome:

- The Phase 6 completion work should be accepted when the monolith no longer owns generation, evaluation, repair, finalization, or canonical runtime JSON behavior.
- The service tests should demonstrate real boundary ownership without heavy mocking.

Signs the implementation is **not** complete:

- `GeCCoModelSearch` still performs core non-CMG work directly.
- Hidden fallback wiring still exists.
- JSON files are still asserted as runtime truth.
- Duplicate monolith methods remain after the service tests pass.
- Any dashboard, provider registry, or export/report files were touched for this Phase 6 work.
