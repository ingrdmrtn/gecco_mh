# Phase 6 Follow-up Fix Plan 2: Review findings for parallel extraction subtracks

This is a second follow-up to:

- [`docs/codebase_cleanup_plan.md`](../codebase_cleanup_plan.md)
- [`docs/codebase_cleanup_plan/README.md`](README.md)
- [`docs/codebase_cleanup_plan/phase_6_parallel_extraction_subtracks.md`](phase_6_parallel_extraction_subtracks.md)
- [`docs/codebase_cleanup_plan/phase_6_parallel_extraction_subtracks_fix_plan.md`](phase_6_parallel_extraction_subtracks_fix_plan.md)

## Why this follow-up exists

The previous Phase 6 fix plan tightened the extraction story, but the current implementation still leaves important behavior owned by the monolith and hidden fallbacks. This follow-up only closes the remaining gaps found in code review. It does **not** restart Phase 6, and it does **not** expand the cleanup into dashboard, export/report, provider registry, or broader architecture work.

## Scope and non-goals

This plan stays strictly inside Phase 6.

Important constraints from the cleanup README that still apply here:

- DuckDB is the canonical runtime state store.
- JSON is **not** the runtime source of truth.
- Runtime JSON outputs are not part of the target architecture.
- No hidden legacy fallbacks.
- No compatibility wrappers.
- No dashboard work.
- No expensive default unit tests.
- Keep the solution deletion-first and simple.

Also:

- `run_gecco.py` should become an orchestration shell, not the owner of generation/evaluation internals.
- Do not add provider registry work.
- Do not add export/report work.
- Do not add new dashboard-facing behavior.

## Review findings to fix

1. `CandidateGenerator` and `CandidateEvaluator` are still thin adapters over monolith-owned methods instead of independent service boundaries.
2. `GeCCoModelSearch.run_n_shots()` still directly owns substantial non-CMG generation, evaluation, and finalization behavior.
3. The Phase 6 tests are still too mock-heavy and do not prove the extracted services independently enough.
4. `FeedbackCoordinator` still has a hidden fallback dependency on `gecco.run_gecco` / monolith-owned import flow.
5. JSON artifact writing still exists as routine runtime behavior in places that should be treated as non-canonical inspection output only.

## TDD-first approach

Use this order:

1. **Characterization tests** to freeze the remaining monolith behavior that must not change yet.
2. **Stronger contract/integration tests** for the extracted services so they can be proven without the monolith.
3. **Deletions only after replacement paths are covered** by the tests above.

### Recommended test files

Start with the existing Phase 6 test file, then add focused service tests where needed:

- `tests/test_phase6_parallel_extraction_subtracks.py`
- `tests/test_phase6_candidate_generator.py`
- `tests/test_phase6_candidate_evaluator.py`
- `tests/test_phase6_feedback_coordinator.py`
- `tests/test_phase6_run_n_shots_non_cmg.py`

Keep these tests lightweight. Use stubs/fakes for external systems, but do not over-mock the service under test.

## Vertical chunks

### A. Freeze current remaining monolith behavior with tests

**Dependency:** none.

**What to do:**

- Add characterization coverage for the current Phase 6 behavior that still exists in `GeCCoModelSearch` and `FeedbackCoordinator`.
- Capture the current non-CMG `run_n_shots()` flow at the boundary level before changing internals.
- Add at least one test that proves the current JSON-writing behavior is present today so it can be removed intentionally later.

**What not to do:**

- Do not refactor the production code in this chunk.
- Do not add broad mocks for the whole monolith just to make tests pass.
- Do not invent new abstractions yet.

**Ownership afterward:**

- Tests own the current behavior snapshot.
- Production code remains unchanged until the next chunks start replacing it.

**Acceptance criteria:**

- `tests/test_phase6_parallel_extraction_subtracks.py` contains explicit characterization tests for the remaining monolith-owned paths.
- The current fallback and JSON-writing behavior is documented by tests before deletion begins.

### B. Remove `FeedbackCoordinator` fallback-to-monolith dependency

**Dependency:** A.

**What to do:**

- Make `FeedbackCoordinator` receive all required collaborators explicitly.
- Remove the hidden fallback import/lookup path through `gecco.run_gecco`.
- Add a focused service test in `tests/test_phase6_feedback_coordinator.py` that fails if the coordinator reaches back into monolith-owned import flow.

**What not to do:**

- Do not add a compatibility wrapper around the old import path.
- Do not leave a “just in case” fallback to monolith state.
- Do not widen this into judge/provider registry work.

**Ownership afterward:**

- `FeedbackCoordinator` owns only explicit coordination logic.
- The monolith no longer supplies hidden defaults for feedback setup.

**Acceptance criteria:**

- The coordinator works only with injected dependencies.
- Tests prove the coordinator can be constructed and exercised without `gecco.run_gecco` fallback behavior.

### C. Extract independent generation boundary (not bound to monolith methods)

**Dependency:** A.

**What to do:**

- Move candidate generation logic out of monolith-owned methods and into a real service boundary.
- Keep the generator focused on explicit inputs and outputs.
- Add a targeted contract test in `tests/test_phase6_candidate_generator.py` that exercises the service directly.

**What not to do:**

- Do not keep the generator as a pass-through to `GeCCoModelSearch`.
- Do not add extra layers of indirection.
- Do not pull in dashboard/export/report concerns.

**Ownership afterward:**

- `CandidateGenerator` owns generation behavior.
- `GeCCoModelSearch` should stop being the place where generation logic lives.

**Acceptance criteria:**

- The generator can be tested without monolith private methods.
- The generator test proves explicit generation behavior with stubbed external model access only.

### D. Extract independent evaluation/finalization boundary

**Dependency:** A and C.

**What to do:**

- Move candidate evaluation and finalization out of `GeCCoModelSearch` into `CandidateEvaluator`.
- Make scoring, evaluation, and any required finalization steps explicit in the service interface.
- Add a targeted contract test in `tests/test_phase6_candidate_evaluator.py` that proves the evaluator directly.

**What not to do:**

- Do not keep the evaluator as a thin adapter over monolith-owned methods.
- Do not hide repair-loop or finalization details inside the monolith.
- Do not introduce expensive default fitting in the unit suite.

**Ownership afterward:**

- `CandidateEvaluator` owns evaluation/finalization behavior.
- The monolith only orchestrates inputs/outputs around the service.

**Acceptance criteria:**

- Evaluation behavior is testable as a standalone service.
- Finalization is no longer owned by `GeCCoModelSearch`.

### E. Route non-CMG `run_n_shots()` through extracted services

**Dependency:** B, C, and D.

**What to do:**

- Refactor `GeCCoModelSearch.run_n_shots()` so it delegates non-CMG generation, evaluation, and finalization to the extracted services.
- Keep `run_gecco.py` as orchestration-only glue.
- Add or extend `tests/test_phase6_run_n_shots_non_cmg.py` to prove the route uses the services instead of monolith-owned internals.

**What not to do:**

- Do not preserve old monolith-owned generation/evaluation internals as a hidden path.
- Do not add a second parallel implementation.
- Do not expand into provider registry or dashboard behavior.

**Ownership afterward:**

- `GeCCoModelSearch` becomes an orchestrator, not the owner of the underlying logic.
- Non-CMG execution goes through the extracted services.

**Acceptance criteria:**

- `run_n_shots()` no longer directly owns substantial non-CMG generation/evaluation/finalization logic.
- The tests prove the new delegation path.

### F. Reduce routine JSON artifact ownership / make non-canonical outputs explicit

**Dependency:** A and E.

**What to do:**

- Identify the JSON writes that are still part of routine runtime behavior.
- Move them out of the canonical runtime path or make them explicitly inspection-only.
- Keep DuckDB as the source of truth for runtime state and artifact coordination.
- Add tests that prove the runtime no longer depends on JSON as canonical state.

**What not to do:**

- Do not turn JSON output into a new source of truth.
- Do not add export/report features.
- Do not keep silent duplicate writes just for convenience.

**Ownership afterward:**

- JSON is only non-canonical inspection output, if retained at all.
- Runtime state remains in DuckDB.

**Acceptance criteria:**

- Tests show JSON is not required for runtime correctness.
- Any remaining JSON output is clearly non-canonical.

### G. Strengthen service tests and delete monolith duplicates

**Dependency:** B, C, D, E, and F.

**What to do:**

- Replace mock-heavy tests with service-focused contract/integration tests.
- Delete monolith-owned duplicate logic only after the service tests protect the replacement path.
- Keep the test suite small, fast, and direct.

**What not to do:**

- Do not delete code before the new service path is protected.
- Do not keep duplicate implementations “just for safety.”
- Do not add expensive default unit tests.

**Ownership afterward:**

- Each service has its own direct test coverage.
- The monolith no longer duplicates the same responsibilities.

**Acceptance criteria:**

- Phase 6 tests prove the extracted services independently.
- Monolith duplicate logic is removed.

## Suggested implementation order

1. Chunk A: freeze current behavior.
2. Chunk B: remove the `FeedbackCoordinator` fallback.
3. Chunk C: extract generation.
4. Chunk D: extract evaluation/finalization.
5. Chunk E: route `run_n_shots()` through the services.
6. Chunk F: remove routine JSON canonical behavior.
7. Chunk G: delete remaining monolith duplicates and tighten tests.

## Suggested verification commands

Run the focused tests after each chunk, then the wider Phase 6 suite:

```bash
conda run -n gecco_mh pytest tests/test_phase6_parallel_extraction_subtracks.py -q
conda run -n gecco_mh pytest tests/test_phase6_candidate_generator.py tests/test_phase6_candidate_evaluator.py -q
conda run -n gecco_mh pytest tests/test_phase6_feedback_coordinator.py tests/test_phase6_run_n_shots_non_cmg.py -q
```

If needed, run the broader related cleanup tests only after the focused service tests pass.

## Completion checklist

- [ ] Characterization tests freeze the remaining monolith behavior.
- [ ] `FeedbackCoordinator` has no hidden fallback to `gecco.run_gecco`.
- [ ] `CandidateGenerator` is an independent boundary.
- [ ] `CandidateEvaluator` is an independent boundary.
- [ ] Non-CMG `run_n_shots()` routes through extracted services.
- [ ] JSON is no longer treated as routine canonical runtime output.
- [ ] Service tests are stronger and less mock-heavy.
- [ ] Monolith duplicate logic has been deleted only after tests protected the replacement path.
- [ ] `run_gecco.py` reads as orchestration-only shell code.
