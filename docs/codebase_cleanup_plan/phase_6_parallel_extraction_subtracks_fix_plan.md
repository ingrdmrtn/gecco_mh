# Phase 6 Follow-up Fix Plan: Parallel extraction subtracks

This is a follow-up plan to:

- [`docs/codebase_cleanup_plan.md`](../codebase_cleanup_plan.md)
- [`docs/codebase_cleanup_plan/README.md`](README.md)
- [`docs/codebase_cleanup_plan/phase_6_parallel_extraction_subtracks.md`](phase_6_parallel_extraction_subtracks.md)

## 1) Context and scope

Phase 6 is intended to split `GeCCoModelSearch` into smaller services behind tests. The overall cleanup scope says:

- DuckDB is the canonical runtime state store.
- JSON is not the runtime source of truth.
- Runtime JSON outputs are not part of the target architecture.
- Dashboard work is out of scope.
- Old names, shims, and legacy code should be deleted once the replacement path is tested.

This fix plan stays within that scope. It does **not** broaden the cleanup into dashboard work or new export/report features.

## 2) Review findings to fix

1. `GeCCoModelSearch` still owns path, artefact, generation, evaluation, feedback, and distributed coordination behaviour directly.
2. The extracted service modules are still thin adapters over `search: Any` rather than independent services with explicit inputs/outputs.
3. `ArtifactStore` still writes JSON as routine runtime artefacts, and group/individual path handling is not explicit enough.
4. `RunContext` exists, but tempdir ownership/cleanup is not fully integrated into the runtime lifecycle.
5. The Phase 6 tests are too mock-heavy; they do not yet prove the service contracts independently.
6. `DistributedCoordinator` is still more of a monolith adapter than a clear DuckDB/status-view coordination service.

## 3) Guiding constraints

- Keep the Phase 6 scope narrow and deletion-first.
- Do not add dashboard changes.
- Do not introduce legacy compatibility wrappers.
- Do not make JSON the canonical runtime path.
- Keep the solution simple and test-driven.
- Use type annotations and Google-style docstrings for new Python code.

## 4) TDD-first implementation order

### 4.1 Write failing tests first

Add or extend focused tests for these behaviours:

| Test area | Purpose |
| --- | --- |
| `RunContext` path ownership | Prove resolved results paths and tempdir cleanup explicitly. |
| `ArtifactStore` contract | Prove write/read round trips and correct group vs individual filename handling. |
| Candidate generation | Prove explicit generation inputs/outputs without depending on `GeCCoModelSearch` internals. |
| Candidate evaluation | Prove explicit evaluation/finalisation behaviour and repair-loop boundaries. |
| Feedback coordination | Prove orchestration uses explicit judge inputs and returns canonical feedback artefacts. |
| Distributed coordination | Prove registry/status interactions through a dedicated coordinator contract. |

Recommended test file:

- `tests/test_phase6_parallel_extraction_subtracks.py`

## 5) Vertical implementation chunks

### Chunk A: `RunContext` and lifecycle ownership

**Scope reference:** Phase 6A in [`phase_6_parallel_extraction_subtracks.md`](phase_6_parallel_extraction_subtracks.md)

**Work:**

- Keep path resolution in `gecco/run_context.py`.
- Make explicit whether the run is group mode or individual mode.
- Ensure tempdir ownership is closed by the runtime, not just by tests.
- Preserve the existing results layout and make it explicit in one place.

**Acceptance:**

- `GeCCoModelSearch` does not resolve run paths itself.
- Tempdirs are created and cleaned up deterministically.

### Chunk B: `ArtifactStore` as the canonical artefact interface

**Scope reference:** Phase 6B and the overall DuckDB/JSON rules in [`README.md`](README.md)

**Work:**

- Keep persistence in `gecco/artifacts.py`.
- Make group vs individual filename logic explicit.
- Keep DuckDB writes/read paths canonical for runtime state.
- Treat any JSON output as non-canonical inspection output only if it must remain.

**Acceptance:**

- No accidental participant suffixes in group-mode paths.
- Runtime state is written through DuckDB first, not JSON first.

### Chunk C: `CandidateGenerator` extraction

**Scope reference:** Phase 6C in [`phase_6_parallel_extraction_subtracks.md`](phase_6_parallel_extraction_subtracks.md)

**Work:**

- Move generation behind explicit inputs and outputs.
- Stop reaching into private `GeCCoModelSearch` methods for generation details.
- Keep generator tests stub-friendly, but verify real service behaviour.

**Acceptance:**

- Generation can be tested without the monolith providing hidden state.

### Chunk D: `CandidateEvaluator` extraction

**Scope reference:** Phase 6D in [`phase_6_parallel_extraction_subtracks.md`](phase_6_parallel_extraction_subtracks.md)

**Work:**

- Keep fitting and repair-loop logic behind a dedicated interface.
- Ensure the evaluator receives everything it needs as explicit inputs.
- Keep finalisation/persistence calls in the service boundary, not in the monolith.

**Acceptance:**

- Evaluation contract tests cover the service directly.

### Chunk E: `FeedbackCoordinator` and orchestrated judge adapter

**Scope reference:** Phase 6E plus the judge rules in [`README.md`](README.md)

**Work:**

- Keep judge orchestration behind one coordinator.
- Make capability-driven behaviour explicit.
- Remove remaining legacy/manual judge-path ownership from `GeCCoModelSearch` once parity is proven.

**Acceptance:**

- Feedback orchestration is explicit and test-covered.

### Chunk F: `DistributedCoordinator` and DuckDB coordination

**Scope reference:** Phase 6F and the DuckDB canonical-state rules in [`README.md`](README.md)

**Work:**

- Make distributed coordination a small, explicit service.
- Keep registry/status reads aligned with DuckDB-backed state.
- Move the remaining monolith-facing coordination glue out of `run_gecco.py`.

**Acceptance:**

- Concurrency and status behaviour are covered by focused tests.

### Chunk G: Remove monolith-owned duplicates

**Scope reference:** Phase 6 exit criteria in [`phase_6_parallel_extraction_subtracks.md`](phase_6_parallel_extraction_subtracks.md)

**Work:**

- Remove the direct path-generation, artefact-write, candidate-generation, evaluation, feedback, and distributed-coordination logic from `GeCCoModelSearch` once each service is covered.
- Keep `run_gecco.py` as an orchestration shell, not the owner of service internals.

**Acceptance:**

- `GeCCoModelSearch` no longer owns these responsibilities directly.

## 6) Suggested verification order

Run the Phase 6 tests first, then the adjacent judge/registry tests that prove the new boundaries still work:

```bash
conda run -n gecco_mh pytest tests/test_phase6_parallel_extraction_subtracks.py -q
conda run -n gecco_mh pytest tests/test_phase4_orchestrated_judge.py tests/test_judge_orchestration.py tests/test_cmg_judge.py tests/test_cmg_registry.py -q
```

## 7) What not to do

- Do not add dashboard changes.
- Do not add compatibility wrappers for old paths or old judge code.
- Do not preserve JSON as the runtime source of truth.
- Do not widen the phase into export/report work.

## 8) Completion checklist

- [ ] `RunContext` owns path resolution and lifecycle cleanly.
- [ ] `ArtifactStore` handles group/individual paths explicitly.
- [ ] Candidate generation and evaluation are testable as standalone services.
- [ ] Judge feedback orchestration is isolated behind a coordinator.
- [ ] Distributed coordination is clearly separated.
- [ ] `GeCCoModelSearch` no longer directly owns these responsibilities.
- [ ] Phase 6 service contract tests pass.
