# Phase 7: DuckDB coordination and status views

## Purpose

Lock down the DuckDB-backed coordination/status architecture after the Phase 5 canonical-state work and the Phase 6 runner/helper ownership fixes.

This phase should be a verification and tightening pass, not a second broad redesign. Phase 5 already moved runtime coordination state into DuckDB and added status/coordination views. Phase 6 moved CMG final status/history publication to service-owned boundaries. Phase 7 should prove those decisions are stable together and remove any unsafe leftover paths.

Follow `docs/codebase_cleanup_plan/implementation_guardrails.md` while implementing this plan. In particular, keep DuckDB canonical, avoid hidden fallback paths, and make deletion of unsafe old behavior part of the work.

## Depends on

- Phase 5 DuckDB canonical state work, including follow-up plans:
  - `docs/codebase_cleanup_plan/phase_5_duckdb_canonical_state_completion_plan.md`
  - `docs/codebase_cleanup_plan/phase_5_duckdb_canonical_state_fix_plan.md`
- Phase 6 service extraction/status ownership work, including follow-up plans:
  - `docs/codebase_cleanup_plan/phase_6_runner_helper_split_fix_plan.md`
  - `docs/codebase_cleanup_plan/phase6_status_ownership_fix_plan.md`
  - `docs/codebase_cleanup_plan/phase6_review_fixes_plan.md`
  - `docs/phase6_revision_difficulty_deck.html`

## Can run in parallel with

- None. This is a final coordination/status verification pass after Phases 5 and 6.

## Non-goals

- Do not add dashboard work.
- Do not add provider registry work.
- Do not add export/report features.
- Do not reintroduce JSON as runtime truth.
- Do not add compatibility wrappers or fallback constructors for old registry files.
- Do not move CMG final status/history publication back into `GeCCoModelSearch`.
- Do not add expensive fitting/HBI/PPC to default tests.

## Likely files/modules to inspect

- `gecco/coordination.py`
- `gecco/distributed_coordinator.py`
- `gecco/candidate_evaluation.py`
- `gecco/artifacts.py`
- `gecco/diagnostic_store/store.py`
- `gecco/diagnostic_store/schema.py`
- `tests/test_phase5_duckdb_canonical_state.py`
- `tests/test_phase6_candidate_evaluator.py`
- `tests/test_phase6_parallel_extraction_subtracks.py`
- `tests/test_cmg_runtime.py`
- `tests/test_cmg_registry.py`
- `tests/test_diagnostic_store_tools.py`

## Current architecture to preserve

- DuckDB is the canonical runtime coordination store.
- JSON outputs, where they still exist, are inspection-only.
- `SharedRegistry` owns DuckDB-backed coordination state.
- `SharedRegistry.open_existing(...)` is the read-only open path and must not initialize schema.
- Ordinary registry writes must not rerun schema DDL after initial setup.
- Registry mutations use the chosen single-writer/transaction strategy.
- `CandidateEvaluator` owns CMG evaluator completion/status/history publication.
- `ArtifactStore.write_iteration_results()` / DuckDB persistence must succeed before terminal registry status is published.
- `GeCCoModelSearch` may orchestrate, but must not overwrite evaluator-owned `complete_no_success` / `had_runnable_model=False` status at runner exit.

## Tests to write or strengthen first

- View consistency tests for `runtime_status_view` and `runtime_coordination_view` that cover:
  - `complete`
  - `complete_no_success`
  - `had_runnable_model=False`
  - candidate generation rows
  - generator status rows
  - judge feedback and judge failure rows
- Restart/reload tests that prove a new `SharedRegistry` or `SharedRegistry.open_existing(...)` observes committed DuckDB state without JSON fallback.
- Concurrency tests that prove the current lock/transaction strategy serializes concurrent registry writes.
- Source or behavior guards proving `GeCCoModelSearch.run_n_shots()` does not call `shared_registry.mark_complete(...)` or otherwise overwrite evaluator-owned terminal CMG status.
- Behavior tests for any remaining `SharedRegistry.mark_complete(...)` API:
  - either prove it is unused/deleted, or prove it cannot overwrite `complete_no_success` / `had_runnable_model=False`.
- Failure-ordering tests proving CMG persistence failure raises and does not publish terminal registry status.

## Implementation tasks

- Treat Phase 7 as a tightening pass over the current design, not a rewrite.
- Audit `SharedRegistry.mark_complete(...)`. Prefer deleting it if no production route needs it. If a legitimate caller remains, constrain it so it cannot overwrite evaluator-owned terminal no-success state, and add tests for that behavior.
- Verify all status/coordination reads come from DuckDB-backed tables or views, not JSON or latest in-memory client state.
- Ensure `runtime_coordination_view` answers historical/per-iteration questions from per-iteration rows, not current client status alone.
- Ensure `runtime_status_view` remains a simple current-client status view.
- Keep `CandidateEvaluator.finalize_iteration_results(...)` ordering strict: canonical persistence first, optional feedback recording second, registry publication last.
- Keep runner CMG helpers thin and orchestration-only.
- Make locking/transaction behavior explicit in tests rather than adding broad new abstractions.

## Deletion tasks

- Remove or constrain remaining ad hoc coordination/status paths that bypass the DuckDB registry.
- Remove any runtime code that reads `shared_registry.json`, `baseline.json`, or other JSON files as authoritative runtime state.
- Remove fallback behavior that silently reconstructs state outside DuckDB.
- Remove terminal-status publication routes owned by the runner instead of the evaluator/service boundary.

## Verification commands

Run Python/test commands with `conda run -n gecco_mh`.

```bash
conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q
conda run -n gecco_mh pytest tests/test_cmg_registry.py -q
conda run -n gecco_mh pytest tests/test_phase6_candidate_evaluator.py -q
conda run -n gecco_mh pytest tests/test_phase6_parallel_extraction_subtracks.py -q
conda run -n gecco_mh pytest tests/test_cmg_runtime.py -q
conda run -n gecco_mh pytest tests/test_diagnostic_store_tools.py -q
git diff --check
```

## Acceptance criteria

- DuckDB remains the only canonical runtime coordination state.
- Runtime/status JSON is not used as a source of truth.
- Read-only registry consumers do not run schema DDL.
- Ordinary writes do not rerun schema creation.
- Concurrent registry writes are stable under the chosen lock/transaction strategy.
- Restart/reload behavior reads committed DuckDB state correctly.
- `runtime_status_view` and `runtime_coordination_view` match canonical registry state for success, no-success, candidate generation, generator status, and judge feedback/failure cases.
- `complete_no_success` and `had_runnable_model=False` cannot be overwritten by runner-exit or generic completion paths.
- CMG terminal status is published only after canonical persistence succeeds.
- No dashboard/provider/export/report work is added.
