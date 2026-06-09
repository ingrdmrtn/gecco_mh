# Phase 5 Follow-up: README-scoped final DuckDB fixes

## Purpose

Close the remaining Phase 5 gaps while following the top-level cleanup-plan
scope in `docs/codebase_cleanup_plan/README.md`.

This follow-up is narrower than the earlier Phase 5 completion/fix plans. The
core DuckDB migration is already in place. The remaining work is to finish the
last README-aligned correctness and test-hygiene fixes without widening scope
into dashboard code.

## Relationship to prior plans

- `docs/codebase_cleanup_plan/README.md`
  - This is the controlling scope document for this follow-up.
  - Its key constraint here is: dashboard work is out of scope.
- `docs/codebase_cleanup_plan/phase_5_duckdb_canonical_state.md`
  - Establishes the Phase 5 goals: DuckDB is canonical runtime state, runtime
    JSON is not the source of truth, and concurrency must be validated.
- `docs/codebase_cleanup_plan/phase_5_duckdb_canonical_state_fix_plan.md`
  - Captured the first round of DuckDB/runtime-state follow-up work.
- `docs/codebase_cleanup_plan/phase_5_duckdb_canonical_state_completion_plan.md`
  - Captured a broader finalization pass, but includes dashboard tasks that are
    not in line with `README.md`.

## Scope decision for this follow-up

- Follow `README.md` over the broader completion-plan scope.
- Do not touch dashboard files.
- Do not reopen the main DuckDB migration design.
- Do not add compatibility fallbacks for JSON-era runtime state.
- Keep the changes limited to the remaining review findings that still matter
  under README-level scope.

## Remaining gaps to fix

1. **Monitor read-path robustness is incomplete.**
   - `gecco/cli/monitor_distributed.py` now reads from DuckDB, but its error
     handling should safely treat unreadable or invalid registry DB files as
     unavailable monitor state rather than crashing.

2. **Baseline concurrency proof is still weaker than the Phase 5 requirement.**
   - The focused Phase 5 test currently uses thread-level concurrency.
   - The original Phase 5 plan asked for concurrent client validation, so the
     final proof should exercise process-level or otherwise clearly distributed
     contention.

3. **The stale script path update is not directly protected by a focused test.**
   - `scripts/test_fit_model.py` was updated to use DuckDB, but the completion
     plan's intended regression test for that script path is still missing.

4. **Test hygiene still contains misleading JSON-era registry filenames.**
   - Related tests still use names like `registry.json` or `.json` temp-file
     suffixes even though runtime JSON is no longer canonical state.
   - The remaining tests should use neutral registry paths or DuckDB-oriented
     names so the test surface matches the target architecture.

## Non-goals

- Do not touch `gecco-mh-dashboard/*`.
- Do not modify the original Phase 5 plan file.
- Do not add migration/backcompat logic for old registry JSON or baseline JSON
  files.
- Do not redesign `SharedRegistry` beyond the narrow monitor/read-path fix.
- Do not add expensive end-to-end LLM or fitting tests to the default unit
  suite.

## Files to inspect

- `gecco/cli/monitor_distributed.py`
- `scripts/test_fit_model.py`
- `tests/test_phase5_duckdb_canonical_state.py`
- `tests/test_cmg_registry.py`
- `tests/test_judge_orchestration.py`
- Any directly related monitor/script tests if a narrower existing test file is
  simpler than adding a new one

## TDD-first implementation order

Write or update the tests below before changing implementation.

### Chunk A: Monitor read-path robustness

1. Add a focused regression test showing that monitor registry loading does not
   crash when `shared_registry.duckdb` exists but is unreadable or not a valid
   DuckDB database.
2. Keep the monitor behavior simple: return no snapshot / waiting state rather
   than raising through the CLI read path.
3. Implement the smallest error-handling change needed in the monitor code.

### Chunk B: Stronger baseline concurrency proof

1. Replace or supplement the thread-based baseline concurrency test with a
   process-based or otherwise clearly distributed concurrency test.
2. Prove that exactly one contender performs the fit and all others observe the
   stored baseline from DuckDB.
3. Keep the test deterministic and lightweight.

### Chunk C: Protect the stale script cleanup with a direct test

1. Add the missing focused test for `scripts/test_fit_model.py`.
2. Prove that the script reads model code from `shared_registry.duckdb` through
   `SharedRegistry`, not from `shared_registry.json`.
3. Keep the test narrow to the script helper path rather than full CLI
   execution.

### Chunk D: Final JSON-era test-hygiene cleanup

1. Update remaining focused/adjacent tests that still use misleading
   `registry.json` names or hidden `.json` concatenation.
2. Prefer suffixless registry paths or explicit DuckDB-oriented names.
3. Keep the cleanup narrow: rename test paths and assertions only where they are
   relevant to Phase 5 runtime-state behavior.

## Suggested tests to add or update

### In `tests/test_phase5_duckdb_canonical_state.py`

- Add a monitor regression test such as:
  - `test_monitor_load_registry_returns_none_for_invalid_duckdb_file`
- Replace or supplement:
  - `test_fit_baseline_if_needed_is_single_fit_under_concurrency`
  with a process-level proof
- Add:
  - `test_test_fit_model_uses_shared_registry_duckdb`

### In existing adjacent tests if simpler

- Rename misleading registry test paths in:
  - `tests/test_cmg_registry.py`
  - `tests/test_judge_orchestration.py`

## Implementation notes

### `gecco/cli/monitor_distributed.py`

- Keep `SharedRegistry.open_existing(...).read()` as the read path.
- Broaden error handling only enough to treat invalid/unreadable DuckDB files as
  unavailable monitor state.
- Do not add JSON fallback behavior.

### `scripts/test_fit_model.py`

- The code path should remain DuckDB-only.
- The new test should protect the helper that loads models from the registry.

### Tests

- Use the `gecco_mh` conda environment.
- Keep tests fast and deterministic.
- Prefer explicit assertions over string tricks.
- Avoid JSON-era names in newly touched test paths.

## Acceptance criteria

This README-scoped Phase 5 follow-up is complete when all of the following are
true:

- `gecco/cli/monitor_distributed.py` safely handles invalid/unreadable
  `shared_registry.duckdb` files without crashing the monitor read path.
- The baseline single-fit guarantee is proven with a stronger concurrency test
  than same-process threads alone.
- `scripts/test_fit_model.py` has a direct regression test proving it reads from
  `shared_registry.duckdb` through `SharedRegistry`.
- Misleading `registry.json` / hidden `.json` test-path naming has been cleaned
  up in the remaining Phase 5-relevant tests.
- No dashboard files were modified.

## Suggested verification commands

Use the `gecco_mh` conda environment for all Python commands:

- `conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q`
- `conda run -n gecco_mh pytest tests/test_cmg_registry.py tests/test_cmg_judge.py tests/test_judge_orchestration.py -q`
- `conda run -n gecco_mh python -m pytest tests -q`

## Suggested implementation order

1. Add the new/updated focused tests first.
2. Fix monitor error handling.
3. Strengthen the baseline concurrency proof.
4. Add the direct `scripts/test_fit_model.py` regression test.
5. Clean up the remaining JSON-era test naming.
6. Re-run the focused Phase 5 tests.
7. Re-run the adjacent registry/judge tests.
