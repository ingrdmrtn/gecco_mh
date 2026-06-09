# Phase 5 Completion Plan: DuckDB Canonical Runtime State

This is a follow-up to:

- [`docs/codebase_cleanup_plan/phase_5_duckdb_canonical_state.md`](phase_5_duckdb_canonical_state.md)
- [`docs/codebase_cleanup_plan/phase_5_duckdb_canonical_state_fix_plan.md`](phase_5_duckdb_canonical_state_fix_plan.md)

## Context

Phase 5 is not done yet. After reviewing the current implementation, the remaining work is to finish the DuckDB canonical runtime-state migration and close a few correctness and hygiene gaps.

This plan is for a junior developer. Keep the solution simple. Fresh runs only. Do **not** add migration or backcompat code for older DB files.

## Review findings to fix

1. **Baseline fitting race is still a blocker.** Multiple distributed clients can call `fit_baseline_if_needed()` at the same time because it still does read-then-fit-then-write without registry-mediated single-fit coordination.
2. **Read-only registry construction is not truly read-only yet.** Monitor/dashboard style readers still go through write-style initialization and schema DDL.
3. **Routine writes still rerun schema creation.** `create_schema()` still runs on ordinary writes through `_with_connection(write=True)` unless callers override `initialise_schema`.
4. **Stale script cleanup is still needed.** `scripts/test_fit_model.py` still reads `shared_registry.json`.
5. **Test hygiene is incomplete.** Some new tests use misleading hidden `'.json'` string concatenation, and the current tests do not yet prove the missing behaviors.

## Scope

### In scope

- `gecco/coordination.py`
- `gecco/baseline.py`
- `gecco/cli/monitor_distributed.py`
- `gecco-mh-dashboard/dashboard/data_adapter.py`
- `scripts/test_fit_model.py`
- Focused Phase 5 tests only

### Out of scope

- Migrating existing databases
- Backward compatibility for old DB files
- Reworking export/report flows
- Full end-to-end LLM runs

## TDD-first implementation order

1. Write failing tests for the missing behaviors.
2. Implement the smallest code change that makes those tests pass.
3. Verify each vertical chunk before moving on.
4. Do not widen scope.

## Vertical chunks

### Chunk 1: True read-only registry open path for readers

**Tests to write first**

- `test_shared_registry_read_path_uses_read_only_duckdb_and_skips_schema_ddl`
- `test_shared_registry_multi_process_reads_do_not_create_schema`
- `test_dashboard_load_registry_snapshot_reads_shared_registry_duckdb`

These should fail before the fix because reader paths still act like writers.

**Implementation targets**

- `gecco/coordination.py`
  - add a clearly separate open-existing/read-only path
  - ensure reader construction uses `duckdb.connect(..., read_only=True)`
  - ensure reader construction does **not** call `create_schema()`
- `gecco/cli/monitor_distributed.py`
  - use the read-only registry path for monitor reads
- `gecco-mh-dashboard/dashboard/data_adapter.py`
  - load snapshots through `SharedRegistry(...).read()` only

**Dependency notes**

- This is the foundation for the rest of Phase 5.
- Do not try to preserve JSON fallback behavior.

**Verification command**

```bash
conda run -n gecco_mh pytest tests/test_cmg_registry.py tests/test_cmg_judge.py -q
```

**Acceptance outcome**

- Readers can open an existing DuckDB registry without schema DDL.
- Monitor/dashboard reads are truly non-mutating.

---

### Chunk 2: Baseline single-fit coordination in DuckDB only

**Tests to write first**

- `test_fit_baseline_if_needed_writes_only_to_registry_and_never_touches_baseline_json`
- `test_fit_baseline_if_needed_is_single_fit_under_concurrency`
- `test_baseline_read_after_initial_fit_uses_registry_state`

These should fail before the fix because baseline fitting still races and still depends on JSON-era behavior.

**Implementation targets**

- `gecco/baseline.py`
  - make `fit_baseline_if_needed()` use the registry as the only runtime-state source
  - add registry-mediated single-fit coordination
  - keep the code path simple: one fit winner, all others observe the stored result
  - do **not** bring back `baseline.json` or `baseline.lock`
- `gecco/coordination.py`
  - add the minimal registry methods needed for baseline coordination if they do not already exist

**Dependency notes**

- This must be treated as a blocking correctness issue.
- Do this after Chunk 1 so baseline readers/writers share the same registry behavior.

**Verification command**

```bash
conda run -n gecco_mh pytest tests/test_cmg_registry.py tests/test_cmg_judge.py -q
```

**Acceptance outcome**

- Only one distributed client fits the baseline.
- Every other concurrent client reuses the stored baseline.
- No baseline JSON files are read or written.

---

### Chunk 3: Stop routine writes from rerunning schema creation

**Tests to write first**

- `test_normal_registry_writes_do_not_recreate_schema_after_initialization`
- `test_with_connection_write_path_skips_create_schema_when_initialised`

These should fail before the fix because normal writes still trigger schema creation unless callers override it.

**Implementation targets**

- `gecco/coordination.py`
  - separate initial store setup from ordinary write mutations
  - keep schema DDL only in the writer initialization path
  - make ordinary write operations reuse the already-initialized store

**Dependency notes**

- Keep the initialization flow explicit: one writer init path, one read-only open-existing path.
- Do not add fallback DDL inside routine write methods.

**Verification command**

```bash
conda run -n gecco_mh pytest tests/test_cmg_registry.py -q
```

**Acceptance outcome**

- After initial setup, normal writes do not rerun `create_schema()`.
- Writer initialization remains separate and explicit.

---

### Chunk 4: CLI/dashboard path cleanup and stale script update

**Tests to write first**

- `test_shared_registry_callers_use_shared_registry_duckdb_paths`
- `test_available_tasks_detects_duckdb_registry_files`
- `test_test_fit_model_uses_shared_registry_duckdb`

These should fail before the fix because some call sites still point at the JSON-era registry path.

**Implementation targets**

- `gecco/cli/monitor_distributed.py`
  - remove any JSON-specific registry-loading assumptions
- `gecco-mh-dashboard/dashboard/data_adapter.py`
  - load registry snapshot from `shared_registry.duckdb`
- `scripts/test_fit_model.py`
  - update the stale script to use `SharedRegistry(...).read()` against `shared_registry.duckdb`

**Dependency notes**

- This chunk depends on Chunk 1.
- Keep the update narrow; do not redesign the CLI or dashboard.

**Verification command**

```bash
conda run -n gecco_mh pytest tests/test_cmg_judge.py tests/test_cmg_registry.py -q
```

**Acceptance outcome**

- All reader-facing paths point to `shared_registry.duckdb`.
- The stale script no longer reads `shared_registry.json`.

---

### Chunk 5: Test hygiene and proof of missing behaviors

**Tests to write first**

- Remove misleading hidden `'.json'` string concatenation in the new tests.
- Add/adjust tests so they prove the actual missing behaviors, especially:
  - concurrent baseline single-fit coordination
  - read-only constructor/open-path behavior
  - normal writes not rerunning schema DDL

Suggested focused names:

- `test_shared_registry_read_path_uses_read_only_duckdb_and_skips_schema_ddl`
- `test_fit_baseline_if_needed_is_single_fit_under_concurrency`
- `test_normal_registry_writes_do_not_recreate_schema_after_initialization`

**Implementation targets**

- `tests/test_phase5_duckdb_canonical_state.py` or the existing Phase 5-adjacent tests

**Dependency notes**

- Do this throughout the work, not at the end only.
- Prefer explicit assertions over string tricks or hidden concatenation.

**Verification command**

```bash
conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py tests/test_cmg_registry.py tests/test_cmg_judge.py -q
```

**Acceptance outcome**

- Tests clearly verify the new behavior.
- No confusing hidden string concatenation remains.

## File-level requirements

### `gecco/coordination.py`

- Keep `SharedRegistry` focused on DuckDB only.
- Provide one explicit initialization path for writers.
- Provide one explicit open-existing read-only path for readers.
- Do not call `create_schema()` from read-only reads.
- Do not rerun schema DDL during ordinary writes.
- Use registry-backed coordination for baseline single-fit behavior.

### `gecco/baseline.py`

- `fit_baseline_if_needed()` should rely on the registry, not JSON files.
- It should coordinate so only one client performs the fit.
- It should not accept or use `baseline_path`.
- It should not use `json` or `fcntl`.

### `gecco/cli/monitor_distributed.py`

- Use `SharedRegistry` read-only access for monitor reads.
- Do not keep any fallback for `shared_registry.json`.

### `gecco-mh-dashboard/dashboard/data_adapter.py`

- Load the registry snapshot with `SharedRegistry(...).read()`.
- Read from `shared_registry.duckdb`.

### `scripts/test_fit_model.py`

- Update the script to read the registry through `SharedRegistry`.
- Use `shared_registry.duckdb`.
- Do not keep the old JSON path.

### Tests

- Prefer one focused Phase 5 test file if that keeps the change small.
- If extending existing tests is simpler, keep the changes narrow.
- Add real concurrency coverage for the baseline race.
- Add real proof that the read-only registry path does not mutate state.

## What not to do

- Do not reintroduce `baseline.json` or `baseline.lock`.
- Do not treat JSON artifacts as runtime source of truth.
- Do not add migration/backcompat support for old DBs.
- Do not call `create_schema()` on normal read paths.
- Do not leave ordinary writes responsible for schema creation.
- Do not use the latest client state to answer historical or per-iteration questions.
- Do not over-engineer with extra abstractions.

## Coding standards

- Keep the implementation simple.
- Add type hints where you touch Python signatures.
- Use Google-style docstrings for new/changed Python functions, consistent with project instructions.
- Format Python with Black.

## Verification commands

Run these after the code changes:

```bash
conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q
conda run -n gecco_mh pytest tests/test_cmg_registry.py tests/test_cmg_judge.py -q
rg -n "baseline\.json|baseline\.lock|shared_registry\.json|create_schema\(" gecco gecco-mh-dashboard scripts tests
```

## Acceptance criteria

Phase 5 is complete only when all of the following are true:

- DuckDB is the canonical runtime state store.
- `SharedRegistry` has a true read-only open-existing path for readers.
- Baseline fitting is single-flight under concurrency.
- Routine writes do not rerun schema creation.
- No runtime code path depends on `shared_registry.json` or `baseline.json`.
- The stale script uses `SharedRegistry(...).read()` against `shared_registry.duckdb`.
- Tests prove the missing behaviors instead of implying them.

## Final checklist mapped to the original Phase 5 goals

- [ ] DuckDB is the source of truth.
- [ ] Runtime JSON no longer owns state.
- [ ] Read-only consumers do not run schema DDL.
- [ ] Concurrent baseline fitting is coordinated through the registry.
- [ ] Historical/runtime reads come from DuckDB, not latest ad hoc client state.
- [ ] CLI and dashboard paths point to `shared_registry.duckdb`.
- [ ] Stale script cleanup is complete.
- [ ] Test hygiene is clean and the tests prove the real behavior.
