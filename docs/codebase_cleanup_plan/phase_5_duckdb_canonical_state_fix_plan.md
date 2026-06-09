# Phase 5 Follow-up Fix Plan: DuckDB Canonical Runtime State

This plan is a follow-up to [`docs/codebase_cleanup_plan/phase_5_duckdb_canonical_state.md`](phase_5_duckdb_canonical_state.md).

## Context

- Original Phase 5 purpose: move runtime state and coordination to DuckDB.
- Original acceptance criteria: DuckDB is source of truth; runtime JSON no longer owns state.
- Review found three gaps:
  1. `SharedRegistry` read paths call `create_schema()` and open read-write connections.
  2. `runtime_coordination_view` / client-count helpers use latest client state for historical iterations.
  3. `baseline.json` still behaves like runtime state.

## Scope

### In scope

- `gecco/coordination.py`
- `gecco/diagnostic_store/schema.py`
- `gecco/baseline.py`
- `gecco/cli/run_gecco_distributed.py`
- `gecco/cli/run_judge_orchestrator.py`
- `gecco/cli/monitor_distributed.py`
- `gecco/cli/reset_distributed.py`
- `gecco-mh-dashboard/dashboard/data_adapter.py`
- `gecco-mh-dashboard/dashboard/config.py`
- `gecco-mh-dashboard/app.py`
- Focused Phase 5 tests only

### Out of scope

- Migrating existing DBs or adding backcompat for old DBs
- Diagnostic artifact rebuild redesign
- Exports/reports
- Full end-to-end LLM runs

Fresh runs only. Do not write migration code for existing `shared_registry.json`/`baseline.json` state.

## TDD-first implementation order

### 1) Add failing Phase 5 tests first

Create one focused test file (recommended, not mandatory): `tests/test_phase5_duckdb_canonical_state.py`.
It is also fine to extend the existing Phase 5-adjacent tests if that keeps the changes tighter.

Add tests in this order:

| Test name | Before fix should fail because | After fix should pass because |
| --- | --- | --- |
| `test_shared_registry_read_path_uses_read_only_duckdb_and_skips_schema_ddl` | `read()` calls `create_schema()` and opens a normal connection | read path uses `duckdb.connect(..., read_only=True)` and never runs DDL |
| `test_shared_registry_multi_process_reads_do_not_create_schema` | multiple readers contend with write-style initialization | multiple readers can read the same DB without schema creation on reads |
| `test_runtime_iteration_history_counts_use_per_iteration_fields` | counts come from latest client state instead of the row for that iteration | counts and `runtime_coordination_view` use per-iteration history fields |
| `test_fit_baseline_if_needed_writes_only_to_registry_and_never_touches_baseline_json` | function still needs `baseline_path`, writes JSON, and uses `.lock` | function only reads/writes the registry and leaves no baseline JSON files behind |
| `test_shared_registry_callers_use_shared_registry_duckdb_paths` | CLI/UI call sites still use `shared_registry.json` | all updated call sites use `shared_registry.duckdb` |

If practical, add one narrow dashboard/monitor test for canonical reads:

- `test_dashboard_load_registry_snapshot_reads_shared_registry_duckdb`
- `test_available_tasks_detects_duckdb_registry_files`

## Vertical implementation chunks

Each chunk should be implemented and verified end-to-end before moving to the next one. Keep them vertical: add the tests first, then the minimum code needed for that chunk, then verify.

### Chunk A: DuckDB read-only registry access and multiprocess concurrency

**Tests to write first**

- Extend `tests/test_cmg_registry.py` or add `tests/test_phase5_duckdb_canonical_state.py` tests for:
  - `test_shared_registry_read_path_uses_read_only_duckdb_and_skips_schema_ddl`
  - `test_shared_registry_multi_process_reads_do_not_create_schema`
  - `test_registry_supports_restart_and_reload`
  - `test_registry_concurrent_clients_share_single_canonical_store`

**Implementation files/functions**

- `gecco/coordination.py`
  - `SharedRegistry.__init__`
  - `SharedRegistry._initialise_store`
  - `SharedRegistry._with_connection`
  - `SharedRegistry.read`
- Optional small adjustments in `gecco/diagnostic_store/schema.py` only if the read-only path exposes schema initialization issues.

**Dependency notes**

- Do this first because all later chunks depend on `read()` being safe and non-mutating.
- Fresh runs only; do not add migrations for older DB files.

**Verification command**

```bash
conda run -n gecco_mh pytest tests/test_cmg_registry.py -q
```

**Original Phase 5 bullets satisfied**

- Tests to write first (DuckDB state contract tests, writer/reader round-trip tests, concurrent client tests)
- Implementation tasks (make DuckDB the canonical runtime state store; validate concurrency with a clear write strategy)
- Deletion tasks (remove code paths that treat JSON as state)
- Concurrency/locking validation
- Acceptance criteria: DuckDB is the source of truth; runtime JSON no longer owns state

### Chunk B: Per-iteration coordination history and view/count correctness

**Tests to write first**

- Extend `tests/test_cmg_registry.py` or add dedicated tests for:
  - `test_runtime_iteration_history_counts_use_per_iteration_fields`
  - `test_runtime_views_match_registry_state`
  - `test_get_max_iteration_for_client_retrying_after_multiple_complete`
  - `test_get_max_iteration_for_client_ignores_retrying`

**Implementation files/functions**

- `gecco/diagnostic_store/schema.py`
  - `CREATE_STATEMENTS` for `runtime_iteration_history`
  - `CREATE OR REPLACE VIEW runtime_coordination_view`
- `gecco/coordination.py`
  - `SharedRegistry.update()`
  - `SharedRegistry.read()`
  - `SharedRegistry.count_clients_complete()`
  - `SharedRegistry.count_clients_with_models()`

**Dependency notes**

- Depends on Chunk A because the read path must be trustworthy before changing count/view logic.
- Fresh runs only; no backcompat for pre-existing DB schemas.

**Verification command**

```bash
conda run -n gecco_mh pytest tests/test_cmg_registry.py -q
```

**Original Phase 5 bullets satisfied**

- Tests to write first (view consistency tests for status/coordination views)
- Implementation tasks (add status and coordination views backed by DuckDB)
- Deletion tasks (remove assumptions that runtime JSON files are canonical)
- Acceptance criteria: historical counts come from per-iteration history, not latest client state

### Chunk C: DuckDB-only baseline runtime state with baseline JSON removed

**Tests to write first**

- Add or extend focused tests for:
  - `test_fit_baseline_if_needed_writes_only_to_registry_and_never_touches_baseline_json`
  - `test_registry_round_trip_preserves_runtime_snapshot`
  - a narrow caller test in `tests/test_cmg_registry.py` or `tests/test_cmg_judge.py` if needed to confirm baseline is retrieved from the registry

**Implementation files/functions**

- `gecco/baseline.py`
  - `fit_baseline_if_needed`
- `gecco/cli/run_gecco_distributed.py`
  - baseline setup call site near `fit_baseline_if_needed(...)`

**Dependency notes**

- Depends on Chunk A because baseline reads/writes go through `SharedRegistry`.
- Keep the runtime baseline path DuckDB-only; do not preserve `baseline.json` as a cache, fallback, or compatibility shim.

**Verification command**

```bash
conda run -n gecco_mh pytest tests/test_cmg_registry.py -q
```

**Original Phase 5 bullets satisfied**

- Tests to write first (DuckDB state contract tests, writer/reader round-trip tests)
- Implementation tasks (runtime JSON no longer owns state; keep future export/report work separate)
- Deletion tasks (remove code paths that treat JSON as state)
- Acceptance criteria: runtime JSON no longer owns state

### Chunk D: CLI/dashboard path cleanup for `shared_registry.duckdb`

**Tests to write first**

- Extend or add focused tests for:
  - `test_shared_registry_callers_use_shared_registry_duckdb_paths`
  - `test_dashboard_load_registry_snapshot_reads_shared_registry_duckdb`
  - `test_available_tasks_detects_duckdb_registry_files`
  - existing CLI tests in `tests/test_cmg_judge.py` if they need path updates

**Implementation files/functions**

- `gecco/cli/run_judge_orchestrator.py`
  - `run_orchestrator`
- `gecco/cli/monitor_distributed.py`
  - `load_registry`
  - any UI text that mentions the registry path
- `gecco/cli/reset_distributed.py`
  - `REGISTRY_FILES`
  - `DUCKDB_STATE_FILES`
  - `BASELINE_FILES`
  - `scan_state`
- `gecco-mh-dashboard/dashboard/data_adapter.py`
  - `load_registry_snapshot`
- `gecco-mh-dashboard/dashboard/config.py`
  - `available_tasks`
- `gecco-mh-dashboard/app.py`
  - `main`

**Dependency notes**

- Depends on Chunk A so the dashboard/CLI code can safely read the registry.
- Depends on Chunk C for removing `baseline.json`/`baseline.lock` references.
- Keep the change focused on path assumptions and messages; do not widen scope to report/export work.

**Verification command**

```bash
conda run -n gecco_mh pytest tests/test_cmg_judge.py tests/test_cmg_registry.py -q
```

**Original Phase 5 bullets satisfied**

- Deletion tasks (remove assumptions that runtime JSON files are canonical)
- Implementation tasks (make DuckDB the canonical runtime state store; remove runtime JSON source-of-truth behavior)
- Acceptance criteria: DuckDB is the source of truth; runtime JSON no longer owns state

## Detailed file-level requirements

Use the vertical chunks as the main plan. The notes below are just file-level guardrails so you do not have to reread the whole plan.

### `gecco/coordination.py`

- `SharedRegistry.__init__`, `_initialise_store`, and `_with_connection` must separate write initialization from read-only reads.
- Read paths must use `duckdb.connect(..., read_only=True)` and skip `create_schema()`.
- Write paths must keep the exclusive-lock + explicit transaction strategy.
- `update()`, `read()`, `count_clients_complete()`, and `count_clients_with_models()` must use per-iteration history fields.

### `gecco/diagnostic_store/schema.py`

- Add the per-iteration columns needed by the historical counts and status view.
- Update `runtime_coordination_view` so it does not depend on latest client state for old iterations.
- Fresh runs only: no schema migration/backcompat work.

### `gecco/baseline.py`

- `fit_baseline_if_needed` should require `registry` and stop using `baseline_path`, `json`, or `fcntl`.
- The function should read from `registry.read()["baseline"]`, fit only when missing, and persist only with `registry.set_baseline(...)`.

### `gecco/cli/run_gecco_distributed.py`

- Update the baseline call site to match the new function signature.
- Remove the `baseline.json` path variable.

### `gecco/cli/run_judge_orchestrator.py`

- Construct `SharedRegistry` from `shared_registry.duckdb`.
- Keep judge behavior unchanged except for the registry path.

### `gecco/cli/monitor_distributed.py`

- Read registry state through `SharedRegistry` only.
- Remove JSON-specific fallback assumptions from registry loading.

### `gecco/cli/reset_distributed.py`

- Remove `shared_registry.json`, `baseline.json`, and `baseline.lock` references.
- Keep only DuckDB state files in reset logic.

### `gecco-mh-dashboard/dashboard/data_adapter.py`

- Load registry state via `SharedRegistry(...).read()`.

### `gecco-mh-dashboard/dashboard/config.py`

- Detect tasks by the presence of `shared_registry.duckdb`.

### `gecco-mh-dashboard/app.py`

- Update waiting text to mention `shared_registry.duckdb`.

### Verification

- Run the focused Phase 5 pytest command from the `gecco_mh` conda environment.
- For the grep check, use plain `rg`:

```bash
rg -n "baseline\.json|baseline\.lock|shared_registry\.json|create_schema\(conn\)" gecco gecco-mh-dashboard tests
```

## Dependencies / execution order

1. Add the failing tests.
2. Fix `SharedRegistry` read/write connection handling.
3. Fix per-iteration history and view logic.
4. Remove baseline JSON runtime behavior.
5. Update CLI/dashboard path assumptions.
6. Run verification and grep for forbidden JSON runtime paths.

## What not to do

- Do not reconstruct runtime coordination state from JSON artifacts.
- Do not keep `baseline.json` as a fallback or cache.
- Do not call `create_schema()` on read paths.
- Do not use latest client status to answer historical per-iteration questions.
- Do not add migration/backcompat code for old DBs.
- Do not expand scope to full end-to-end LLM runs.

## Coding standards

- Use type annotations for new/changed Python signatures.
- Add Google-style docstrings for new/changed functions.
- Keep the implementation simple.
- Format Python with Black.

## Verification commands

Use these commands after the code changes:

```bash
conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q
conda run -n gecco_mh pytest tests/test_cmg_registry.py tests/test_cmg_judge.py -q
rg -n "baseline\.json|baseline\.lock|shared_registry\.json|create_schema\(conn\)" gecco gecco-mh-dashboard tests
```

If the project has an agreed formatter/linter, run it only for the touched files.

## Acceptance criteria

- DuckDB is the source of truth.
- Runtime JSON no longer owns state.
- `SharedRegistry.read()` uses read-only DuckDB connections.
- Historical counts come from per-iteration history, not latest client state.
- Baseline runtime state lives only in DuckDB.
- CLI and dashboard paths/messages point to `shared_registry.duckdb`.

## Final checklist mapped to the original Phase 5 plan

- [ ] Original Phase 5 “Tests to write first” (lines 25-31) are covered: DuckDB state contract tests, writer/reader round-trip tests, concurrent client tests, restart/reload tests, and view consistency tests.
- [ ] Original Phase 5 “Implementation tasks” (lines 33-39) are covered: DuckDB is the canonical runtime state store, JSON source-of-truth behavior is removed, views are backed by DuckDB, future export/report work stays separate, and concurrency is validated with a clear write strategy.
- [ ] Original Phase 5 “Deletion tasks” (lines 41-45) are covered: code paths that treat JSON as state are removed, and runtime JSON files are no longer assumed canonical.
- [ ] Original Phase 5 “Concurrency/locking validation” (lines 46-50) is covered: concurrent clients and restart/reload behavior are tested, and the read/write lock strategy is verified.
- [ ] Original Phase 5 “Acceptance criteria” (lines 52-55) are met: DuckDB is the source of truth, and runtime JSON no longer owns state.
