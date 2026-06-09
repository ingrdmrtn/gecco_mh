# Phase 7 review fixes: read-only registry opens and CLI terminal status

## Summary

This is a focused follow-up to Phase 7.

Implement exactly two fixes and nothing else:

1. `gecco/cli/run_test_evaluation.py::run_test_evaluation()` must open the existing `shared_registry.duckdb` via `SharedRegistry.open_existing(registry_path)`, not `SharedRegistry(registry_path)`.
2. `gecco/cli/monitor_distributed.py` CLI-only `build_summary_stats()` and `build_client_table()` must treat `complete_no_success` as a terminal non-error status.

Scope stays unchanged. Dashboard work is out of scope: do not touch `gecco-mh-dashboard/**`. No dashboard tests.

## Pipeline Map

### A. test-evaluation registry read pipeline

`results_dir` (CLI args) -> `registry_path` -> `SharedRegistry.open_existing(registry_path)` -> collect candidates / read baseline -> write `top_models_test.json` + optional diagnostic store write.

Contract: the registry is read-only post-processing and the canonical DuckDB snapshot for this run. It must not initialize schema, run DDL, or touch runtime coordination state unnecessarily.
No manifest/hash freshness mechanism is required for this focused fix because no derived registry artifact is produced; freshness/provenance is runtime proof that the provided temp `results_dir/shared_registry.duckdb` is read on each call and no stale/global/default registry handle is accepted.

Producer-consumer handoff trace:

- writer/source of truth: the prior pipeline produced `results_dir/shared_registry.duckdb`
- reader: `run_test_evaluation()`
- same source of truth: the on-disk DuckDB at `registry_path`
- drift test: the regression must fail if runtime stops calling `open_existing(registry_path)` or starts using the constructor path.

### B. CLI monitor status pipeline

`load_registry` -> `SharedRegistry.open_existing().read()` -> snapshot `client_entries` -> `build_summary_stats()` / `build_client_table()` -> Rich CLI output.

Contract: `client_entries` is a snapshot read from the registry, and `complete_no_success` must be treated as terminal/non-error in both summary and table rendering.

Producer-consumer handoff trace:

- writer/source of truth: registry producers update the DuckDB snapshot
- reader: monitor CLI snapshot path
- same helper/source of truth: `client_entries`
- drift test: the regression must prove `complete_no_success` is counted as completed and rendered as terminal, not error.

## Contract Matrix

| Asset / interface | Canonical source of truth | Producer / writer | Readers / consumers | Required behavior | Validation / check | Positive test | Negative test | Forbidden fallback / shortcut |
|---|---|---|---|---|---|---|---|---|
| `shared_registry.duckdb` for test-evaluation | On-disk DuckDB at `results_dir/shared_registry.duckdb` | Upstream Phase 7 registry writer | `run_test_evaluation()` | Read-only canonical snapshot for post-processing | Must be opened from the provided `results_dir` at runtime | `test_run_test_evaluation_opens_existing_registry_read_only` proves the temp registry is opened from `results_dir` | `test_run_test_evaluation_does_not_use_default_results_dir` fails if a default/global path is read | Hardcoded paths, default results dir, JSON fallback |
| `SharedRegistry.open_existing` | Existing registry file | N/A | `run_test_evaluation()`, monitor CLI | Opens without initializing schema/DDL/runtime coordination | Assert `SharedRegistry.open_existing(registry_path)` is called with the provided path | `test_run_test_evaluation_opens_existing_registry_read_only` mocks `open_existing` and verifies path+call count | Constructor path is used instead | `SharedRegistry(registry_path)` on read-only path |
| `run_test_evaluation` registry handoff | Runtime `registry_path` from CLI args | `results_dir` CLI parsing | candidate collection / baseline reading | Preserve read-only handoff to canonical registry | Test must fail if runtime path is not honored | `test_run_test_evaluation_does_not_use_default_results_dir` sets temp non-default `results_dir` and leaves default empty/unusable | Existence-only assertion without `open_existing` | Hidden global state, stale registry handle |
| `monitor` `client_entries` snapshot | Registry read snapshot | `SharedRegistry.open_existing().read()` | summary stats + client table | Use current snapshot only | Snapshot-based test | `test_monitor_summary_counts_complete_no_success_as_completed` and `test_monitor_client_table_treats_complete_no_success_as_terminal` reflect current entries | Hidden JSON/default fallback | Re-reading some alternate source |
| `complete_no_success` terminal status classification | Monitor status taxonomy | Registry/client status writer upstream | `build_summary_stats()`, `build_client_table()` | Count as completed and render as terminal non-error | Assert completed summary includes it; table style/classification is non-error | `test_monitor_summary_counts_complete_no_success_as_completed` + `test_monitor_client_table_treats_complete_no_success_as_terminal` | Shown as red/error or excluded from completed summary | Treat as non-terminal, error, or unknown |

## Failure Cases To Prevent

- Using `SharedRegistry(...)` on the read-only test-evaluation path.
- Command string is correct but runtime still instantiates the constructor.
- Tests mock away registry opening too much and no longer verify the handoff.
- `complete_no_success` shown as red/error.
- `complete_no_success` excluded from the completed summary.
- Accidentally touching `gecco-mh-dashboard/**`.
- Existence-only test on registry path without asserting `open_existing`.
- Hidden JSON/default fallback.
- Accepting both correct and incorrect registry locations.
- Tests that only check strings in CLI commands.

## Implementation Phases

### Phase 1: `run_test_evaluation` read-only registry open

Tests first:

- Add/extend a focused regression test in `tests/test_phase5_duckdb_canonical_state.py`.
- Use `tmp_path` with a non-default `results_dir` containing `shared_registry.duckdb`.
- Add a negative case that leaves the default results location empty/unusable and fails if the implementation silently reads that default path.
- Patch expensive work (`load_config`, `load_splits`, `collect_candidates`, and baseline/registry reads as needed).
- Assert the runtime path calls `SharedRegistry.open_existing(registry_path)` and does not instantiate `SharedRegistry(registry_path)`.
- Do not patch away both `SharedRegistry.open_existing` and the registry object in a way that makes the open path unobservable.

Implementation steps:

- Change `run_test_evaluation()` to open the existing registry with `SharedRegistry.open_existing(registry_path)`.
- Keep the rest of the flow unchanged.
- Do not add fallback constructors or extra initialization.

Tests to run:

- Targeted phase test for `run_test_evaluation`.

Expected behavior:

- Before implementation: test fails because constructor/open path is wrong.
- After implementation: test passes and the registry remains read-only.

Greps/review checks:

- Verify `run_test_evaluation` uses `open_existing`.
- Verify no constructor call remains for this registry handoff.
- Verify no dashboard files changed.

Do not proceed unless the regression proves the runtime path reads the registry at the provided `results_dir` and does not reuse stale/global/default handles.

### Phase 2: CLI monitor terminal status classification

Tests first:

- Add tests for `build_summary_stats()` covering `running`, `complete`, and `complete_no_success`.
- Add a `build_client_table()` test that proves `complete_no_success` is terminal and non-error.
- Keep assertions minimal and avoid brittle Rich rendering where possible.

Implementation steps:

- Update `build_summary_stats()` so completed includes `complete_no_success`.
- Update `build_client_table()` so `complete_no_success` follows the terminal/success styling path.
- Keep changes local and CLI-only.

Tests to run:

- Targeted monitor/distributed tests.

Expected behavior:

- Before implementation: `complete_no_success` is missing or treated as error.
- After implementation: it is counted as completed and rendered as terminal/non-error.

Greps/review checks:

- Verify `complete_no_success` appears in the CLI monitor terminal status logic/tests.
- Verify no dashboard files changed.

Do not proceed unless both summary counting and table classification pass.

### Phase 3: integration/review verification

Tests first:

- Run the focused Phase 7 and related regression suites.
- Confirm the registry read path and monitor classification together remain stable.

Implementation steps:

- Make only final small fixes if a test exposes drift.
- Do not widen scope.

Tests to run:

- Phase 7-focused pytest commands listed below.

Expected behavior:

- Before implementation: at least one targeted regression fails.
- After implementation: all targeted regressions pass.

Greps/review checks:

- Confirm no `gecco-mh-dashboard/**` files changed.
- Confirm no schema/registry redesign was introduced.

Do not proceed unless the final audit matches the contract matrix.

## Forbidden Patterns

- Dashboard files.
- Schema changes.
- Registry redesign.
- Compatibility wrappers.
- JSON fallback.
- Hardcoded results paths.
- Hidden global state.
- Accepting both correct and incorrect registry locations.
- Existence-only validation.
- Tests that only check command strings.
- Tests that mock away the registry-opening behavior.
- Any change outside the two listed fixes.

## Final Verification

Run these commands with `conda run -n gecco_mh`:

```bash
conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q -k "run_test_evaluation or complete_no_success or monitor_distributed"
conda run -n gecco_mh pytest tests/test_phase7_duckdb_coordination_and_status_views.py -q
conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q
conda run -n gecco_mh pytest tests/test_cmg_registry.py -q
git diff --check
```

Audit checks:

- Verify `run_test_evaluation` uses `SharedRegistry.open_existing` and does not instantiate `SharedRegistry` for `registry_path`.
- Suggested audit: `git diff --name-only -- gecco-mh-dashboard` (or equivalent) should return no files.
- Verify no `gecco-mh-dashboard/**` files changed.
- Verify `complete_no_success` appears in monitor CLI terminal status logic and tests.

Acceptance checklist:

- `run_test_evaluation()` opens the existing DuckDB registry read-only from the provided `results_dir`.
- The implementation does not initialize schema, run DDL, or touch runtime coordination state on this post-processing path.
- `complete_no_success` is treated as terminal/non-error in monitor summary and client table output.
- No dashboard files or tests were touched.
- No fallback, hardcoded path, or global registry handle was introduced.

## Residual Risks

- Rich rendering details may remain slightly implementation-specific, so tests should stay focused on classification, not pixel-perfect output.
- This scope intentionally does not add manifest/hash freshness tracking; correctness depends on reading the current DuckDB snapshot at runtime.
