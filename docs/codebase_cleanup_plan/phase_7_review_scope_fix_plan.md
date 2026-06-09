# Phase 7 review scope fix plan

## 1. Summary

Goal: make the Phase 7 review-fixes implementation mergeable by keeping only the two approved behavior fixes and removing or isolating out-of-scope changes.

Main risk: the core fixes are correct, but the implementation becomes unreviewable because dashboard files, adjacent cleanup, or unrelated broad worktree changes are included as evidence for this plan.

## 2. Pipeline Map

1. Pre-flight worktree recording
   - Input: current `git status --short`
   - Output: list of pre-existing dirty files
   - Consumer: final receipt and changed-file audit

2. Test-evaluation registry read path
   - CLI args provide `results_dir`
   - `run_test_evaluation()` derives `registry_path = Path(results_dir) / "shared_registry.duckdb"`
   - `SharedRegistry.open_existing(registry_path)` opens the existing registry
   - `collect_candidates()` and baseline read consume that registry snapshot
   - Output remains `results_dir / "bics" / "top_models_test.json"`

3. Monitor CLI status rendering path
   - `load_registry()` reads `results_dir / "shared_registry.duckdb"`
   - `build_summary_stats()` consumes `client_entries`
   - `build_client_table()` consumes `client_entries`
   - `complete_no_success` is terminal and non-error in both consumers

4. Scope compliance path
   - Implementation changes are limited to allowed files
   - Tests are limited to allowed test files
   - Forbidden paths are audited with `git diff --name-only -- <path>`
   - Any changed file outside the allowed list makes the implementation incomplete unless the user explicitly approves a scope change

## 3. Contract Matrix

| Asset / Interface | Canonical Source Of Truth | Producer / Writer | Readers / Consumers | Required Behavior | Validation / Check | Positive Test | Negative Test | Forbidden Fallback / Shortcut | Verification Command / Review Check |
|---|---|---|---|---|---|---|---|---|---|
| `run_test_evaluation()` registry handoff | Runtime `results_dir / "shared_registry.duckdb"` | Upstream distributed run | `run_test_evaluation()`, `collect_candidates()`, baseline read | Must call `SharedRegistry.open_existing(registry_path)` with the explicit runtime path; must not call `SharedRegistry(registry_path)` | Mock constructor to fail and assert `open_existing` receives exact temp path | `test_run_test_evaluation_opens_existing_registry_read_only` | `test_run_test_evaluation_does_not_use_default_results_dir` | Constructor path, default results dir, hardcoded path, JSON fallback, global registry handle | `grep -n "SharedRegistry" gecco/cli/run_test_evaluation.py`; targeted pytest |
| Monitor `complete_no_success` status classification | `client_entries[*].status` from registry snapshot | Registry producers | `build_summary_stats()`, `build_client_table()` | `complete_no_success` counts as complete and renders terminal/non-error | Assert summary count and table status style/classification | `test_monitor_summary_counts_complete_no_success_as_completed`; `test_monitor_client_table_treats_complete_no_success_as_terminal` | Table test must fail if status is red/error or excluded from terminal path | Treating as error, unknown, non-terminal, or excluding from complete count | `grep -n "complete_no_success" gecco/cli/monitor_distributed.py tests/test_phase7_duckdb_coordination_and_status_views.py`; targeted pytest |
| Scope firewall | Allowed-file list in this plan | Implementer | Reviewer, final receipt | Only approved implementation/test files may be changed for this plan | Changed-file audit against allowed and forbidden paths | `git diff --name-only` contains only allowed plan files, excluding pre-existing dirty files | `git diff --name-only -- gecco-mh-dashboard` must be empty for this plan | Dashboard edits, docs edits, scripts edits, schema changes, broad cleanup, opportunistic refactors | `git diff --name-only`; `git diff --name-only -- gecco-mh-dashboard`; `git diff --name-only -- docs bash scripts config README.md` |

## 4. Scope Firewall

### Allowed Implementation Files

- `gecco/cli/run_test_evaluation.py`
- `gecco/cli/monitor_distributed.py`

### Allowed Test Files

- `tests/test_phase5_duckdb_canonical_state.py`
- `tests/test_phase7_duckdb_coordination_and_status_views.py`

### Forbidden Files And Directories

- `gecco-mh-dashboard/**`
- `docs/**`, except this plan file
- `bash/**`
- `scripts/**`
- `config/**`
- `README.md`
- `.claude/**`
- `gecco/coordination.py`
- `gecco/diagnostic_store/**`
- unrelated `gecco/**` modules
- unrelated `tests/**` files

### Tempting But Forbidden Adjacent Fixes

- Dashboard DuckDB consistency fixes
- Dashboard tests
- Registry schema changes
- Diagnostic store schema changes
- CLI script cleanup
- Documentation updates beyond this plan file
- Config normalization
- CRLF or formatting-only edits
- Removing legacy scripts
- Broad exception-handling changes unrelated to the two contracts

### Explicit Non-Goals

- No dashboard behavior changes
- No dashboard tests
- No schema migration or schema redesign
- No compatibility wrappers
- No JSON fallback
- No manifest, hash, timestamp, or freshness mechanism
- No cleanup of unrelated dirty files
- No opportunistic refactors

### If Forbidden Files Appear Necessary

Stop and ask the user for an explicit scope change. Do not touch the forbidden file. A changed file outside the allowed lists makes the implementation incomplete unless the user explicitly approves that scope change.

### Scope Review Commands

```bash
git status --short
git diff --name-only
git diff --name-only -- gecco-mh-dashboard
git diff --name-only -- docs bash scripts config README.md .claude
git diff --name-only -- gecco/coordination.py gecco/diagnostic_store
```

## 5. Test Inventory

| Contract Row | Positive Proof | Negative Proof | Review / Command |
|---|---|---|---|
| `run_test_evaluation()` registry handoff | `test_run_test_evaluation_opens_existing_registry_read_only` | `test_run_test_evaluation_does_not_use_default_results_dir` | `conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q -k "run_test_evaluation"` |
| Monitor status classification | `test_monitor_summary_counts_complete_no_success_as_completed`; `test_monitor_client_table_treats_complete_no_success_as_terminal` | Same table test must fail if status falls into red/error path; summary test must fail if excluded from complete count | `conda run -n gecco_mh pytest tests/test_phase7_duckdb_coordination_and_status_views.py -q -k "complete_no_success or monitor"` |
| Scope firewall | Changed-file audit contains only allowed files for this plan | Forbidden-path diffs are empty for this plan | `git diff --name-only -- gecco-mh-dashboard`; `git diff --name-only -- docs bash scripts config README.md .claude` |

Schema and consumer-alignment tests are not required because this plan does not add or mirror a structured downstream schema.

Freshness or provenance checks are not required because this plan does not create a derived registry artifact. Correctness is proven by reading the runtime `results_dir/shared_registry.duckdb` path on each call.

## 6. Failure Cases To Prevent

- `run_test_evaluation()` checks that a registry file exists but then opens a default/global registry.
- A test only checks command strings while runtime still uses `SharedRegistry(...)`.
- Tests mock away `SharedRegistry.open_existing()` so the path handoff is unobservable.
- `complete_no_success` is counted as neither running nor complete.
- `complete_no_success` is rendered as red/error.
- Dashboard files are touched because they also mention `shared_registry.duckdb`.
- Extra broad cleanup is included with the two focused fixes.
- A dirty pre-existing worktree makes unrelated changes look like part of this plan.
- A negative test passes because an unrelated validation error fires before the intended contract is exercised.

## 7. Implementation Phases

### Phase 0: Pre-Flight Worktree Recording

Goal: separate pre-existing dirt from this plan's changes.

Files allowed to change: none.

Required steps:

1. Run:

```bash
git status --short
git diff --name-only
```

2. Record all dirty files as pre-existing.
3. Do not revert or modify unrelated pre-existing changes.
4. Do not count pre-existing dirty files as evidence that this plan changed them.

Phase gate:

- Do not proceed unless the implementer has recorded the pre-flight dirty-file list.
- Final receipt must distinguish "changed for this plan" from "pre-existing dirty."

### Phase 1: Keep Only The Test-Evaluation Registry Fix

Goal: ensure `run_test_evaluation()` opens the exact runtime registry read-only through `SharedRegistry.open_existing`.

Files allowed to change:

- `gecco/cli/run_test_evaluation.py`
- `tests/test_phase5_duckdb_canonical_state.py`

Files forbidden:

- Everything else.

Required implementation details:

- Derive `registry_path = Path(results_dir) / "shared_registry.duckdb"`.
- Call exactly `SharedRegistry.open_existing(registry_path)`.
- Do not instantiate `SharedRegistry(registry_path)` on this path.
- Do not add fallback paths.
- Do not add JSON fallback.
- Do not add schema initialization.

Tests to add or keep first:

- `test_run_test_evaluation_opens_existing_registry_read_only`
- `test_run_test_evaluation_does_not_use_default_results_dir`

Negative test requirement:

- The default-path test must use a temporary non-default `results_dir`.
- It must fail if `run_test_evaluation()` reads a default/global path instead of the explicit path.
- It must not pass merely because config loading, split loading, or candidate collection failed first.

Tests to run:

```bash
conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q -k "run_test_evaluation"
```

Review checks:

```bash
grep -n "SharedRegistry" gecco/cli/run_test_evaluation.py
git diff --name-only
```

Do not proceed unless:

- Targeted tests pass.
- Review shows `open_existing(registry_path)`.
- No fallback constructor exists in `run_test_evaluation()`.
- Changed files for this phase are allowed.

### Phase 2: Keep Only The Monitor Status Fix

Goal: treat `complete_no_success` as terminal/non-error in CLI-only monitor summary and table.

Files allowed to change:

- `gecco/cli/monitor_distributed.py`
- `tests/test_phase7_duckdb_coordination_and_status_views.py`

Files forbidden:

- `gecco-mh-dashboard/**`
- all other files not listed above.

Required implementation details:

- In `build_summary_stats()`, count statuses in `{"complete", "complete_no_success"}` as complete.
- In `build_client_table()`, render statuses in `{"complete", "complete_no_success"}` with the terminal/success styling path.
- Keep the change local to the CLI monitor.
- Do not change dashboard status logic.
- Do not change registry schema or producers.

Tests to add or keep first:

- `test_monitor_summary_counts_complete_no_success_as_completed`
- `test_monitor_client_table_treats_complete_no_success_as_terminal`

Negative test requirement:

- The summary test must fail if `complete_no_success` is excluded from completed count.
- The table test must fail if `complete_no_success` falls through to red/error styling.

Tests to run:

```bash
conda run -n gecco_mh pytest tests/test_phase7_duckdb_coordination_and_status_views.py -q -k "complete_no_success or monitor"
```

Review checks:

```bash
grep -n "complete_no_success" gecco/cli/monitor_distributed.py tests/test_phase7_duckdb_coordination_and_status_views.py
git diff --name-only -- gecco-mh-dashboard
```

Do not proceed unless:

- Targeted tests pass.
- `complete_no_success` appears only in allowed implementation/test files for this plan.
- Dashboard diff for this plan is empty.
- Changed files for this phase are allowed.

### Phase 3: Scope Cleanup And Final Audit

Goal: remove any plan-owned out-of-scope changes or identify them as pre-existing/unapproved.

Files allowed to change:

- Only allowed files listed in the scope firewall.

Files explicitly forbidden:

- `gecco-mh-dashboard/**`
- `docs/**`, except this plan file
- `bash/**`
- `scripts/**`
- `config/**`
- `README.md`
- `.claude/**`
- unrelated `gecco/**`
- unrelated `tests/**`

Required implementation details:

- If out-of-scope files were modified by this plan, remove those plan-owned changes.
- If out-of-scope files were already dirty before implementation, leave them untouched and report them as pre-existing.
- Do not fix dashboard or docs to make them consistent with the CLI changes.

Tests to run:

```bash
conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q -k "run_test_evaluation or complete_no_success or monitor_distributed"
conda run -n gecco_mh pytest tests/test_phase7_duckdb_coordination_and_status_views.py -q
conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q
conda run -n gecco_mh pytest tests/test_cmg_registry.py -q
git diff --check
```

Review checks:

```bash
git diff --name-only
git diff --name-only -- gecco-mh-dashboard
git diff --name-only -- docs bash scripts config README.md .claude
git diff --name-only -- gecco/coordination.py gecco/diagnostic_store
```

Do not proceed unless:

- All targeted tests pass.
- `git diff --check` passes.
- Every file changed for this plan is in the allowed list.
- Any dirty forbidden files are explicitly reported as pre-existing or require user-approved scope expansion.

## 8. Forbidden Patterns

- Hardcoded registry paths
- Default results directory fallback
- Hidden global registry handles
- Accepting both correct and incorrect registry locations
- `SharedRegistry(registry_path)` in `run_test_evaluation()` read-only path
- JSON fallback
- Schema initialization on the post-processing read path
- Tests that only check command strings
- Tests that mock away the registry-opening behavior they are meant to prove
- Tests that assert only that an error happened without asserting the specific contract
- Dashboard edits
- Documentation edits beyond this plan file
- Script cleanup
- Formatting-only edits
- Adjacent migrations
- Broad exception handling unrelated to the two named fixes
- Compatibility wrappers added for convenience

## 9. Final Verification

Run:

```bash
conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q -k "run_test_evaluation or complete_no_success or monitor_distributed"
conda run -n gecco_mh pytest tests/test_phase7_duckdb_coordination_and_status_views.py -q
conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q
conda run -n gecco_mh pytest tests/test_cmg_registry.py -q
git diff --check
```

Manual checks:

```bash
grep -n "SharedRegistry" gecco/cli/run_test_evaluation.py
grep -n "complete_no_success" gecco/cli/monitor_distributed.py tests/test_phase7_duckdb_coordination_and_status_views.py
git diff --name-only
git diff --name-only -- gecco-mh-dashboard
git diff --name-only -- docs bash scripts config README.md .claude
git diff --name-only -- gecco/coordination.py gecco/diagnostic_store
```

Required implementation verification receipt:

- Exact tests and commands run.
- Which contract rows each command covers.
- Any contract rows not verified and why.
- Tests expected to fail before the fix and pass after the fix.
- Manual review checks performed, including grep output summaries.
- Pre-flight `git status --short` summary.
- Whether the worktree was already dirty.
- Final changed-file list for this plan.
- Confirmation that every changed file for this plan is allowed by the scope firewall.
- Forbidden-path audit summaries.
- Any tempting adjacent fixes intentionally deferred.
- Statement that no dashboard files were changed for this plan, or explicit user approval if they were.

Work is incomplete if the receipt is missing, vague, or omits any contract row.

### Final Acceptance Checklist

- `run_test_evaluation()` calls `SharedRegistry.open_existing(results_dir / "shared_registry.duckdb")`.
- `run_test_evaluation()` does not call `SharedRegistry(...)` for the read-only registry handoff.
- A temp non-default `results_dir` test proves the default results location is not silently used.
- `build_summary_stats()` counts `complete_no_success` as complete.
- `build_client_table()` renders `complete_no_success` through the terminal/non-error path.
- Targeted Phase 5, Phase 7, and CMG registry tests pass.
- `git diff --check` passes.
- Changed files for this plan are limited to the allowed implementation and test files.
- `git diff --name-only -- gecco-mh-dashboard` has no plan-owned changes.
- No docs, scripts, config, dashboard, schema, or unrelated cleanup changes are included as part of this plan.

## 10. Residual Risks

- Rich table assertions may remain somewhat tied to Rich internals.
- A broad pre-existing dirty worktree can still complicate review, so the final receipt must clearly separate plan-owned changes from pre-existing changes.
- This plan intentionally does not validate dashboard consistency because dashboard work is explicitly out of scope.
