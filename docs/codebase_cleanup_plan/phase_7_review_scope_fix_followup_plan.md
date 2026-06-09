# Phase 7 review scope fix follow-up plan

## 1. Summary

Goal: address the review findings in `docs/codebase_cleanup_plan/phase_7_review_scope_fix_findings.md` without widening the original Phase 7 review-fixes scope.

The required work is deliberately small:

- Strengthen the weak `run_test_evaluation()` negative test so it fails for both default-path drift and constructor use.
- Remove the adjacent `duckdb.Error` monitor exception-handling change if it is plan-owned, or stop for user clarification if it is pre-existing/user-owned.
- Produce a persisted implementation receipt that separates current pre-existing dirty files from files changed for this follow-up.

Main risk: the worktree is already broadly dirty, so unrelated dashboard, docs, scripts, schema, config, and cleanup changes can be mistaken for this plan's implementation. This plan makes that visible through pre-flight recording, changed-file audits, and a required receipt.

## 2. Pipeline Map

### A. Worktree provenance pipeline

`git status --short` -> current dirty-file baseline -> implement only allowed follow-up edits -> final `git status --short` -> persisted receipt.

Producer-consumer handoff trace:

- writer/source of truth: current Git worktree at the start of this follow-up
- reader: implementer and reviewer
- same source of truth: `git status --short` and `git diff --name-only`
- drift check: final receipt must distinguish files changed for this follow-up from files already dirty at follow-up pre-flight

This cannot reconstruct the missing historical pre-flight from the earlier implementation. The receipt must explicitly say that the historical pre-flight is unavailable and that the current baseline is only for this follow-up.

### B. Test-evaluation negative-test pipeline

`tmp_path explicit results_dir` -> `registry_path = explicit_results/shared_registry.duckdb` -> patched `SharedRegistry.open_existing(registry_path)` -> `run_test_evaluation()` -> assertions that constructor was not called and `open_existing()` used the explicit path exactly once.

Producer-consumer handoff trace:

- writer/source of truth: test-created `registry_path` under a non-default temp `results_dir`
- reader: `run_test_evaluation()`
- same source of truth: runtime `results_dir` argument
- drift test: `test_run_test_evaluation_does_not_use_default_results_dir` must fail if runtime uses a default path, a global/stale path, or the constructor path

### C. Monitor exception-handling scope pipeline

`load_registry()` -> `SharedRegistry.open_existing(path).read()` -> unavailable registry handling.

This follow-up is not allowed to add or keep broad adjacent monitor resilience behavior as plan-owned work. The only monitor behavior in the original contract is `complete_no_success` classification in `build_summary_stats()` and `build_client_table()`.

Producer-consumer handoff trace:

- writer/source of truth: registry producers write `shared_registry.duckdb`
- reader: monitor CLI `load_registry()` and status-rendering helpers
- same source of truth: `client_entries` snapshot
- drift check: grep/review proves only classification logic changed for this plan; `duckdb.Error` handling is either absent from the plan-owned diff or explicitly documented as pre-existing/user-approved

## 3. Contract Matrix

| Asset / Interface | Canonical Location Or Source Of Truth | Writer / Producer | Readers / Consumers | Required Contents, Schema, Or Behavior | Validation Function Or Check | Positive Test | Negative Test | Forbidden Fallback Or Shortcut | Named Verification Command Or Review Check |
|---|---|---|---|---|---|---|---|---|---|
| `test_run_test_evaluation_does_not_use_default_results_dir` | `tests/test_phase5_duckdb_canonical_state.py` | Test author | pytest, reviewer | Uses a temp non-default `results_dir`; asserts `SharedRegistry.open_existing(registry_path)` exactly once; makes `SharedRegistry(...)` constructor use fail; fails if default/global/stale path is used | Test assertions on `registry_cls.open_existing` and `registry_cls.assert_not_called()` | `test_run_test_evaluation_opens_existing_registry_read_only` | `test_run_test_evaluation_does_not_use_default_results_dir` | Existence-only path check; default results dir fallback; constructor path; hidden global registry handle | `conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q -k "run_test_evaluation"` |
| Monitor exception-handling scope | `gecco/cli/monitor_distributed.py::load_registry()` | Monitor CLI implementation | Monitor CLI, tests | This follow-up must not include plan-owned `duckdb.Error` import/catch or invalid-DuckDB resilience tests unless user explicitly approves scope expansion | Grep/diff check for `import duckdb` and `duckdb.Error`; changed-file receipt classifies any remaining occurrence as pre-existing/user-approved | Existing monitor classification tests continue to pass | Review check fails if plan-owned `duckdb.Error` handling remains without approval | Broad exception handling; adjacent resilience behavior; compatibility fallback | `grep -n "duckdb\|duckdb.Error" gecco/cli/monitor_distributed.py tests/test_phase5_duckdb_canonical_state.py`; `git diff -- gecco/cli/monitor_distributed.py tests/test_phase5_duckdb_canonical_state.py` |
| Implementation receipt | `docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md` | Implementer | Reviewer, future implementers | Persist exact commands run, contract coverage, unverified rows, current pre-flight dirty baseline, final changed-file list for this follow-up, forbidden-path audit summaries, and deferred adjacent fixes | Manual review of receipt against this contract matrix | Receipt lists all commands and contract rows | Receipt is missing, vague, or omits any contract row | Chat-only receipt; ambiguous "tests passed" without commands; claiming forbidden dirty files are plan-owned/clean without evidence | `test -f docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md`; manual review of receipt |
| Scope firewall | Allowed-file list in this plan | Implementer | Reviewer | Files changed for this follow-up must be limited to allowed implementation, test, and receipt files; all other dirty files must be listed as pre-existing current-baseline dirt or require explicit user approval | `git diff --name-only`; forbidden-path audits; receipt comparison to pre-flight baseline | Allowed changed-file audit passes | Forbidden-path audit has follow-up-owned changes | Dashboard/docs/scripts/config/schema cleanup; broad test rewrites; opportunistic refactors | `git diff --name-only`; `git diff --name-only -- gecco-mh-dashboard`; `git diff --name-only -- docs bash scripts config README.md .claude`; `git diff --name-only -- gecco/coordination.py gecco/diagnostic_store` |

Schema and consumer-alignment tests are not required because this follow-up does not add a validator, schema, migration, or consumer-mirrored structured file.

Freshness/provenance mechanisms are not required because this follow-up does not generate downstream artifacts from upstream inputs. The only provenance needed is worktree provenance through pre-flight/final Git status and the persisted receipt.

## 4. Scope Firewall

### Allowed Implementation Files

- `gecco/cli/monitor_distributed.py`, only to remove the plan-owned `duckdb` import / `duckdb.Error` catch if confirmed not pre-existing or user-approved.

### Allowed Test Files

- `tests/test_phase5_duckdb_canonical_state.py`, only to strengthen `test_run_test_evaluation_does_not_use_default_results_dir` and remove any plan-owned invalid/unreadable DuckDB monitor tests if they only support the out-of-scope `duckdb.Error` change.

### Allowed Documentation / Receipt Files

- `docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_plan.md`
- `docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md`

### Explicitly Forbidden Files And Directories

- `gecco-mh-dashboard/**`
- `docs/**`, except the two allowed follow-up files above
- `bash/**`
- `scripts/**`
- `config/**`
- `README.md`
- `.claude/**`
- `gecco/coordination.py`
- `gecco/diagnostic_store/**`
- unrelated `gecco/**` modules
- unrelated `tests/**` files
- `tests/test_phase7_duckdb_coordination_and_status_views.py`, unless the user explicitly approves touching the broad Phase 7 test file in this follow-up

### Tempting But Forbidden Adjacent Fixes

- Dashboard DuckDB path/status cleanup
- Dashboard tests
- Registry schema changes
- Diagnostic store schema changes
- Broad Phase 7 test cleanup
- Baseline concurrency rewrites
- Script-helper registry tests
- Monitor invalid-DuckDB resilience work, unless explicitly approved
- Documentation cleanup beyond the receipt
- Config normalization
- CRLF or formatting-only edits
- Removing legacy scripts
- Refactoring Rich table helpers

### Explicit Non-Goals

- Do not reconstruct or rewrite the entire earlier Phase 7 implementation.
- Do not clean the broad dirty worktree.
- Do not revert or modify unrelated pre-existing user changes.
- Do not make dashboard, scripts, docs, schema, or config consistent with this behavior.
- Do not add compatibility layers or fallback paths.
- Do not add manifests, hashes, migrations, or freshness tracking.
- Do not add new broad runtime tests.

### If Scope Is Ambiguous

If a dirty file outside the allowed lists appears to be required, stop and ask the user one concise question. Do not modify the file. A changed file outside the allowed lists makes implementation incomplete unless the user explicitly approves a scope change.

If the implementer cannot determine whether the `duckdb.Error` monitor change and its tests are pre-existing/user-owned or plan-owned, stop and ask the user before removing them.

### Scope Review Commands

Run and summarize:

```bash
git status --short
git diff --name-only
git diff --name-only -- gecco-mh-dashboard
git diff --name-only -- docs bash scripts config README.md .claude
git diff --name-only -- gecco/coordination.py gecco/diagnostic_store
git diff --name-only -- tests/test_phase7_duckdb_coordination_and_status_views.py
```

The receipt must say which outputs are pre-existing current-baseline dirt and which, if any, are changed for this follow-up.

## 5. Test Inventory

| Contract Row | Positive Tests / Checks | Negative Tests / Checks | Verification Command |
|---|---|---|---|
| Strong default-results negative test | `test_run_test_evaluation_opens_existing_registry_read_only` | `test_run_test_evaluation_does_not_use_default_results_dir` with constructor failure and exact `open_existing(registry_path)` assertion | `conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q -k "run_test_evaluation"` |
| Monitor exception-handling scope | Existing monitor classification tests still pass if run | Grep/diff flags plan-owned `duckdb.Error` import/catch unless documented as pre-existing/user-approved | `grep -n "duckdb\|duckdb.Error" gecco/cli/monitor_distributed.py tests/test_phase5_duckdb_canonical_state.py` |
| Receipt | Receipt file exists and maps every contract row to evidence | Missing/vague receipt fails review | `test -f docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md` plus manual review |
| Scope firewall | Changed-file audit contains only allowed follow-up files after subtracting current-baseline dirt | Forbidden-path audits have follow-up-owned changes | Scope review commands listed above |

Behavioral coverage is intentionally limited to the weak negative test because the core runtime implementation already passed the earlier review. Scope and non-goal contracts are verified by changed-file audits and the receipt rather than new runtime tests.

## 6. Failure Cases To Prevent

- A default/global registry path is used but the negative test still passes because it only checked that the default directory exists.
- `SharedRegistry(...)` is used instead of `SharedRegistry.open_existing(...)`, and only the positive test catches it.
- A combined test passes because `load_config`, `load_splits`, or `collect_candidates` fails before registry opening is exercised.
- `duckdb.Error` handling remains in the plan-owned monitor diff even though it is adjacent resilience behavior outside the contract.
- Invalid-DuckDB monitor tests force out-of-scope broad exception handling to remain.
- Dashboard files are changed because they mention `shared_registry.duckdb`.
- Broad dirty worktree changes are presented as if they were verified by this plan.
- The receipt exists only in chat and cannot be reviewed from the repo.
- The receipt omits forbidden-path audit outputs, making scope unverifiable.
- An implementer cleans unrelated broad changes while trying to satisfy the scope firewall.

## 7. Implementation Phases

### Phase 0: Current Worktree Baseline

Goal: record current dirty state for this follow-up before making edits.

Files allowed to change: none.

Files explicitly forbidden: all files.

Files that may look related but must not be touched:

- `gecco-mh-dashboard/**`
- `gecco/coordination.py`
- `gecco/diagnostic_store/**`
- `tests/test_phase7_duckdb_coordination_and_status_views.py`
- docs other than the receipt file

Required steps:

1. Run:

```bash
git status --short
git diff --name-only
git diff --name-only -- gecco-mh-dashboard
git diff --name-only -- docs bash scripts config README.md .claude
git diff --name-only -- gecco/coordination.py gecco/diagnostic_store
git diff --name-only -- tests/test_phase7_duckdb_coordination_and_status_views.py
```

2. Save the outputs or summaries for the final receipt.
3. Mark all currently dirty files as current-baseline pre-existing for this follow-up.
4. State explicitly in the receipt that this does not prove historical pre-flight state for the earlier implementation.

Tests to run: none.

Expected pass/fail behavior: commands should complete; non-empty output is acceptable but must be recorded.

Review checks: the final receipt must include this baseline.

Do not proceed unless: current dirty-file baseline is recorded.

Changed-file audit for the phase: no files changed.

Evidence to report: exact commands run and summary of dirty/forbidden outputs.

### Phase 1: Strengthen The Default-Path Negative Test

Goal: make the `run_test_evaluation()` negative test fail for default-path drift and constructor use.

Files allowed to change:

- `tests/test_phase5_duckdb_canonical_state.py`

Files explicitly forbidden:

- all other files

Files that may look related but must not be touched:

- `gecco/cli/run_test_evaluation.py`, unless the strengthened test fails and the runtime implementation actually needs the already-approved original fix
- `tests/test_phase7_duckdb_coordination_and_status_views.py`
- dashboard tests

Required implementation details:

- In `test_run_test_evaluation_does_not_use_default_results_dir`, keep the temp non-default `results_dir`.
- Keep expensive work patched so the test reaches registry opening and does not fail earlier.
- Set `registry_cls.side_effect = AssertionError("constructor path must not be used")`.
- Use an `_open_existing(path)` side effect that asserts `path == registry_path` and `path != default_results_dir / "shared_registry.duckdb"` before returning `fake_registry`.
- After `run_test_evaluation()`, assert `registry_cls.open_existing.assert_called_once_with(registry_path)`.
- After `run_test_evaluation()`, assert `registry_cls.assert_not_called()`.

Tests to add first or alongside the change:

- Update only `test_run_test_evaluation_does_not_use_default_results_dir`.

Negative test:

- The strengthened test must fail if the implementation uses a default path, global/stale path, or constructor path.
- It must not pass because an unrelated earlier validation error fired.

Phase gate tests:

```bash
conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q -k "run_test_evaluation"
```

Expected pass/fail behavior:

- Before strengthening, the test can pass while missing constructor/open call assertions.
- After strengthening, it passes only if the exact runtime `registry_path` is opened through `open_existing()` and the constructor is unused.

Review checks:

```bash
grep -n "test_run_test_evaluation_does_not_use_default_results_dir" -A60 tests/test_phase5_duckdb_canonical_state.py
git diff -- tests/test_phase5_duckdb_canonical_state.py
git diff --name-only
```

Do not proceed unless:

- Targeted test passes.
- Diff touches only the intended test block, except pre-existing current-baseline changes already recorded.
- No forbidden files were modified.

Changed-file audit for the phase:

```bash
git diff --name-only -- tests/test_phase5_duckdb_canonical_state.py gecco/cli/run_test_evaluation.py gecco-mh-dashboard docs bash scripts config README.md .claude gecco/coordination.py gecco/diagnostic_store
```

Evidence to report:

- exact test command and result
- test block grep summary
- files changed in this phase

### Phase 2: Resolve The Monitor Exception-Handling Scope Finding

Goal: ensure out-of-scope `duckdb.Error` monitor resilience behavior is not plan-owned.

Files allowed to change:

- `gecco/cli/monitor_distributed.py`
- `tests/test_phase5_duckdb_canonical_state.py`, only if removing tests that solely support the out-of-scope monitor resilience change

Files explicitly forbidden:

- all other files

Files that may look related but must not be touched:

- `tests/test_phase7_duckdb_coordination_and_status_views.py`
- dashboard files/tests
- registry schema files

Required implementation details:

- Inspect `gecco/cli/monitor_distributed.py` for `import duckdb` and `except (..., duckdb.Error)`.
- If the implementer can confirm this was introduced as part of the Phase 7 review-scope-fix work and has no explicit user approval, remove `import duckdb` and remove `duckdb.Error` from `load_registry()` exception handling.
- If `duckdb` is only used for this catch, the import must be removed with the catch.
- If tests exist only to prove invalid DuckDB files return `None`, remove those tests from this plan-owned diff or stop and ask if they are user-approved pre-existing work.
- Do not change `complete_no_success` summary/table classification.
- Do not add replacement exception handling.

Tests to add first: none.

Negative checks:

- Grep must not show plan-owned `duckdb.Error` handling after the phase unless the receipt documents explicit user approval or pre-existing ownership.

Phase gate tests:

```bash
conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q -k "run_test_evaluation or monitor_distributed"
conda run -n gecco_mh pytest tests/test_phase7_duckdb_coordination_and_status_views.py -q -k "complete_no_success or monitor"
```

Expected pass/fail behavior:

- Tests for the approved contracts should pass.
- If removing out-of-scope invalid-DuckDB behavior causes unrelated tests to fail, stop and ask whether that behavior is approved scope rather than adding fallback handling.

Review checks:

```bash
grep -n "duckdb\|duckdb.Error" gecco/cli/monitor_distributed.py tests/test_phase5_duckdb_canonical_state.py
grep -n "complete_no_success" gecco/cli/monitor_distributed.py tests/test_phase7_duckdb_coordination_and_status_views.py
git diff -- gecco/cli/monitor_distributed.py tests/test_phase5_duckdb_canonical_state.py
```

Do not proceed unless:

- The out-of-scope exception-handling finding is resolved by removal, explicit user approval, or documented pre-existing ownership.
- Approved monitor classification tests still pass.
- No forbidden files were modified.

Changed-file audit for the phase:

```bash
git diff --name-only -- gecco/cli/monitor_distributed.py tests/test_phase5_duckdb_canonical_state.py gecco-mh-dashboard docs bash scripts config README.md .claude gecco/coordination.py gecco/diagnostic_store
```

Evidence to report:

- whether `duckdb.Error` handling was removed, retained as pre-existing, or retained with explicit user approval
- grep output summary
- exact test commands and results
- files changed in this phase

### Phase 3: Persist The Follow-Up Receipt

Goal: make the implementation receipt reviewable from the repository.

Files allowed to change:

- `docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md`

Files explicitly forbidden:

- all other files

Required implementation details:

Create or update `docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md` with these sections:

- Pre-flight current-baseline `git status --short` summary.
- Statement that historical pre-flight for the earlier implementation is unavailable unless the user provides it.
- Files changed for this follow-up.
- Files already dirty at follow-up pre-flight and left untouched.
- Exact commands/tests run.
- Contract rows covered by each command.
- Contract rows not verified and why.
- Tests expected to fail before the fix and pass after.
- Manual review checks and grep summaries.
- Forbidden-path audit outputs or summaries.
- Confirmation every file changed for this follow-up is allowed by this scope firewall.
- Adjacent fixes intentionally deferred.

Tests to add first: none.

Negative checks:

- Receipt is incomplete if it omits any contract row or any forbidden-path audit summary.

Phase gate checks:

```bash
test -f docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md
git diff -- docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md
```

Expected pass/fail behavior:

- Receipt file exists and contains concrete commands/results, not vague prose.

Review checks:

```bash
grep -n "Contract\|Pre-flight\|Forbidden-path\|Files changed\|Commands" docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md
git diff --name-only
```

Do not proceed unless:

- Receipt exists.
- Receipt maps every contract row to evidence or a justified gap.
- Receipt clearly separates current-baseline dirty files from follow-up changes.

Changed-file audit for the phase:

```bash
git diff --name-only -- docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md docs bash scripts config README.md .claude gecco-mh-dashboard gecco/coordination.py gecco/diagnostic_store
```

Evidence to report:

- receipt path
- sections completed
- any contract rows not verified

### Phase 4: Final Verification And Scope Audit

Goal: prove the follow-up contracts and scope limits are satisfied.

Files allowed to change: none, except fixing a phase-gate failure in an allowed file from the relevant earlier phase.

Files explicitly forbidden:

- all forbidden files listed in the Scope Firewall

Tests to run:

```bash
conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q -k "run_test_evaluation or monitor_distributed"
conda run -n gecco_mh pytest tests/test_phase7_duckdb_coordination_and_status_views.py -q -k "complete_no_success or monitor"
git diff --check
```

Recommended broader checks if feasible:

```bash
conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q
conda run -n gecco_mh pytest tests/test_phase7_duckdb_coordination_and_status_views.py -q
conda run -n gecco_mh pytest tests/test_cmg_registry.py -q
```

Review checks:

```bash
grep -n "SharedRegistry" gecco/cli/run_test_evaluation.py
grep -n "test_run_test_evaluation_does_not_use_default_results_dir" -A60 tests/test_phase5_duckdb_canonical_state.py
grep -n "duckdb\|duckdb.Error" gecco/cli/monitor_distributed.py tests/test_phase5_duckdb_canonical_state.py
grep -n "complete_no_success" gecco/cli/monitor_distributed.py tests/test_phase7_duckdb_coordination_and_status_views.py
git status --short
git diff --name-only
git diff --name-only -- gecco-mh-dashboard
git diff --name-only -- docs bash scripts config README.md .claude
git diff --name-only -- gecco/coordination.py gecco/diagnostic_store
git diff --name-only -- tests/test_phase7_duckdb_coordination_and_status_views.py
```

Expected pass/fail behavior:

- Required targeted tests pass.
- `git diff --check` passes.
- Forbidden-path audits may be non-empty because of current-baseline dirty files, but the receipt must identify them as pre-existing and not follow-up-owned.
- Any new follow-up-owned forbidden-file change fails the phase.

Do not proceed unless:

- Required tests/checks are run or explicitly justified as not run.
- Receipt is updated with final command outputs and changed-file audit.
- Every follow-up-owned changed file is allowed by this plan.

Changed-file audit for the phase:

```bash
git diff --name-only
git status --short
```

Evidence to report:

- exact commands and results
- final status summary
- final allowed follow-up changed files
- forbidden-path audit summaries

## 8. Forbidden Patterns

- Hardcoded registry paths.
- Default results directory fallback.
- Hidden global registry handles.
- Accepting both correct and incorrect registry locations.
- `SharedRegistry(registry_path)` in the `run_test_evaluation()` read-only path.
- Existence-only validation in the default-path negative test.
- Tests that only check command strings when runtime behavior matters.
- Tests that mock away `SharedRegistry.open_existing()` so the path handoff is unobservable.
- Tests that pass because an unrelated earlier error fires.
- Broad exception handling added for monitor convenience.
- Compatibility wrappers or fallback behavior.
- Dashboard edits.
- Documentation edits beyond the allowed follow-up plan and receipt files.
- Script cleanup.
- Config cleanup.
- Schema changes or migrations.
- Formatting-only edits or CRLF churn.
- Opportunistic refactors.
- Cleaning unrelated broad dirty worktree files.

## 9. Final Verification

Required commands:

```bash
conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q -k "run_test_evaluation or monitor_distributed"
conda run -n gecco_mh pytest tests/test_phase7_duckdb_coordination_and_status_views.py -q -k "complete_no_success or monitor"
git diff --check
```

Recommended commands if feasible:

```bash
conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q
conda run -n gecco_mh pytest tests/test_phase7_duckdb_coordination_and_status_views.py -q
conda run -n gecco_mh pytest tests/test_cmg_registry.py -q
```

Manual review commands:

```bash
grep -n "SharedRegistry" gecco/cli/run_test_evaluation.py
grep -n "test_run_test_evaluation_does_not_use_default_results_dir" -A60 tests/test_phase5_duckdb_canonical_state.py
grep -n "duckdb\|duckdb.Error" gecco/cli/monitor_distributed.py tests/test_phase5_duckdb_canonical_state.py
grep -n "complete_no_success" gecco/cli/monitor_distributed.py tests/test_phase7_duckdb_coordination_and_status_views.py
test -f docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md
git status --short
git diff --name-only
git diff --name-only -- gecco-mh-dashboard
git diff --name-only -- docs bash scripts config README.md .claude
git diff --name-only -- gecco/coordination.py gecco/diagnostic_store
git diff --name-only -- tests/test_phase7_duckdb_coordination_and_status_views.py
```

Final acceptance checklist:

- `test_run_test_evaluation_does_not_use_default_results_dir` asserts exact `open_existing(registry_path)` use.
- `test_run_test_evaluation_does_not_use_default_results_dir` fails if `SharedRegistry(...)` constructor is called.
- `test_run_test_evaluation_does_not_use_default_results_dir` uses a temp non-default `results_dir` and does not pass through an earlier unrelated failure.
- Plan-owned `duckdb.Error` monitor exception handling is removed, or retained only with explicit user approval/pre-existing ownership documented in the receipt.
- Approved `complete_no_success` monitor classification remains intact.
- Receipt file exists at `docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md`.
- Receipt maps each contract row to commands/checks or a justified gap.
- Receipt includes current pre-flight dirty baseline and final changed-file list.
- Receipt clearly says historical pre-flight for the earlier implementation is unavailable, if still unavailable.
- Every file changed for this follow-up is allowed by this plan.
- Forbidden-path audits are summarized, and any non-empty output is classified as current-baseline pre-existing or user-approved.
- No dashboard, schema, scripts, config, README, `.claude`, or unrelated tests/files are modified for this follow-up.

Required implementation verification receipt:

- Exact tests and commands run.
- Which contract rows each command covers.
- Any contract rows not verified and why.
- Any tests expected to fail before the fix and pass after the fix.
- Manual review checks performed, including grep output summaries.
- Current pre-flight `git status --short` summary and whether the worktree was already dirty.
- Final changed-file list for this follow-up.
- Confirmation that every file changed for this follow-up is listed as allowed by the scope firewall.
- Output summaries from forbidden-path audits.
- Any tempting adjacent fixes intentionally deferred because they were out of scope.

Work is incomplete if the receipt is missing, vague, or omits a contract row.

## 10. Residual Risks

- The missing historical pre-flight state from the earlier implementation cannot be reconstructed from Git alone. This follow-up can only establish a current-baseline receipt.
- The worktree is broadly dirty, so review still requires discipline to separate current-baseline dirt from follow-up edits.
- If the `duckdb.Error` monitor behavior was intentionally introduced by the user before this follow-up, removing it would be wrong. The implementer must ask before removing it if ownership is unclear.
- Broad Phase 7 tests in `tests/test_phase7_duckdb_coordination_and_status_views.py` remain out of scope for this follow-up unless explicitly approved.
