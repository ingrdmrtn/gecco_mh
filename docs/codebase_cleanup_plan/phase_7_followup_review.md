# Phase 7 Follow-Up Review

Source of truth: `docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_plan.md`.

## 1. Findings

### High: Receipt contract is not implemented

- `docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md` is missing.
- Violates Contract Matrix row `Implementation receipt`, Phase 3, Phase 4, and Final Acceptance Checklist items 563-566.
- Evidence: `Read` returned file-not-found; `test -f docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md` produced no success output and the file is absent from `glob docs/codebase_cleanup_plan/*receipt*.md`.
- Impact: pre-flight baseline, final changed-file ownership, command evidence, contract coverage, unverified rows, and forbidden-path audit summaries are not persisted. The reviewer cannot distinguish follow-up-owned changes from current-baseline dirt using the required artifact.

### High: Required negative test is still weak

- `tests/test_phase5_duckdb_canonical_state.py:321-359` does not satisfy the required implementation details for `test_run_test_evaluation_does_not_use_default_results_dir`.
- Violates Contract Matrix row `test_run_test_evaluation_does_not_use_default_results_dir` and Phase 1 lines 235-240.
- Missing required `registry_cls.side_effect = AssertionError("constructor path must not be used")`.
- Missing required assertion `registry_cls.open_existing.assert_called_once_with(registry_path)` after `run_test_evaluation()`.
- Missing required assertion `registry_cls.assert_not_called()` after `run_test_evaluation()`.
- `_open_existing(path)` asserts `path == registry_path`, but does not assert `path != default_results_dir / "shared_registry.duckdb"` as required.
- The test still ends with only `assert default_results_dir.exists()` and `assert result is None`, so it does not prove the constructor path is forbidden or that `open_existing()` was called exactly once.

Relevant snippet:

```python
def _open_existing(path):
    assert path == registry_path
    return fake_registry

...
registry_cls.open_existing.side_effect = _open_existing
...
assert default_results_dir.exists()
assert result is None
```

### High: Out-of-scope `duckdb.Error` monitor handling remains plan-owned in the diff

- `gecco/cli/monitor_distributed.py:10` imports `duckdb` and `gecco/cli/monitor_distributed.py:43` catches `duckdb.Error`.
- Violates Contract Matrix row `Monitor exception-handling scope`, Scope Firewall lines 71 and 106, Phase 2 lines 309-314, and Forbidden Pattern `Broad exception handling added for monitor convenience`.
- The diff shows this as an added change from HEAD:

```diff
+import duckdb
...
-    except (FileNotFoundError, OSError):
+    except (FileNotFoundError, OSError, duckdb.Error):
```

- No receipt exists to document explicit user approval or pre-existing ownership, so this must be treated as an unresolved scope violation under the plan.

### High: Tests supporting the forbidden monitor resilience behavior remain

- `tests/test_phase5_duckdb_canonical_state.py:260-281` adds invalid/unreadable DuckDB monitor tests.
- Violates Phase 2 lines 312 and 320, Test Inventory row `Monitor exception-handling scope`, and Failure Cases 160-161.
- These tests appear solely to support `load_registry()` returning `None` for invalid/unreadable DuckDB files, which the plan identifies as out-of-scope monitor resilience unless user-approved. No receipt documents approval or pre-existing ownership.

### Medium: Allowed test file contains broad changes beyond the follow-up scope

- `tests/test_phase5_duckdb_canonical_state.py` is allowed only to strengthen `test_run_test_evaluation_does_not_use_default_results_dir` and remove plan-owned invalid-DuckDB monitor tests.
- The diff also changes baseline tests, concurrency behavior, script-helper coverage, dashboard discovery assertions, and imports: lines 3-49, 157-257, 362-389, and 498-509.
- Violates Scope Firewall line 75, Phase 1 lines 242-244, Phase 1 gate line 273, Phase 2 files/purpose limits, and Forbidden Patterns `Script cleanup`, `Dashboard edits`, `Opportunistic refactors`, and `Broad test rewrites`.
- Without the missing receipt, these cannot be classified as pre-existing current-baseline dirt or user-approved work.

### Medium: Scope firewall cannot be verified because changed-file ownership is not documented

- `git status --short` shows many dirty forbidden paths, including `.claude/settings.json`, `README.md`, `bash/**`, `config/**`, `gecco-mh-dashboard/**`, `gecco/coordination.py`, `gecco/diagnostic_store/schema.py`, `scripts/**`, unrelated `gecco/**`, and unrelated `tests/**`.
- Violates Contract Matrix row `Scope firewall` and Phase 0/Phase 4 requirements because the required receipt does not mark which files were current-baseline pre-existing versus changed for this follow-up.
- The review can observe broad dirt, but cannot validate the required provenance separation.

### Medium: Implementation phases did not satisfy gates

- Phase 0 gate: no persisted current-baseline output exists in the receipt.
- Phase 1 gate: the strengthened negative test is incomplete and the diff touches broad blocks in `tests/test_phase5_duckdb_canonical_state.py`.
- Phase 2 gate: `duckdb.Error` handling remains without documented approval/pre-existing ownership.
- Phase 3 gate: receipt file is missing.
- Phase 4 gate: required tests pass, but scope and receipt gates fail.

## 2. Open Questions

- None block review confidence. Under the plan, missing receipt documentation means remaining `duckdb.Error` handling and broad test changes cannot be accepted as pre-existing/user-approved.

## 3. Verification Summary

Observed commands/checks:

- `git status --short`: broad dirty worktree observed, including many forbidden paths.
- `git diff --name-only`: broad modified/deleted tracked files observed; untracked files are visible only via status/glob.
- `git diff --name-only -- gecco-mh-dashboard`: `gecco-mh-dashboard/app.py`, `gecco-mh-dashboard/dashboard/config.py`.
- `git diff --name-only -- docs bash scripts config README.md .claude`: many forbidden docs/scripts/config/README/.claude changes observed.
- `git diff --name-only -- gecco/coordination.py gecco/diagnostic_store`: `gecco/coordination.py`, `gecco/diagnostic_store/schema.py`.
- `git diff --name-only -- tests/test_phase7_duckdb_coordination_and_status_views.py`: no tracked diff output, but file is untracked in `git status --short`.
- `git diff --check`: no whitespace errors reported; only CRLF warnings were emitted.
- `conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q -k "run_test_evaluation or monitor_distributed"`: `2 passed, 13 deselected`.
- `conda run -n gecco_mh pytest tests/test_phase7_duckdb_coordination_and_status_views.py -q -k "complete_no_success or monitor"`: `6 passed, 12 deselected`.
- `grep -n "duckdb\|duckdb.Error" gecco/cli/monitor_distributed.py tests/test_phase5_duckdb_canonical_state.py`: found `import duckdb`, `duckdb.Error`, and invalid/unreadable monitor tests.
- `grep -n "complete_no_success" gecco/cli/monitor_distributed.py tests/test_phase7_duckdb_coordination_and_status_views.py`: complete-no-success classification references observed.
- `test -f docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md`: receipt absent.

Contract rows verified:

- Runtime `run_test_evaluation()` uses `SharedRegistry.open_existing(registry_path)` at `gecco/cli/run_test_evaluation.py:138-143`.
- Approved `complete_no_success` monitor classification remains present at `gecco/cli/monitor_distributed.py:60-67` and `gecco/cli/monitor_distributed.py:264-268`; targeted tests pass.

Contract rows not verified or failed:

- `test_run_test_evaluation_does_not_use_default_results_dir`: failed contract coverage; missing required assertions.
- `Monitor exception-handling scope`: failed; `duckdb.Error` handling remains in diff without documented approval/pre-existing ownership.
- `Implementation receipt`: failed; receipt missing.
- `Scope firewall`: not verifiable and currently failing as an implementation review gate because ownership separation is missing.

## 4. Scope Audit

Changed files observed from `git status --short`:

- Allowed follow-up files with tracked changes: `gecco/cli/monitor_distributed.py`, `tests/test_phase5_duckdb_canonical_state.py`.
- Required allowed receipt file: missing, `docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md`.
- Reviewer-created report: `docs/codebase_cleanup_plan/phase_7_followup_review.md`.
- Forbidden or suspicious tracked dirty files: `.claude/settings.json`, `README.md`, `bash/*.sh`, `config/*.yaml`, `gecco-mh-dashboard/app.py`, `gecco-mh-dashboard/dashboard/config.py`, `gecco/artifacts.py`, `gecco/candidate_evaluation.py`, `gecco/candidate_generation.py`, `gecco/cli/run_judge_orchestrator.py`, `gecco/construct_feedback/*`, `gecco/coordination.py`, `gecco/diagnostic_store/schema.py`, `gecco/run_gecco.py`, `scripts/*.py`, `tests/fixtures/phase0/legacy_cli_inventory.json`, and many unrelated `tests/*.py`.
- Forbidden or suspicious untracked files/directories: `config/__init__.py`, broad `docs/**`, `gecco/__main__.py`, multiple `gecco/cli/*.py`, `test_ppc_speedup.py`, `tests/test_phase1_docs_reset.py`, `tests/test_phase2_cli_contract.py`, and `tests/test_phase7_duckdb_coordination_and_status_views.py`.

Allowed-file assessment:

- `gecco/cli/monitor_distributed.py`: file is allowed only to remove plan-owned `duckdb` handling. Actual diff adds/retains `duckdb.Error`; not allowed under the plan without receipt-backed approval.
- `tests/test_phase5_duckdb_canonical_state.py`: file is allowed only for the strengthened negative test and removal of out-of-scope monitor resilience tests. Actual diff includes broad unrelated test changes and keeps out-of-scope monitor resilience tests; not allowed under the plan without receipt-backed approval.
- `docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_plan.md`: allowed plan file exists but is untracked as part of broad `docs/codebase_cleanup_plan/` status.
- `docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md`: allowed and required, but missing.

Forbidden/suspicious changes:

- Dashboard, scripts, config, README, `.claude`, coordination, diagnostic store, unrelated `gecco/**`, and unrelated `tests/**` are dirty. Because the receipt is missing, the plan-required distinction between pre-existing dirt and follow-up-owned changes is not available.
- Forbidden patterns introduced or retained: `duckdb.Error` broad monitor handling, invalid/unreadable DuckDB monitor tests, broad test rewrites, and unverifiable scope provenance.
