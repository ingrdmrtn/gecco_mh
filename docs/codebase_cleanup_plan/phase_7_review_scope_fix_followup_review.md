# Phase 7 Review Scope Fix Follow-Up Review

Source plan: `docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_plan.md`.

## 1. Findings

### High: Receipt scope ownership is not consistent with the actual worktree diff

Violates: Contract Matrix row `Implementation receipt`; Contract Matrix row `Scope firewall`; Phase 1 gate requiring the diff to touch only the intended test block except recorded pre-existing current-baseline changes; Phase 3 requirement to separate files changed for this follow-up from files already dirty at pre-flight.

The receipt claims the follow-up-owned test-file work was limited to removing two invalid-DuckDB monitor tests and strengthening `test_run_test_evaluation_does_not_use_default_results_dir`:

`docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md:13-19`

```md
- `tests/test_phase5_duckdb_canonical_state.py`: removed two tests that existed only to support invalid/unreadable DuckDB monitor resilience behavior; strengthened `test_run_test_evaluation_does_not_use_default_results_dir` ...
```

The actual tracked diff for `tests/test_phase5_duckdb_canonical_state.py` contains additional broad changes outside that required negative-test block, including:

- `tests/test_phase5_duckdb_canonical_state.py:5-58`: added `multiprocessing`, `_load_project_module()`, and import-time loading of `scripts/test_fit_model.py`.
- `tests/test_phase5_duckdb_canonical_state.py:160-257`: changed baseline tests from JSON/lock assertions to DuckDB assertions and rewrote the concurrency test from thread contention to process contention.
- `tests/test_phase5_duckdb_canonical_state.py:260-294`: added `test_run_test_evaluation_opens_existing_registry_read_only`.
- `tests/test_phase5_duckdb_canonical_state.py:342-370`: added `test_test_fit_model_uses_shared_registry_duckdb`, covering a script helper that is not in this follow-up scope.
- `tests/test_phase5_duckdb_canonical_state.py:478-484`: changed dashboard task discovery fixture data from `legacy_registry.json` to `legacy_registry.txt`.

Those changes may be pre-existing current-baseline dirt, but the receipt does not identify them individually as pre-existing or follow-up-owned. Because `tests/test_phase5_duckdb_canonical_state.py` was already dirty at pre-flight, the plan requires a receipt precise enough to separate the follow-up edit from existing dirt. The current receipt does not make that separation reviewable.

Action: update the receipt with exact pre-flight status/diff evidence or explicit per-hunk ownership for `tests/test_phase5_duckdb_canonical_state.py`; otherwise this review cannot verify that Phase 1 stayed within scope.

### Medium: Receipt claims a monitor `duckdb.Error` removal that is not present in the current tracked diff

Violates: Contract Matrix row `Implementation receipt`; Phase 2 evidence requirement to report whether `duckdb.Error` handling was removed, retained as pre-existing, or retained with explicit user approval.

The receipt says:

`docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md:15`

```md
- `gecco/cli/monitor_distributed.py`: removed the out-of-scope `duckdb.Error` monitor resilience change by removing the `duckdb` import and removing `duckdb.Error` from `load_registry()` exception handling.
```

The observed tracked diff for `gecco/cli/monitor_distributed.py` contains only `complete_no_success` classification changes in `build_client_table()` and `build_summary_stats()`; it does not show removal of an `import duckdb` or an `except (..., duckdb.Error)` clause. Current source has no `duckdb` import and `load_registry()` catches only `(FileNotFoundError, OSError)` at `gecco/cli/monitor_distributed.py:35-43`.

This may mean the `duckdb.Error` change was already removed before this follow-up, or that the relevant removal happened in an unrecorded baseline. As written, the receipt overstates what the actual diff proves.

Action: correct the receipt to say the current review observed no `duckdb.Error` handling and explain whether removal occurred before this follow-up, during this follow-up, or was already absent at the captured baseline.

### Medium: Pre-flight and final dirty-file baselines are summarized too vaguely to verify the Scope Firewall

Violates: Contract Matrix row `Implementation receipt`; Phase 0 required steps; Phase 3 receipt requirements; Phase 4 final verification requirements.

The receipt summarizes the baseline as broad categories instead of recording exact `git status --short` and final changed-file outputs:

- `docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md:9`: “multiple `gecco/**` modules” and “many `tests/**` files”.
- `docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md:11`: “plus many forbidden-path baseline files”.
- `docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md:21-24`: forbidden paths are described by category rather than exact files.
- `docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md:44-46`: final status and final `git diff --name-only` are summarized, but exact final outputs are not persisted.

The plan allows summaries in some phase evidence, but the Contract Matrix requires the receipt to persist current pre-flight dirty baseline and final changed-file list for this follow-up. In a broadly dirty worktree, category summaries are not enough to distinguish pre-existing dirty files from follow-up-owned changes.

Action: add exact pre-flight and final `git status --short` / `git diff --name-only` outputs, or an exact file-by-file table that marks each file as pre-existing, follow-up-owned, or review-output-only.

### Medium: User approval / ownership for removed invalid-DuckDB monitor tests is not evidenced

Violates: Phase 2 requirement to remove invalid-DuckDB monitor tests only if they are plan-owned, or stop and ask if they are user-approved pre-existing work; Phase 2 evidence requirement.

The receipt says invalid/unreadable DuckDB monitor resilience tests were removed “after user confirmation”:

`docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md:51`

```md
Invalid/unreadable DuckDB monitor resilience tests were removed as scope creep after user confirmation.
```

The receipt does not identify the confirmation source, timestamp, or exact tests removed. Because the plan explicitly requires stopping for clarification if ownership is unclear, this undocumented approval is not reviewable from the repo.

Action: document the approval source or mark the ownership as unverified and ask for confirmation before treating those removals as compliant.

## 2. Open Questions

Was there an explicit user confirmation, outside the persisted receipt, approving removal of the invalid/unreadable DuckDB monitor resilience tests and classifying them as plan-owned scope creep?

Were the broad `tests/test_phase5_duckdb_canonical_state.py` diffs outside `test_run_test_evaluation_does_not_use_default_results_dir` present before this follow-up’s Phase 0 baseline?

## 3. Verification Summary

Observed commands/checks run during review:

```bash
git status --short
git diff --name-only
git diff -- tests/test_phase5_duckdb_canonical_state.py gecco/cli/monitor_distributed.py docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md
git diff --name-only -- gecco-mh-dashboard
git diff --name-only -- docs bash scripts config README.md .claude
git diff --name-only -- gecco/coordination.py gecco/diagnostic_store
git diff --name-only -- tests/test_phase7_duckdb_coordination_and_status_views.py
conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q -k "run_test_evaluation or monitor_distributed"
conda run -n gecco_mh pytest tests/test_phase7_duckdb_coordination_and_status_views.py -q -k "complete_no_success or monitor"
git diff --check
grep -n "duckdb\|duckdb.Error" gecco/cli/monitor_distributed.py tests/test_phase5_duckdb_canonical_state.py
grep -n "SharedRegistry" gecco/cli/run_test_evaluation.py
grep -n "complete_no_success" gecco/cli/monitor_distributed.py tests/test_phase7_duckdb_coordination_and_status_views.py
grep -n "test_run_test_evaluation_does_not_use_default_results_dir" -A60 tests/test_phase5_duckdb_canonical_state.py
```

Observed results:

- `tests/test_phase5_duckdb_canonical_state.py -k "run_test_evaluation or monitor_distributed"`: `2 passed, 11 deselected`.
- `tests/test_phase7_duckdb_coordination_and_status_views.py -k "complete_no_success or monitor"`: `6 passed, 12 deselected`.
- `git diff --check`: no whitespace errors reported; CRLF replacement warnings appeared for pre-existing dirty files.
- `grep "duckdb\|duckdb.Error"`: no `import duckdb` or `duckdb.Error` catch observed; legitimate `shared_registry.duckdb` and registry references remain.
- `grep "SharedRegistry" gecco/cli/run_test_evaluation.py`: `SharedRegistry.open_existing(registry_path)` observed at `gecco/cli/run_test_evaluation.py:143`.
- `grep "test_run_test_evaluation_does_not_use_default_results_dir" -A60`: strengthened negative-test assertions observed at `tests/test_phase5_duckdb_canonical_state.py:297-339`.
- `grep "complete_no_success"`: monitor classification code observed at `gecco/cli/monitor_distributed.py:61` and `gecco/cli/monitor_distributed.py:266`; tests observed in `tests/test_phase7_duckdb_coordination_and_status_views.py`.

Contract rows verified:

- `test_run_test_evaluation_does_not_use_default_results_dir`: implementation present and targeted test passes.
- Monitor exception-handling scope: current code has no `duckdb.Error` catch; targeted monitor classification tests pass.

Contract rows not fully verified:

- Implementation receipt: not verified because the receipt is incomplete/inaccurate against the current diff.
- Scope firewall: not verified because exact pre-flight/final ownership is not persisted, and the tracked diff contains broad changes in an allowed test file beyond the stated follow-up edit.

## 4. Scope Audit

Tracked changed files observed by `git diff --name-only` include many forbidden paths, including `.claude/settings.json`, `README.md`, `bash/**`, `config/**`, `gecco-mh-dashboard/**`, `gecco/coordination.py`, `gecco/diagnostic_store/schema.py`, deleted `scripts/**`, unrelated `gecco/**`, and unrelated `tests/**`. The receipt classifies these as pre-existing baseline dirt, but does not persist exact file-by-file baseline/final ownership.

Allowed files with relevant changes:

- `gecco/cli/monitor_distributed.py`: allowed only for removing plan-owned `duckdb` import/catch. Actual tracked diff contains `complete_no_success` classification changes, not `duckdb.Error` removal. This appears to be earlier Phase 7 work rather than the follow-up edit, but the receipt claims follow-up ownership inaccurately.
- `tests/test_phase5_duckdb_canonical_state.py`: allowed only to strengthen `test_run_test_evaluation_does_not_use_default_results_dir` and remove invalid-DuckDB tests if scope-owned. The required negative test is present, but the tracked diff also includes broad test rewrites/additions outside the receipt’s stated follow-up work.
- `docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_plan.md`: allowed by the plan; currently untracked under the broadly untracked docs directory.
- `docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md`: allowed receipt file; currently untracked and present.

Forbidden or suspicious changes:

- `gecco-mh-dashboard/app.py` and `gecco-mh-dashboard/dashboard/config.py`: forbidden by Scope Firewall; observed as tracked dirty and claimed as pre-existing.
- `.claude/settings.json`, `README.md`, `bash/**`, `config/**`, deleted `scripts/**`: forbidden by Scope Firewall; observed as tracked dirty and claimed as pre-existing.
- `gecco/coordination.py` and `gecco/diagnostic_store/schema.py`: forbidden by Scope Firewall; observed as tracked dirty and claimed as pre-existing.
- Unrelated `tests/**` and `gecco/**` files: forbidden by Scope Firewall; observed as tracked dirty and claimed as pre-existing.
- `tests/test_phase7_duckdb_coordination_and_status_views.py`: forbidden unless approved; no tracked diff, but the file is untracked in `git status --short`.

No forbidden-pattern runtime issue was found in `run_test_evaluation()` for this contract: it uses `results_dir / "shared_registry.duckdb"` and `SharedRegistry.open_existing(registry_path)`.
