# Phase 7 Review Scope Fix Follow-Up Receipt

Source plan: `docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_plan.md`.

## Pre-Flight Current Baseline

Historical pre-flight state for the earlier implementation is unavailable from Git alone. This receipt records only the current baseline captured immediately before this follow-up's edits in this conversation.

Exact `git status --short` baseline captured before this follow-up's edits:

```text
 M .claude/settings.json
 M README.md
 M bash/launch_vllm_server.sh
 M bash/run_cmg_generator.sh
 M bash/run_gecco_distributed.sh
 M bash/run_judge_orchestrator.sh
 M bash/run_test_evaluation.sh
 M config/two_step_factors_distributed.yaml
 M config/two_step_factors_gemma4.yaml
 M config/two_step_factors_glm5.yaml
 M config/two_step_factors_minimax_m2.yaml
 M config/two_step_factors_minimax_m27.yaml
 M config/two_step_factors_nemotron.yaml
 M config/two_step_factors_opencode_go.yaml
 M config/two_step_factors_step35.yaml
 M gecco-mh-dashboard/app.py
 M gecco-mh-dashboard/dashboard/config.py
 M gecco/artifacts.py
 M gecco/candidate_evaluation.py
 M gecco/candidate_generation.py
 M gecco/cli/monitor_distributed.py
 M gecco/cli/run_judge_orchestrator.py
 D gecco/construct_feedback/feedback.py
 M gecco/construct_feedback/orchestrated.py
 M gecco/construct_feedback/tool_judge.py
 M gecco/coordination.py
 M gecco/diagnostic_store/schema.py
 M gecco/run_gecco.py
 D scripts/launch_cmg_distributed.py
 D scripts/launch_distributed.py
 D scripts/monitor_distributed.py
 D scripts/reset_distributed.py
 D scripts/run_gecco_distributed.py
 D scripts/run_judge_orchestrator.py
 D scripts/run_test_evaluation.py
 M tests/fixtures/phase0/legacy_cli_inventory.json
 M tests/test_cmg_judge.py
 M tests/test_cmg_launcher.py
 M tests/test_cmg_registry.py
 M tests/test_cmg_runtime.py
 M tests/test_judge_orchestration.py
 M tests/test_phase0_characterization.py
 M tests/test_phase4_orchestrated_judge.py
 M tests/test_phase5_duckdb_canonical_state.py
 M tests/test_phase6_candidate_evaluator.py
 M tests/test_phase6_candidate_generator.py
 M tests/test_phase6_parallel_extraction_subtracks.py
?? config/__init__.py
?? docs/baseline_model_two_step_factors.md
?? docs/centralised_judge_implementation.md
?? docs/codebase_cleanup_plan.md
?? docs/codebase_cleanup_plan/
?? docs/codebase_cleanup_plan_deck.html
?? docs/codebase_deepdive_deck.html
?? docs/contract_first_planner_agent_prompt.md
?? docs/judge_implementation_record.md
?? docs/phase6_revision_difficulty_deck.html
?? docs/ppc_speedup_implementation.md
?? gecco/__main__.py
?? gecco/cli/__init__.py
?? gecco/cli/launch_cmg_distributed.py
?? gecco/cli/launch_distributed.py
?? gecco/cli/reset_distributed.py
?? gecco/cli/run_local_client.py
?? gecco/cli/run_test_evaluation.py
?? test_ppc_speedup.py
?? tests/test_phase1_docs_reset.py
?? tests/test_phase2_cli_contract.py
?? tests/test_phase7_duckdb_coordination_and_status_views.py
```

Exact tracked `git diff --name-only` baseline captured before this follow-up's edits:

```text
.claude/settings.json
README.md
bash/launch_vllm_server.sh
bash/run_cmg_generator.sh
bash/run_gecco_distributed.sh
bash/run_judge_orchestrator.sh
bash/run_test_evaluation.sh
config/two_step_factors_distributed.yaml
config/two_step_factors_gemma4.yaml
config/two_step_factors_glm5.yaml
config/two_step_factors_minimax_m2.yaml
config/two_step_factors_minimax_m27.yaml
config/two_step_factors_nemotron.yaml
config/two_step_factors_opencode_go.yaml
config/two_step_factors_step35.yaml
gecco-mh-dashboard/app.py
gecco-mh-dashboard/dashboard/config.py
gecco/artifacts.py
gecco/candidate_evaluation.py
gecco/candidate_generation.py
gecco/cli/monitor_distributed.py
gecco/cli/run_judge_orchestrator.py
gecco/construct_feedback/feedback.py
gecco/construct_feedback/orchestrated.py
gecco/construct_feedback/tool_judge.py
gecco/coordination.py
gecco/diagnostic_store/schema.py
gecco/run_gecco.py
scripts/launch_cmg_distributed.py
scripts/launch_distributed.py
scripts/monitor_distributed.py
scripts/reset_distributed.py
scripts/run_gecco_distributed.py
scripts/run_judge_orchestrator.py
scripts/run_test_evaluation.py
tests/fixtures/phase0/legacy_cli_inventory.json
tests/test_cmg_judge.py
tests/test_cmg_launcher.py
tests/test_cmg_registry.py
tests/test_cmg_runtime.py
tests/test_judge_orchestration.py
tests/test_phase0_characterization.py
tests/test_phase4_orchestrated_judge.py
tests/test_phase5_duckdb_canonical_state.py
tests/test_phase6_candidate_evaluator.py
tests/test_phase6_candidate_generator.py
tests/test_phase6_parallel_extraction_subtracks.py
```

## Files Changed For This Follow-Up And Ownership

The files below were already dirty at the captured baseline. Ownership is therefore by hunk, not by whole-file diff.

| File / Hunk | Ownership For This Follow-Up | Evidence / Notes |
|---|---|---|
| `tests/test_phase5_duckdb_canonical_state.py`, `test_run_test_evaluation_does_not_use_default_results_dir` assertions | Follow-up-owned | Added `registry_cls.side_effect = AssertionError("constructor path must not be used")`, default-path inequality inside `_open_existing()`, `registry_cls.open_existing.assert_called_once_with(registry_path)`, and `registry_cls.assert_not_called()`. |
| `tests/test_phase5_duckdb_canonical_state.py`, removed `test_monitor_load_registry_returns_none_for_invalid_duckdb_file` and `test_monitor_load_registry_returns_none_for_unreadable_duckdb_file` from the dirty worktree | Follow-up-owned | User clarified in this conversation that monitor invalid-DuckDB resilience was not necessary and could be treated as scope creep: "I'm pretty sure it's not really necessary so we can treat this as scope creep". |
| `gecco/cli/monitor_distributed.py`, removal of dirty-worktree `import duckdb` and `except (..., duckdb.Error)` | Follow-up-owned relative to the captured dirty baseline | The pre-edit dirty diff in this conversation showed `+import duckdb` and `except (FileNotFoundError, OSError, duckdb.Error)`. The final tracked diff no longer contains that hunk because the file now matches HEAD for `load_registry()`. Current source catches only `(FileNotFoundError, OSError)`. |
| `docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md` | Follow-up-owned | Required persisted receipt. |
| `tests/test_phase5_duckdb_canonical_state.py`, broad existing diffs outside the two follow-up hunks above | Current-baseline pre-existing / not follow-up-owned | Already present in the pre-edit dirty diff: `multiprocessing`, `_load_project_module()`, baseline DuckDB rename/assertion changes, process-contention rewrite, `test_run_test_evaluation_opens_existing_registry_read_only`, `test_test_fit_model_uses_shared_registry_duckdb`, and dashboard fixture `legacy_registry.txt`. |
| `gecco/cli/monitor_distributed.py`, `complete_no_success` classification changes | Current-baseline pre-existing / not follow-up-owned | Already present in the pre-edit dirty diff and part of the original Phase 7 behavior, not this follow-up's monitor-scope removal. |
| All other dirty tracked/untracked files from the baseline status above | Current-baseline pre-existing / not follow-up-owned | Left untouched by this follow-up. |

## Forbidden-Path Audits

Exact baseline and final forbidden-path audit outputs were the same for follow-up ownership purposes.

`git diff --name-only -- gecco-mh-dashboard`:

```text
gecco-mh-dashboard/app.py
gecco-mh-dashboard/dashboard/config.py
```

`git diff --name-only -- docs bash scripts config README.md .claude`:

```text
.claude/settings.json
README.md
bash/launch_vllm_server.sh
bash/run_cmg_generator.sh
bash/run_gecco_distributed.sh
bash/run_judge_orchestrator.sh
bash/run_test_evaluation.sh
config/two_step_factors_distributed.yaml
config/two_step_factors_gemma4.yaml
config/two_step_factors_glm5.yaml
config/two_step_factors_minimax_m2.yaml
config/two_step_factors_minimax_m27.yaml
config/two_step_factors_nemotron.yaml
config/two_step_factors_opencode_go.yaml
config/two_step_factors_step35.yaml
scripts/launch_cmg_distributed.py
scripts/launch_distributed.py
scripts/monitor_distributed.py
scripts/reset_distributed.py
scripts/run_gecco_distributed.py
scripts/run_judge_orchestrator.py
scripts/run_test_evaluation.py
```

`git diff --name-only -- gecco/coordination.py gecco/diagnostic_store`:

```text
gecco/coordination.py
gecco/diagnostic_store/schema.py
```

`git diff --name-only -- tests/test_phase7_duckdb_coordination_and_status_views.py`:

```text
```

`tests/test_phase7_duckdb_coordination_and_status_views.py` was untracked in `git status --short` before and after this follow-up, but it had no tracked diff output.

## Commands And Results

- `git status --short`: recorded exact current dirty baseline shown above.
- `git diff --name-only`: recorded exact tracked dirty baseline shown above.
- `git diff -- gecco/cli/monitor_distributed.py tests/test_phase5_duckdb_canonical_state.py`: pre-edit review showed broad existing dirty hunks plus the weak negative test and dirty-worktree `duckdb.Error` monitor handling; final diff shows the broad existing hunks, strengthened negative test, and no `duckdb.Error` monitor handling.
- `conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q -k "run_test_evaluation or monitor_distributed"`: passed, `2 passed, 11 deselected`.
- `conda run -n gecco_mh pytest tests/test_phase7_duckdb_coordination_and_status_views.py -q -k "complete_no_success or monitor"`: passed, `6 passed, 12 deselected`.
- `git diff --check`: passed with CRLF replacement warnings for pre-existing dirty files; no whitespace errors were reported.
- `test -f docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md`: passed.
- `grep -n "Pre-Flight\|Files Changed\|Forbidden-Path\|Commands\|Contract Coverage" docs/codebase_cleanup_plan/phase_7_review_scope_fix_followup_receipt.md`: found the required receipt sections.
- `grep -n "SharedRegistry" gecco/cli/run_test_evaluation.py`: found `SharedRegistry.open_existing(registry_path)`.
- `grep -n "test_run_test_evaluation_does_not_use_default_results_dir" -A60 tests/test_phase5_duckdb_canonical_state.py`: found the strengthened negative test assertions.
- `grep -n "duckdb\|duckdb.Error" gecco/cli/monitor_distributed.py tests/test_phase5_duckdb_canonical_state.py`: found legitimate `shared_registry.duckdb` and DuckDB registry references, but no `duckdb` import and no `duckdb.Error` catch.
- `grep -n "complete_no_success" gecco/cli/monitor_distributed.py tests/test_phase7_duckdb_coordination_and_status_views.py`: found monitor classification code and tests.

## Contract Coverage

- `test_run_test_evaluation_does_not_use_default_results_dir`: covered by the strengthened test and the phase5 targeted pytest command. The test uses a temp non-default `results_dir`, asserts the exact explicit `registry_path`, asserts the default path is not used, sets the `SharedRegistry(...)` constructor to fail, asserts `open_existing(registry_path)` exactly once, and asserts the constructor was not called.
- Monitor exception-handling scope: current source and final diff contain no `duckdb` import and no `duckdb.Error` catch. The invalid/unreadable DuckDB monitor tests were removed from the dirty worktree after explicit user confirmation that the monitor resilience behavior should be treated as scope creep.
- Implementation receipt: covered by this file.
- Scope firewall: follow-up-owned edits were limited to allowed files and allowed hunks. Broad dirty files and broad dirty hunks in allowed files are documented as current-baseline pre-existing above.

## Contract Rows Not Verified

- Historical pre-flight ownership for the earlier implementation is not verified because the earlier baseline is unavailable.
- Broad current-baseline dirty files and broad pre-existing hunks are not verified for behavior or correctness because they are outside this follow-up scope.

## Manual Review Checks

- `test_run_test_evaluation_does_not_use_default_results_dir` contains `registry_cls.side_effect = AssertionError("constructor path must not be used")`, `_open_existing()` checks for the explicit path and against the default path, `registry_cls.open_existing.assert_called_once_with(registry_path)`, and `registry_cls.assert_not_called()`.
- `gecco/cli/run_test_evaluation.py` uses `SharedRegistry.open_existing(registry_path)`.
- `gecco/cli/monitor_distributed.py` has no `duckdb` import and `load_registry()` catches only `(FileNotFoundError, OSError)`.
- `tests/test_phase5_duckdb_canonical_state.py` no longer contains the invalid/unreadable DuckDB monitor resilience tests.
- `complete_no_success` classification remains in `gecco/cli/monitor_distributed.py` and is covered by the passing phase7 targeted pytest command.

## Expected Fail-Before / Pass-After Behavior

- Before this follow-up, `test_run_test_evaluation_does_not_use_default_results_dir` could pass without proving constructor avoidance or exact `open_existing(registry_path)` usage.
- After this follow-up, the test fails if runtime uses `SharedRegistry(...)`, uses the default registry path, or does not call `SharedRegistry.open_existing(registry_path)` exactly once.
- Before this follow-up, dirty-worktree monitor invalid/unreadable DuckDB tests supported out-of-scope `duckdb.Error` handling.
- After this follow-up, those tests and the broad exception handling are absent as approved scope-creep removal.

## Deferred Adjacent Fixes

- Dashboard DuckDB path/status cleanup.
- Registry, diagnostic-store, schema, migration, and config cleanup.
- Broad Phase 7 test cleanup.
- Script-helper cleanup beyond existing current-baseline changes.
- Any monitor invalid-DuckDB resilience behavior.
