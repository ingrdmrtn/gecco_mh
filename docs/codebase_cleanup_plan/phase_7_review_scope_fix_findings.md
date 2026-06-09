# Phase 7 Review Scope Fix Findings

## 1. Findings

### High: Scope firewall cannot be verified because forbidden paths are dirty and no pre-flight receipt is present

Plan violation: Scope Firewall, Phase 0 gate, Phase 3 final audit, Final Verification receipt.

`git status --short` and forbidden-path audits show many dirty files outside the allowed list, including dashboard, docs, scripts, config, `.claude`, `gecco/coordination.py`, `gecco/diagnostic_store/schema.py`, unrelated `gecco/**`, and unrelated `tests/**`.

No implementation receipt or pre-flight dirty-file record was found in the worktree, so these cannot be distinguished as pre-existing versus plan-owned.

Examples:

- `gecco-mh-dashboard/app.py`
- `gecco-mh-dashboard/dashboard/config.py`
- `.claude/settings.json`
- `README.md`
- `bash/*.sh`
- `scripts/*.py`
- `config/*.yaml`
- `gecco/coordination.py`
- `gecco/diagnostic_store/schema.py`
- many unrelated `gecco/**` and `tests/**` files

Action: provide the required receipt with pre-flight dirty files, or remove/segregate all plan-owned out-of-scope changes.

### High: Allowed test files include broad unrelated changes beyond the two approved contracts

Plan violation: Scope Firewall, Explicit Non-Goals, Forbidden Patterns "opportunistic refactors", Phase 1/2 allowed changes.

`tests/test_phase5_duckdb_canonical_state.py` includes changes unrelated to the Phase 7 review-scope fix, including:

- project module loader for `scripts/test_fit_model.py`: `tests/test_phase5_duckdb_canonical_state.py:39`
- baseline concurrency rewrite using multiprocessing: `tests/test_phase5_duckdb_canonical_state.py:195`
- monitor invalid/unreadable DuckDB tests: `tests/test_phase5_duckdb_canonical_state.py:260`
- script helper registry test: `tests/test_phase5_duckdb_canonical_state.py:362`
- dashboard/task-discovery legacy file behavior change: `tests/test_phase5_duckdb_canonical_state.py:498`

`tests/test_phase7_duckdb_coordination_and_status_views.py` is untracked and contains broad Phase 7 tests beyond the required monitor classification tests, including deletion guards, runtime view checks, concurrency checks, runner/evaluator ownership checks, and CMG persistence checks: `tests/test_phase7_duckdb_coordination_and_status_views.py:41-502`.

Action: identify these as pre-existing in the receipt or remove/isolate them from this plan's diff.

### Medium: `monitor_distributed.py` includes an adjacent exception-handling change not authorized by the plan

Plan violation: Phase 2 "Keep the change local to the CLI monitor" for status classification only; Forbidden Pattern "Broad exception handling unrelated to the two named fixes".

The diff adds `duckdb` import and catches `duckdb.Error` in `load_registry()`:

- `gecco/cli/monitor_distributed.py:10`
- `gecco/cli/monitor_distributed.py:43`

This is not part of the required `complete_no_success` summary/table classification contract.

Action: remove this change from the plan-owned diff or explicitly justify it as pre-existing/user-approved scope expansion.

### Medium: Negative test for default results directory is weak

Plan violation: Contract Matrix row `run_test_evaluation()` registry handoff; Phase 1 negative test requirement.

`test_run_test_evaluation_does_not_use_default_results_dir` asserts the default directory exists but does not assert `open_existing()` was called, and it does not make the constructor fail:

- `tests/test_phase5_duckdb_canonical_state.py:321`
- `tests/test_phase5_duckdb_canonical_state.py:342`
- `tests/test_phase5_duckdb_canonical_state.py:349`

If implementation used `SharedRegistry(...)` instead of `SharedRegistry.open_existing(...)`, this negative test could still pass independently of the default-path contract. The positive test catches constructor use, but the plan requires the negative default-path test itself to fail if the wrong/default path is used.

Action: make the negative test assert `registry_cls.open_existing.assert_called_once_with(registry_path)` and make constructor use fail.

### Medium: Required implementation receipt is missing

Plan violation: Final Verification, Required implementation verification receipt.

No receipt was found with:

- exact tests and commands run
- contract rows covered by each command
- unverified rows and reasons
- expected fail-before/pass-after tests
- grep/manual check summaries
- pre-flight `git status --short`
- final changed-file list for this plan
- forbidden-path audit summaries
- statement that dashboard files were not changed for this plan or explicit approval

Action: add or provide the receipt before merge review can be completed.

## 2. Open Questions

- Which dirty/untracked files were present before this Phase 7 review-scope-fix implementation began?

## 3. Verification Summary

Observed static checks:

- `git status --short`
- `git diff --name-only`
- `git diff --stat`
- `git diff --name-only -- gecco-mh-dashboard`
- `git diff --name-only -- docs bash scripts config README.md .claude`
- `git diff --name-only -- gecco/coordination.py gecco/diagnostic_store`
- `git diff --check`
- static grep/review for `SharedRegistry`
- static grep/review for `complete_no_success`

Tests not observed/run in this review:

- `conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q -k "run_test_evaluation"`
- `conda run -n gecco_mh pytest tests/test_phase7_duckdb_coordination_and_status_views.py -q -k "complete_no_success or monitor"`
- full Phase 3/final verification pytest commands

Reason: review session was read-only; no implementation receipt proving prior execution was found.

Contract rows verified statically:

- `run_test_evaluation()` uses `SharedRegistry.open_existing(registry_path)` at `gecco/cli/run_test_evaluation.py:143`.
- `build_summary_stats()` counts `complete_no_success` as complete at `gecco/cli/monitor_distributed.py:264-268`.
- `build_client_table()` renders `complete_no_success` green at `gecco/cli/monitor_distributed.py:60-67`.

Contract rows not fully verified:

- `run_test_evaluation()` registry handoff: implementation present, but required negative test is weak and targeted pytest proof is absent from the review artifact.
- Monitor status classification: implementation/tests present, but targeted pytest proof is absent from the review artifact.
- Scope firewall: not verified because forbidden paths are dirty and no pre-flight/receipt separates pre-existing files from plan-owned changes.

## 4. Scope Audit

Allowed files changed or untracked:

- `gecco/cli/run_test_evaluation.py`: allowed implementation file; untracked.
- `gecco/cli/monitor_distributed.py`: allowed implementation file; includes one suspicious out-of-scope exception-handling change.
- `tests/test_phase5_duckdb_canonical_state.py`: allowed test file; contains broad unrelated changes unless documented pre-existing.
- `tests/test_phase7_duckdb_coordination_and_status_views.py`: allowed test file; untracked and contains broad unrelated Phase 7 coverage unless documented pre-existing.

Forbidden dirty paths observed:

- `.claude/settings.json`
- `README.md`
- `bash/launch_vllm_server.sh`
- `bash/run_cmg_generator.sh`
- `bash/run_gecco_distributed.sh`
- `bash/run_judge_orchestrator.sh`
- `bash/run_test_evaluation.sh`
- `config/*.yaml`
- `config/__init__.py`
- `docs/**`
- `gecco-mh-dashboard/app.py`
- `gecco-mh-dashboard/dashboard/config.py`
- `gecco/coordination.py`
- `gecco/diagnostic_store/schema.py`
- `scripts/*.py`
- unrelated `gecco/**`
- unrelated `tests/**`

Forbidden or suspicious changes:

- Dashboard files are dirty, violating the dashboard firewall unless proven pre-existing.
- Docs/scripts/config files are dirty, violating forbidden paths unless proven pre-existing.
- `gecco/coordination.py` and `gecco/diagnostic_store/schema.py` are dirty, explicitly forbidden unless proven pre-existing.
- CRLF normalization warnings appeared for config and dashboard files during diff checks, matching the plan's forbidden formatting-only/CRLF risk.
