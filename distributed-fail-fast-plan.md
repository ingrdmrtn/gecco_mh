# Distributed Fail-Fast Plan

## Summary

Add fail-fast abort coordination for distributed GeCCo runs: when any client fails, all surviving clients and the judge orchestrator should observe a shared abort signal and exit non-zero quickly instead of waiting at per-iteration barriers. The main risk this prevents is partial distributed runs consuming walltime and producing misleading results after one client is no longer participating.

## Scope

Allowed implementation files:

- `gecco/coordination.py`: shared abort state API, abort-aware wait loops.
- `gecco/diagnostic_store/schema.py`: minimal runtime abort/state table if needed.
- `gecco/cli/run_gecco_distributed.py`: catch unhandled client failures, publish abort, re-raise.
- `gecco/run_gecco.py`: abort checks before/inside long distributed client waits.
- `gecco/cli/run_judge_orchestrator.py`: abort-aware orchestrator exit behavior if not fully covered by `SharedRegistry` wait loops.
- `gecco/candidate_generation.py` and `gecco/candidate_evaluation.py`: only if needed to publish abort for exceptions that are currently swallowed or downgraded before reaching `run_gecco_distributed.py`.

Allowed test files:

- `tests/test_judge_orchestration.py`
- `tests/test_phase4_orchestrated_judge.py`
- `tests/test_phase5_duckdb_canonical_state.py`
- `tests/test_phase7_duckdb_coordination_and_status_views.py`
- Add one focused test file under `tests/` only if these files cannot host the behavior cleanly.

Allowed verification artifacts:

- None required. Final response listing commands and results is sufficient.

Forbidden files/directories:

- Do not modify config YAMLs under `config/`.
- Do not modify dashboard files under `gecco-mh-dashboard/`.
- Do not modify generated results, logs, or Slurm outputs under `results/` or `logs/`.
- Do not modify prompt templates or LLM message construction.

Production surfaces in scope:

- Distributed client entrypoint: `gecco.cli.run_gecco_distributed.run_distributed_client`.
- Client iteration loop: `gecco.run_gecco.GeCCoModelSearch.run_n_shots`.
- Judge feedback wait: `SharedRegistry.wait_for_judge_feedback`.
- Orchestrator client barrier: `SharedRegistry.wait_for_clients_complete`, used by `gecco.cli.run_judge_orchestrator.run_orchestrator`.
- Registry persistence: DuckDB runtime tables created by `gecco.diagnostic_store.schema.create_schema`.
- Launch behavior is out of scope except to preserve existing `afterok` final-eval dependency behavior.

Non-goals:

- Do not dynamically reduce `n_clients` after a failure.
- Do not add client heartbeat/preemption recovery.
- Do not retry failed clients or resubmit Slurm jobs.
- Do not make final evaluation run after partial distributed failure.
- Do not change judge mode enablement beyond existing behavior.

Worktree rules:

- Run `git status --short` before implementation.
- Classify unrelated pre-existing tracked and untracked changes briefly before editing.
- Do not revert or modify unrelated pre-existing changes.
- Final changed-file review must compare actual changed files against this scope; any file outside scope requires user approval.

## Contracts

| Contract | Code path / symbol | Positive proof | Negative proof |
| --- | --- | --- | --- |
| Client failure publishes abort and exits non-zero | `run_distributed_client`, `SharedRegistry.request_abort` | Test raises inside client execution and asserts abort row has failing `client_id`, reason, and status `failed` | Test must not swallow the exception or return success |
| Surviving clients stop waiting quickly | `SharedRegistry.wait_for_judge_feedback`, `GeCCoModelSearch.run_n_shots` | Test pre-populates abort and asserts wait raises immediately without sleeping until timeout | Test verifies `wait_for_judge_feedback` is not allowed to return `None` timeout feedback when abort exists |
| Orchestrator stops waiting quickly | `SharedRegistry.wait_for_clients_complete`, `run_orchestrator` | Test pre-populates abort and asserts barrier raises before timeout | Test verifies fallback/proceed-with-available-results path is not used on abort |
| Abort state is durable and readable across registry handles | `SharedRegistry` DuckDB methods and schema | Test writes abort through one registry instance and reads/raises through another | Test verifies missing abort returns no abort and normal wait timeout behavior remains available |
| Final eval remains blocked by failed client array | `launch_distributed._build_regular_launch_plan` existing behavior | Existing/updated launcher test asserts final eval dependency is still `afterok:<client_array_job_id>` | No switch to `afterany` or unconditional final-eval launch |

## Tests And Verification

Add or update targeted tests:

- `tests/test_phase5_duckdb_canonical_state.py`
  - Add `test_abort_state_round_trips_across_registry_handles`.
  - Add `test_raise_if_aborted_is_noop_without_abort`.
- `tests/test_judge_orchestration.py` or `tests/test_phase4_orchestrated_judge.py`
  - Add `test_wait_for_judge_feedback_raises_on_abort_without_timeout_wait`.
  - Add `test_wait_for_clients_complete_raises_on_abort_without_proceeding`.
- `tests/test_phase4_orchestrated_judge.py`
  - Add or extend a `run_n_shots` distributed client test so pre-existing abort prevents centralized wait/timeout feedback.
- `tests/test_phase2_cli_contract.py`
  - Keep or add assertion that final evaluation uses `afterok` dependency on client array.
- `tests/test_phase2_cli_contract.py` or a focused new CLI test
  - Add `test_distributed_client_publishes_abort_on_unhandled_exception` using mocks around expensive setup. It must exercise `run_distributed_client` enough to prove the `except` path writes to `SharedRegistry`, not only a helper.

Verification commands:

- `conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py tests/test_judge_orchestration.py tests/test_phase4_orchestrated_judge.py tests/test_phase2_cli_contract.py`
- If a new test file is added, include it in the command.
- Run `git status --short` after tests and review changed files against Scope.

Manual review checks:

- Confirm abort exceptions include enough context: failing client id, iteration when known, and original exception class/message.
- Confirm wait loops check abort on every poll before timeout handling.
- Confirm no broad `except Exception: pass` or fallback-to-timeout behavior is introduced.
- Confirm schema changes are idempotent for existing DuckDB files via `CREATE TABLE IF NOT EXISTS` or equivalent migration-safe DDL.

## Implementation Steps

1. Baseline scope check
   - Run `git status --short`.
   - Note unrelated existing changes and do not touch them.

2. Add abort persistence contract first
   - In `gecco/diagnostic_store/schema.py`, add the smallest durable table needed for one active abort record, e.g. `runtime_abort` with singleton key, `client_id`, `iteration`, `reason`, `created_at`.
   - In `gecco/coordination.py`, add `request_abort`, `get_abort`, and `raise_if_aborted`.
   - Tests: add round-trip and no-op tests before or alongside implementation.
   - Check: abort written by one `SharedRegistry` instance is visible to another.

3. Make registry waits abort-aware
   - Update `wait_for_clients_complete` and `wait_for_judge_feedback` to call `raise_if_aborted()` once before entering the loop and once per poll before timeout fallback.
   - Keep normal timeout behavior when no abort exists.
   - Tests: assert abort raises immediately and does not return timeout/fallback results.
   - Check: no sleep-heavy tests; use tiny poll/timeout values or monkeypatch sleep if necessary.

4. Publish abort from distributed client failures
   - In `run_distributed_client`, wrap the main work after registry creation in `try/except Exception`.
   - On exception, call `registry.request_abort(reason=..., client_id=resolved_client_id)` and set client status to `failed` using a focused registry method or existing update path with empty results.
   - Re-raise the original exception so Slurm marks the array task failed.
   - Tests: mock expensive setup and force an exception after registry creation; assert abort/status persisted and exception propagated.

5. Stop clients before starting expensive work when another client failed
   - In `GeCCoModelSearch.run_n_shots`, call `self.shared_registry.raise_if_aborted()` before each iteration when `shared_registry` is present.
   - Rely on abort-aware `wait_for_judge_feedback` for centralized wait loops.
   - Tests: pre-populate abort and assert `run_n_shots` raises before generating/evaluating or producing timeout feedback.

6. Stop orchestrator cleanly
   - If `SharedRegistry.wait_for_clients_complete` raising is sufficient, ensure `run_orchestrator` does not catch and downgrade the abort.
   - If needed, add a narrow abort check in `run_orchestrator` before each iteration.
   - Tests: orchestrator or wait-loop test must prove no fallback judge feedback is written after abort.

7. Preserve launch/final-eval behavior
   - Review existing launcher tests for `afterok`; update only if the test expectations need to include the new behavior.
   - Do not add cancellation or `scancel` behavior in this change.

8. Final verification
   - Run the targeted pytest command.
   - Run `git status --short`.
   - Review changed files against Scope and ensure no unrelated files were modified.

## Forbidden Patterns

- Do not use in-memory globals for abort state; it must be DuckDB-backed and visible across processes.
- Do not reduce `n_clients` or proceed with partial client results after abort.
- Do not convert abort into timeout feedback for clients.
- Do not write fallback judge feedback after abort.
- Do not catch abort exceptions and return success from client or orchestrator entrypoints.
- Do not change Slurm dependencies from `afterok` to `afterany`.
- Do not add hardcoded paths to specific run IDs, results directories, or Slurm job IDs.
- Do not mock away the registry behavior in tests that claim to prove cross-process visibility.
- Do not perform opportunistic refactors of candidate generation, evaluation, dashboard, or config loading.

## Acceptance Checklist

Client failure abort:

- Test proves unhandled distributed client exception writes abort state and failed client status.
- Test proves the original exception propagates so the process exits non-zero.

Surviving client behavior:

- Test proves `wait_for_judge_feedback` raises on abort before timeout fallback.
- Test proves `run_n_shots` checks abort before expensive generation/evaluation in distributed mode.

Orchestrator behavior:

- Test proves `wait_for_clients_complete` raises on abort before proceeding with available clients.
- Review confirms `run_orchestrator` does not downgrade abort into fallback judge feedback.

Persistence behavior:

- Test proves abort state round-trips across separate `SharedRegistry` handles.
- Schema DDL is idempotent and safe for existing registries.

Launch behavior:

- Existing or updated test proves final evaluation remains `afterok` dependent on the client array.

Verification:

- `conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py tests/test_judge_orchestration.py tests/test_phase4_orchestrated_judge.py tests/test_phase2_cli_contract.py` passes, plus any new focused test file.
- Final `git status --short` reviewed.
- Changed files are limited to the allowed scope or user-approved additions.
- Any remaining unverified risk is stated in the final response.
