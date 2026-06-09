# Cleanup Acceptance Gaps Plan

## Summary

Close the remaining cleanup-plan acceptance gaps found in review: make centralized judge evidence DuckDB-first end to end, retire the `judge.orchestrated` runtime flag instead of accepting inconsistent values, remove old lesion-named runtime/config/test surfaces, and keep dashboard work out of scope. The main risk this plan prevents is declaring DuckDB canonical while production paths still depend on JSON artifacts or legacy config names.

## Scope

Allowed implementation files:

- `config/schema.py`
- `gecco/cli/launch_distributed.py`
- `gecco/cli/run_judge_orchestrator.py`
- `gecco/run_gecco.py` only to remove the retired CMG runtime dependency on `judge.orchestrated`
- `gecco/construct_feedback/tool_judge.py` only if tightening capability prompts/post-processing is required by tests below
- `gecco/coordination.py` only if a small read/query helper is needed for orchestrator evidence
- `gecco/diagnostic_store/store.py` and `gecco/diagnostic_store/schema.py` only if a minimal DuckDB copy/import helper is needed
- `config/*.yaml` only to rename/update lesion-named configs and task names

Allowed test files:

- `tests/test_phase2_cli_contract.py`
- `tests/test_phase3_config_validation.py`
- `tests/test_phase4_orchestrated_judge.py`
- `tests/test_phase5_duckdb_canonical_state.py`
- `tests/test_phase7_duckdb_coordination_and_status_views.py`
- `tests/test_cmg_runtime.py` only for focused proof that CMG runtime no longer depends on `judge.orchestrated`
- Add one focused test file only if these files become too crowded

Allowed verification artifacts:

- This plan file only
- Temporary files under pytest `tmp_path` only

Forbidden files/directories:

- `gecco-mh-dashboard/**`
- `scripts/**` except if a failing test proves a direct dependency on the changed config names
- `bash/**`
- `README.md`, `.claude/**`, broad docs outside this plan
- Unrelated model-fitting, PPC, HBI, or provider backend files

Non-goals:

- No dashboard behavior updates.
- No broad report/export rewrite.
- No compatibility aliases for old lesion config filenames.
- No console-script packaging work unless a CLI test already proves it is required.
- No expensive fitting, HBI, or PPC in default tests.

Worktree rules:

- Run `git status --short` before implementation.
- Do not revert or modify unrelated pre-existing changes.
- Because the worktree is broad and dirty, classify pre-existing dirty files briefly before edits and compare final changed files against that baseline.
- Scope audits must include tracked and untracked files.
- Any implementation-owned changed file outside the allowed scope means the work is incomplete unless the user approves the scope change.

## Contracts

| Contract | Proof |
| --- | --- |
| The centralized judge does not require `results/*/bics/iter*.json` to build evidence for normal runtime. It reads/copies/queries DuckDB state instead. | Add/update a Phase 4 or Phase 5 test where `ArtifactStore(inspection_output_enabled=False)` writes no JSON, the runtime DuckDB has models, and `run_orchestrator` can construct the judge store without calling `rebuild_from_artifacts` or scanning `bics/*.json`. |
| `judge.orchestrated` is not a meaningful user switch. `false` and the field itself must fail validation or be removed from configs; launch/client behavior must not diverge. | Add config validation test rejecting `judge.orchestrated: false` or any explicit `judge.orchestrated` field, update launcher tests so orchestrator launch is inferred from `judge` presence/capabilities rather than raw YAML `orchestrated`, and ensure CMG runtime validation no longer requires a hidden `cfg.judge.orchestrated` attribute. |
| No production config/test surface uses old lesion names for capability variants. | Add/update test that production config filenames and task names do not contain `_lesion`, `lesion_`, or `lesion`; update references to new capability-oriented names. |
| Capability-limited judge behavior is enforced before generation where plausible, not only after broad all-capability prompting. | Add/update tests for at least `attempted_models_overview` and no-`recommendations` cases so prompts/output do not request or retain prohibited content. Source review confirms no hidden legacy fallback. |
| Dashboard remains out of scope. | Final `git diff --name-only -- gecco-mh-dashboard` has no implementation-owned changes; if pre-existing dirty dashboard files remain, classify them as pre-existing only. |

## Tests And Verification

Add or update tests:

- `tests/test_phase4_orchestrated_judge.py`: orchestrator uses DuckDB evidence without JSON rebuild fallback in normal runtime.
- `tests/test_phase3_config_validation.py`: reject explicit `judge.orchestrated`; production config filenames/task names are capability-oriented and lesion-free.
- `tests/test_phase2_cli_contract.py`: distributed launcher does not parse raw YAML to decide orchestrator mode and validates config through `load_config`.
- `tests/test_cmg_runtime.py` if Step 4 touches `gecco/run_gecco.py`: CMG runtime infers centralized judge availability without requiring `judge.orchestrated`.
- `tests/test_phase3_config_validation.py` or `tests/test_phase4_orchestrated_judge.py`: capability-specific judge prompt/output constraints for the most likely mistakes.

Commands to run:

- `conda run -n gecco_mh pytest tests/test_phase2_cli_contract.py tests/test_phase3_config_validation.py tests/test_phase4_orchestrated_judge.py tests/test_phase5_duckdb_canonical_state.py tests/test_phase7_duckdb_coordination_and_status_views.py -q`
- If touched: `conda run -n gecco_mh pytest tests/test_phase8_report_export.py -q`

Manual review checks:

- `git status --short` before and after.
- `git diff --name-only` and include untracked files in the final scope audit.
- `git diff --name-only -- gecco-mh-dashboard` to prove no owned dashboard changes.
- Source grep/review for `judge.orchestrated`, `_lesion`, `lesion`, `rebuild_from_artifacts`, and `bics/*.json` in runtime paths.

## Implementation Steps

1. Baseline the dirty worktree.
   - Run `git status --short` and classify current dirty files as pre-existing, especially dashboard, docs, scripts, config, and tests.
   - Do not edit anything yet.

2. Write the failing DuckDB judge evidence test.
   - In `tests/test_phase4_orchestrated_judge.py`, construct or patch a runtime where DuckDB has iteration/model data and `bics/iter*.json` is absent.
   - Assert the orchestrator path does not call `rebuild_from_artifacts` for normal runtime evidence and still passes a populated store to `ToolUsingJudge`.

3. Implement the smallest DuckDB evidence path.
   - Prefer reusing existing `diagnostics.duckdb`/`diagnostics_*.duckdb` stores or adding a focused copy/import helper over rebuilding from JSON.
   - Update `gecco/cli/run_judge_orchestrator.py` to use DuckDB evidence directly.
   - Keep `gecco/diagnostic_store/rebuild.py` as diagnostic/import-only if untouched; do not make it a normal runtime fallback.

4. Retire `judge.orchestrated` consistently.
   - Add validation tests in `tests/test_phase3_config_validation.py` rejecting explicit `judge.orchestrated`.
   - Update `config/schema.py` to reject the field with a clear message.
   - Update `gecco/cli/launch_distributed.py` to load validated config and infer orchestrator launch from judge presence/capabilities, not raw YAML `orchestrated`.
   - If CMG runtime still requires `cfg.judge.orchestrated`, update `gecco/run_gecco.py` and its focused tests so centralized judge enablement is inferred from supported validated state rather than a hidden compatibility attribute.
   - Remove `orchestrated:` from production configs if present.

5. Rename lesion-named configs and references.
   - Pick capability-oriented names, for example `*_capabilities_full.yaml`, `*_capabilities_random_feedback.yaml`, `*_capabilities_no_tools.yaml`, and `*_capabilities_summary_only.yaml`.
   - Update production config task names to remove lesion wording.
   - Update tests that reference old filenames.
   - Do not leave duplicate old files or compatibility aliases.

6. Tighten capability enforcement only where tests require it.
   - If tests show prohibited content is only removed after broad prompting, update `ToolUsingJudge` prompt construction to describe enabled capabilities explicitly.
   - Keep the change local; do not rewrite judge semantics broadly.

7. Run targeted tests and final audits.
   - Run the command set above.
   - Re-run grep/source checks for retired names and JSON runtime dependencies.
   - Compare final changed files to the baseline classification.

## Forbidden Patterns

- Do not re-enable JSON iteration artifacts just to make the orchestrator work.
- Do not keep old lesion config files as wrappers, symlinks, aliases, or duplicate examples.
- Do not make `judge.orchestrated: false` silently mean true.
- Do not add hidden fallback from DuckDB to JSON for normal runtime evidence.
- Do not mock away the evidence source in tests; tests must fail if JSON scanning is reintroduced as the runtime path.
- Do not touch dashboard files or add dashboard tests.
- Do not perform broad formatting or CRLF-only edits in forbidden paths.

## Acceptance Checklist

- `git status --short` was captured before implementation and final changed files are compared against it.
- Targeted pytest command passes under `conda run -n gecco_mh`.
- No implementation-owned changes under `gecco-mh-dashboard/**`, `scripts/**`, `bash/**`, `.claude/**`, or unrelated docs.
- No production config filename or task name contains `lesion`.
- Explicit `judge.orchestrated` fails validation with a clear error.
- Normal centralized judge runtime does not depend on `bics/iter*.json` or `rebuild_from_artifacts`.
- Any remaining JSON writes are inspection/export/diagnostic-only and are not runtime source of truth.
- Unverified risks, if any, are listed in the final response with the skipped command or reason.
