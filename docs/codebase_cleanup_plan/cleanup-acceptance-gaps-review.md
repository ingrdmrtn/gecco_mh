# Cleanup Acceptance Gaps Review

## Findings

### Medium: Capability-limited prompts still request prohibited recommendation/citation content

- Violates: Contract row 4 and Implementation Step 6.
- Evidence: the shared judge system prompt still tells the model to produce recommendations and citations for every non-short-circuited capability set:

```py
# gecco/construct_feedback/tool_judge.py:297-303
After gathering evidence across all angles, produce:
- A list of 3–5 concrete recommendations for the next iteration.
...
Be specific: cite model names, parameter names, BIC values, and r values from the data.
```

`synthesize_for_persona()` removes the "What to try next" section only from the later user synthesis prompt (`gecco/construct_feedback/tool_judge.py:1817-1827`), and post-processing strips output afterward (`gecco/construct_feedback/tool_judge.py:1871`). This still relies on broad all-capability prompting for the analysis/system prompt in no-recommendations/no-citations modes, contrary to the plan requirement to enforce limits before generation where plausible.

### Medium: DuckDB-first orchestrator test does not prove JSON rebuild/scanning would fail

- Violates: Contract row 1 and Forbidden Pattern 4.
- Evidence: `tests/test_phase4_orchestrated_judge.py:627-772` creates DuckDB stores with `inspection_output_enabled=False`, but it does not patch/assert that `rebuild_from_artifacts` is not called and does not make `bics/*.json` scanning fail if reintroduced. A future implementation could scan an empty `bics` directory or call a JSON fallback and this test would not necessarily catch the forbidden runtime dependency.
- Source review of `gecco/cli/run_judge_orchestrator.py:34-59` and `239-265` currently appears DuckDB-first, but the required proof is weaker than the plan specifies.

### Medium: Required lesion-free production filename/task-name proof is missing

- Violates: Contract row 3 and Tests And Verification requirement for `tests/test_phase3_config_validation.py`.
- Evidence: `tests/test_phase3_config_validation.py:29-39` hard-codes a representative `PRODUCTION_CONFIGS` list and `tests/test_phase3_config_validation.py:198-205` only checks those configs load. The suite does not assert that all production config filenames and loaded `task.name` values are free of `_lesion`, `lesion_`, or `lesion`.
- Review grep of `config/*.yaml` found no `lesion` matches, so the current config state appears clean, but the required automated proof is absent.

### Medium: Implementation receipt/baseline evidence is absent

- Violates: Worktree rules (`cleanup-acceptance-gaps-plan.md:51-57`), Acceptance Checklist (`cleanup-acceptance-gaps-plan.md:140-147`), and the review requirement to distinguish pre-existing dirty files from plan-owned changes.
- Evidence: no cleanup-acceptance implementation receipt was found under `docs/codebase_cleanup_plan/`; only this plan and review file exist for the acceptance-gaps work. Without a receipt or exact captured outputs, I could not verify pre-flight `git status --short`, final status, `git diff --name-only`, untracked-file audit, dashboard diff audit, command outputs, skipped checks, or changed-file ownership.

## Open Questions

- None blocking the findings above.

## Verification Summary

Observed by review:

- Read the full plan.
- Reviewed key implementation/test files in the allowed scope.
- Source checks performed with repository search/read:
  - `config/*.yaml` contains no `lesion` or `orchestrated` matches.
  - `gecco/run_gecco.py` no longer references `cfg.judge.orchestrated` in the reviewed runtime paths.
  - `gecco/cli/run_judge_orchestrator.py` does not import or call `rebuild_from_artifacts` and builds a judge store from `diagnostics*.duckdb`.
  - Remaining `rebuild_from_artifacts` references are in diagnostic/import-oriented code and tests, not in `run_judge_orchestrator.py`.

Not observed / still required:

- Required targeted pytest command was not evidenced by a receipt.
- Required `git status --short` before/after was not evidenced by a receipt.
- Required `git diff --name-only`, untracked-file audit, and `git diff --name-only -- gecco-mh-dashboard` were not evidenced by a receipt.
- Required grep/review outcomes were not recorded by an implementation receipt.

Contract rows verified:

- Retired `judge.orchestrated`: code review found schema rejection and no remaining CMG runtime dependency in reviewed paths.
- Dashboard remains out of scope: no reviewed implementation file is under `gecco-mh-dashboard/**`, but ownership cannot be fully verified without status/diff receipt.

Contract rows not fully verified:

- DuckDB-first centralized judge evidence: implementation appears DuckDB-first, but the required negative proof against JSON rebuild/scanning is incomplete.
- Lesion-free production config/test surface: current config grep is clean, but the required automated filename/task-name proof is missing.
- Capability-limited behavior: not satisfied for prompt-level no-recommendations/no-citations enforcement.

## Scope Audit

Files reviewed as implementation-owned candidates:

- `config/schema.py` — allowed.
- `gecco/cli/launch_distributed.py` — allowed.
- `gecco/cli/run_judge_orchestrator.py` — allowed.
- `gecco/run_gecco.py` — allowed only to remove the retired CMG runtime dependency; reviewed usage is consistent with that allowance.
- `gecco/diagnostic_store/store.py` — allowed if needed for minimal DuckDB import helper.
- `gecco/construct_feedback/tool_judge.py` — allowed if tightening capability prompts/post-processing is required; current prompt enforcement remains incomplete.
- `tests/test_phase2_cli_contract.py` — allowed.
- `tests/test_phase3_config_validation.py` — allowed.
- `tests/test_phase4_orchestrated_judge.py` — allowed.
- `tests/test_cmg_runtime.py` — allowed for focused CMG proof.
- `config/*.yaml` capability-oriented config updates — allowed.

Scope limitations:

- Changed-file ownership could not be conclusively audited because no implementation receipt/status output was available.
- No implementation-owned dashboard changes were identified during source review, but final dashboard diff proof is still missing.
