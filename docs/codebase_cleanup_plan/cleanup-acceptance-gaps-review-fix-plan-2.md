# Cleanup Acceptance Gaps Review Fix Plan 2

## Summary

Close the five review findings from `cleanup-acceptance-gaps-review-fix-findings.md`: fix the receipt filename/content, remove no-recommendations prompt language that still asks for next-iteration guidance, broaden production config discovery, and record the required focused verification. The main risk this prevents is treating the prior implementation as accepted while prompt behavior and config-surface proof remain weaker than the contract.

## Scope

Allowed implementation files:

- `gecco/construct_feedback/tool_judge.py` for capability-aware prompt text only.

Allowed test files:

- `tests/test_phase3_config_validation.py` for config-surface discovery and prompt-message assertions.

Allowed verification artifacts:

- Create/update `docs/codebase_cleanup_plan/cleanup-acceptance-gaps-review-fix-receipt.md`.
- Delete `docs/codebase_cleanup_plan/cleanup-acceptance-gaps-review-receipt.md` only if it is confirmed to be the previous implementation-owned wrong receipt, not user-authored work.

Forbidden files/directories:

- `gecco-mh-dashboard/**`
- `scripts/**`, `bash/**`, `.claude/**`
- `config/*.yaml` unless the broadened production-surface test finds a current lesion violation and the user approves updating configs.
- `tests/test_phase4_orchestrated_judge.py` unless a regression is introduced by this follow-up.
- Broad docs outside the two receipt paths named above and this plan.

Non-goals:

- Do not revisit the DuckDB-first orchestrator implementation or tests unless existing targeted tests fail after these edits.
- Do not rename configs proactively.
- Do not redesign judge synthesis or persona profiles beyond removing prohibited no-recommendations/no-citations prompt instructions.
- Do not add new receipt formats, manifests, or compatibility wrappers.

Worktree rules:

- Run `git status --short` before implementation and record a brief baseline classification in the correct receipt.
- Do not revert or modify unrelated pre-existing changes.
- Include tracked and untracked files in the final scope audit because receipt creation/deletion is expected.
- A changed file outside this scope means the implementation is incomplete unless the user approves the scope change.

## Contracts

| Contract | Code path or symbol | Positive proof | Negative proof |
| --- | --- | --- | --- |
| No-recommendations modes are not asked for next-iteration actionable guidance before generation. | `_build_judge_system_prompt()`, `_JUDGE_SYSTEM_PROMPT`, `_build_synthesis_prompt()`, and `synthesize_for_persona()` LLM-bound system/user messages in `gecco/construct_feedback/tool_judge.py`. | Prompt tests capture actual LLM-bound messages for a no-recommendations config and show descriptive feedback instructions still exist. | Same tests assert no-recommendations system/user prompts omit `recommendations`, `what to try next`, `key_recommendations`, `actionable feedback`, `suggestions`, and `improve the next iteration`; final output stripping alone is not sufficient. |
| Lesion-free config proof covers the explicit production config surface. | Production config discovery in `tests/test_phase3_config_validation.py`. | Test starts from `config/*.yaml`, explicitly excludes only `judge_tool_example.yaml` and `test_orchestrator.yaml` as non-production fixtures, loads every remaining YAML with `load_config()`, and asserts filename and `task.name` are lesion-free. | Test fails if any included config filename or loaded `task.name` contains `_lesion`, `lesion_`, or standalone `lesion`; it must not use `two_step_factors_*.yaml` as the initial surface. |
| Receipt evidence is at the required path and durable enough for review. | `docs/codebase_cleanup_plan/cleanup-acceptance-gaps-review-fix-receipt.md`. | Receipt includes pre-flight status, final status, final `git diff --name-only`, final untracked audit, dashboard diff output, exact test commands/results, and skipped-check rationale. | The old wrong receipt path is absent or explicitly documented as pre-existing user work; no final claim depends only on transient terminal output. |

## Tests And Verification

Tests to add/update:

- `tests/test_phase3_config_validation.py`: broaden `PRODUCTION_CONFIGS` discovery to start from `CONFIG_DIR.glob("*.yaml")` with explicit non-production exclusions.
- `tests/test_phase3_config_validation.py`: strengthen prompt-message assertions for no-recommendations modes to catch actionable next-iteration wording.

Commands to run under the project conda environment:

- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py -q`
- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py::test_orchestrated_persona_synthesis_captures_capability_limited_llm_messages -q`
- `conda run -n gecco_mh pytest tests/test_phase4_orchestrated_judge.py -q` only if `tests/test_phase4_orchestrated_judge.py` or orchestrator-related code changes unexpectedly.

Manual checks to record in the correct receipt:

- Pre-flight and final `git status --short`.
- Final `git diff --name-only`.
- Final `git diff --name-only -- gecco-mh-dashboard`.
- Final untracked-file audit from `git status --short`.
- Receipt path check confirming `cleanup-acceptance-gaps-review-fix-receipt.md` exists and the wrong receipt path is either removed or classified.

## Implementation Steps

1. Baseline and receipt correction.
   - Run `git status --short`.
   - Create `docs/codebase_cleanup_plan/cleanup-acceptance-gaps-review-fix-receipt.md` and record the baseline.
   - If `cleanup-acceptance-gaps-review-receipt.md` is the prior implementation-owned wrong receipt, move its useful content into the correct receipt and delete the wrong file; otherwise leave it untouched and classify it in the receipt.

2. Strengthen no-recommendations prompt test.
   - In `tests/test_phase3_config_validation.py`, extend the existing prompt-capture test for no-recommendations configs to assert absence of actionable next-iteration wording in the actual system and user messages.
   - Run the focused prompt-test command and confirm it fails before the prompt fix or note if the current failure is already evident by inspection.

3. Remove prohibited prompt wording for disabled recommendations.
   - In `gecco/construct_feedback/tool_judge.py`, make next-iteration improvement/guidance wording conditional on `recommendations` where it implies concrete suggestions.
   - Preserve descriptive comparative feedback for no-recommendations modes, so the judge can still summarize what worked and what did not.
   - Do not remove citation/metric analysis language unless the no-citations prompt test fails for citation-instruction wording.

4. Broaden production config discovery.
   - In `tests/test_phase3_config_validation.py`, define production configs from all `config/*.yaml` minus explicit non-production fixture exclusions: `judge_tool_example.yaml` and `test_orchestrator.yaml`.
   - Keep the existing load-through-schema and filename/`task.name` lesion assertions for every included config.
   - If a newly included config cannot load or contains lesion wording, stop and ask whether it is production or should be added to the explicit exclusion list.

5. Verify and complete the receipt.
   - Run the required commands.
   - Record exact command results and final manual-check outputs in `cleanup-acceptance-gaps-review-fix-receipt.md`.
   - Compare final changed files with the allowed scope before handoff.

## Forbidden Patterns

- Do not satisfy prompt tests by checking helper output instead of actual LLM-bound system/user messages.
- Do not keep broad prompt wording that asks disabled no-recommendations modes for actionable next-iteration improvements and rely on post-processing.
- Do not replace the config-surface check with a hard-coded representative or prefix-only list.
- Do not silently exclude config YAMLs without naming the excluded files and why they are non-production.
- Do not edit dashboard, scripts, bash files, or unrelated configs.
- Do not introduce broad prompt rewrites, persona-profile redesign, or opportunistic adjacent fixes.

## Acceptance Checklist

Prompt contract:

- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py::test_orchestrated_persona_synthesis_captures_capability_limited_llm_messages -q` passes.
- Captured no-recommendations LLM-bound prompts omit recommendation/actionable-next-iteration wording while retaining descriptive feedback instructions.

Config-surface contract:

- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py -q` passes.
- Production config discovery starts from `config/*.yaml` and explicitly excludes only named non-production fixtures.

Receipt/scope contract:

- `docs/codebase_cleanup_plan/cleanup-acceptance-gaps-review-fix-receipt.md` exists and contains exact baseline/final status, final diff names, untracked audit, dashboard diff output, commands/results, and skipped-check rationale.
- `docs/codebase_cleanup_plan/cleanup-acceptance-gaps-review-receipt.md` is removed if it was the previous wrong implementation-owned receipt, or classified if left untouched.
- Final changed files are limited to the allowed scope, with no implementation-owned dashboard changes.
