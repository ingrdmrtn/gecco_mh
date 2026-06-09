# Cleanup Acceptance Gaps Review Fix Plan

## Summary

Fix the four remaining acceptance-review gaps: enforce capability-limited judge prompts before generation, strengthen DuckDB-first orchestrator tests against JSON fallback, automate lesion-free production config proof, and capture implementation baseline/final evidence. The main risk this prevents is accepting cleanup work that only appears correct by source review or post-processing while tests still allow forbidden runtime or prompt behavior.

## Scope

Allowed implementation files:

- `gecco/construct_feedback/tool_judge.py` for capability-aware prompt construction only.
- `gecco/cli/run_judge_orchestrator.py` only if the stronger DuckDB test exposes a real JSON rebuild/scanning path.
- `config/*.yaml` only if the new lesion-free test finds production filename or `task.name` violations.

Allowed test files:

- `tests/test_phase4_orchestrated_judge.py`
- `tests/test_phase3_config_validation.py`
- Existing focused judge prompt/output test file if one already covers `ToolUsingJudge` or `synthesize_for_persona`; otherwise add one small test file under `tests/` with a `test_tool_judge_*` name.

Allowed verification artifacts:

- `docs/codebase_cleanup_plan/cleanup-acceptance-gaps-review-fix-receipt.md` with baseline/final status, changed-file list, command outputs, and skipped checks.
- Temporary pytest files under `tmp_path` only.

Forbidden files/directories:

- `gecco-mh-dashboard/**`
- `scripts/**`, `bash/**`, `.claude/**`
- Broad docs outside the receipt named above.
- Unrelated model-fitting, PPC, HBI, provider backend, report export, or dashboard code.

Non-goals:

- Do not rework `judge.orchestrated` retirement unless these focused tests expose a direct regression.
- Do not rename configs proactively; only fix violations found by the lesion-free production-surface test.
- Do not redesign judge synthesis, recommendations, citations, or persona behavior beyond capability prompt constraints.
- Do not add compatibility aliases or JSON fallback paths.

Worktree rules:

- Run `git status --short` before implementation and record it in the receipt.
- Classify pre-existing dirty tracked and untracked files before edits, including any dashboard files.
- Do not revert or modify unrelated pre-existing changes.
- Scope audits must include tracked and untracked files because one allowed output is a new receipt and one new test file is possible.
- A changed file outside the allowed scope means implementation is incomplete unless the user approves the scope change.

## Contracts

| Contract | Code path or symbol | Positive proof | Negative proof |
| --- | --- | --- | --- |
| Capability-limited judge modes do not ask the LLM for prohibited recommendations or citations before generation. | `ToolUsingJudge` prompt construction in `gecco/construct_feedback/tool_judge.py`, including the shared judge system/developer prompt and the synthesis user prompt used by `synthesize_for_persona()`. | Tests capture the actual messages/prompts sent for a no-`recommendations` mode and a no-`citations` or `attempted_models_overview` mode and show allowed content remains. | Same tests assert prohibited phrases such as `recommendations`, `What to try next`, `cite model names`, `BIC values`, and `r values` are absent from prompt layers for modes that disable them, not merely stripped from final output. |
| Normal orchestrator evidence remains DuckDB-first and cannot silently rebuild from JSON. | `run_orchestrator` path in `gecco/cli/run_judge_orchestrator.py`; any imported DuckDB store helper it calls. | `tests/test_phase4_orchestrated_judge.py` builds runtime evidence from `diagnostics*.duckdb` with JSON inspection output disabled and still reaches judge construction with populated evidence. | Test patches `rebuild_from_artifacts` to fail if called and makes `bics/*.json` scanning fail if attempted, so an empty or regenerated JSON path cannot pass. |
| Production config filenames and loaded task names are lesion-free. | Production config discovery in `tests/test_phase3_config_validation.py`; loaded `Config.task.name`. | Test discovers the production config surface from `config/*.yaml`, loads each production config, and passes with current clean names. | Test fails if any production config filename or loaded `task.name` contains `_lesion`, `lesion_`, or standalone `lesion`; do not rely on a hard-coded representative list. |
| Baseline and final verification evidence is durable enough for review. | Receipt `docs/codebase_cleanup_plan/cleanup-acceptance-gaps-review-fix-receipt.md`. | Receipt contains pre-flight `git status --short`, final `git status --short`, final `git diff --name-only`, untracked-file audit, dashboard diff audit, and exact test commands/results. | Receipt identifies skipped commands or out-of-scope dirty files; no final claim relies only on transient terminal output. |

## Tests And Verification

Add or update tests:

- `tests/test_phase4_orchestrated_judge.py`: strengthen the existing DuckDB-first orchestrator test so it fails on `rebuild_from_artifacts` calls or `bics/*.json` scanning.
- `tests/test_phase3_config_validation.py`: replace or supplement the representative production config check with dynamic `config/*.yaml` discovery and lesion-free filename/`task.name` assertions.
- Existing `ToolUsingJudge` prompt tests, or one new focused `tests/test_tool_judge_capability_prompts.py`: capture actual LLM-bound messages for no-recommendations and no-citations/overview capability modes.

Commands to run under the project conda environment:

- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py tests/test_phase4_orchestrated_judge.py -q`
- `conda run -n gecco_mh pytest <focused-tool-judge-test-file> -q`
- If `gecco/cli/run_judge_orchestrator.py` changes: `conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py -q`

Manual review checks to record in the receipt:

- `git status --short` before edits and after verification.
- `git diff --name-only` plus an untracked-file audit from final status.
- `git diff --name-only -- gecco-mh-dashboard` must show no implementation-owned dashboard changes.
- Review `gecco/construct_feedback/tool_judge.py` for remaining shared prompt text that asks disabled modes for recommendations or citations.
- Review `gecco/cli/run_judge_orchestrator.py` for normal-runtime imports/calls to `rebuild_from_artifacts` or JSON `bics` iteration scanning if that file changes.

## Implementation Steps

1. Baseline and receipt setup.
   - Run `git status --short` before edits.
   - Create `docs/codebase_cleanup_plan/cleanup-acceptance-gaps-review-fix-receipt.md` and record pre-existing tracked/untracked dirty files.
   - Do not edit implementation files before the baseline is recorded.

2. Strengthen DuckDB-first test first.
   - Update `tests/test_phase4_orchestrated_judge.py` around the existing DuckDB-first orchestrator test.
   - Patch `rebuild_from_artifacts` to raise if called.
   - Make `bics/*.json` unavailable or monkeypatch JSON/glob scanning so a JSON dependency fails loudly.
   - Run the focused Phase 4 test and only edit `gecco/cli/run_judge_orchestrator.py` if the test reveals a real runtime dependency.

3. Add dynamic lesion-free production config proof.
   - Update `tests/test_phase3_config_validation.py` to discover production config files from `config/*.yaml` instead of relying only on `PRODUCTION_CONFIGS`.
   - Load each production config through the normal config loader and assert filename and `task.name` are lesion-free.
   - If this fails, update only the violating `config/*.yaml` files and direct test references.

4. Add prompt-level capability tests.
   - Locate the narrowest existing test surface that can instantiate or drive `ToolUsingJudge`/`synthesize_for_persona` without making external LLM calls.
   - Capture the actual system/developer/user messages passed to the LLM for no-recommendations and no-citations or overview-only capability modes.
   - Assert disabled prompt content is absent before generation and allowed analysis content remains present.

5. Tighten capability prompt construction.
   - In `gecco/construct_feedback/tool_judge.py`, make the shared prompt conditional on enabled capabilities so disabled modes are not asked for recommendations or citations.
   - Keep existing post-processing as defense in depth, but do not rely on it as the primary enforcement.
   - Avoid broad wording changes for fully capable modes unless a test requires it.

6. Verify and complete receipt.
   - Run the targeted commands listed above.
   - Record command outputs, skipped checks, final status, final changed files, untracked files, and dashboard diff audit in the receipt.
   - Compare final changed files against the allowed scope before handing off.

## Forbidden Patterns

- Do not satisfy capability tests by asserting on helper return strings while ignoring actual LLM-bound system/developer/user messages.
- Do not keep a broad shared judge prompt that requests recommendations or citations for disabled capability sets and rely only on output stripping.
- Do not mock away the orchestrator evidence source so JSON rebuild/scanning could pass undetected.
- Do not re-enable JSON inspection artifacts or add DuckDB-to-JSON fallback for normal orchestrator runtime.
- Do not hard-code a small representative production config list for lesion checks.
- Do not rename unrelated configs, add compatibility wrappers, or duplicate old config files.
- Do not edit dashboard files or perform opportunistic adjacent refactors.

## Acceptance Checklist

Capability prompt contract:

- Focused prompt tests pass and capture actual LLM-bound messages.
- Disabled recommendation/citation modes do not include prohibited recommendation/citation instructions before generation.

DuckDB-first contract:

- Phase 4 test passes with `rebuild_from_artifacts` patched to fail.
- Phase 4 test fails if `bics/*.json` scanning is reintroduced as the normal runtime evidence path.

Lesion-free config contract:

- Dynamic production config test passes for every `config/*.yaml` production config.
- No production config filename or loaded `task.name` contains `_lesion`, `lesion_`, or standalone `lesion`.

Receipt and scope contract:

- `docs/codebase_cleanup_plan/cleanup-acceptance-gaps-review-fix-receipt.md` contains baseline status, final status, final changed files, untracked audit, dashboard diff audit, commands/results, and skipped checks.
- Final changed files are limited to the allowed scope or have explicit user approval.
- `git diff --name-only -- gecco-mh-dashboard` has no implementation-owned changes.
