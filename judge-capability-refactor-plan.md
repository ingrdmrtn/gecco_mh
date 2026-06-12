# Judge Capability Refactor Plan

## Summary

Refactor judge capabilities so each option has one clear behavior: `attempted_models_overview` lists tried model names only, `performance_summary` reports metric-only performance, and `diagnostic_detail` gates detailed diagnostics. Drop `citations`, `coverage`, and `mechanistic_coherence` from the public capability surface. The main risk this plan prevents is performance, status, diagnostic, or recommendation information leaking into supposedly narrow judge modes via the existing generic LLM prompt path.

## Scope

Implementation must start on a new branch after inspecting the worktree:

- Run `git status --short` before implementation.
- Create a new branch, e.g. `git switch -c judge-capability-refactor`.
- Do not revert or modify unrelated pre-existing changes.
- Include tracked and untracked files in the initial and final scope audit.

Allowed implementation files:

- `config/schema.py`
- `gecco/construct_feedback/tool_judge.py`
- `gecco/construct_feedback/orchestrated.py` only if artifact assembly needs a small adjustment
- `config/archive/*.yaml` only for `judge.capabilities` key updates from retired names to the new taxonomy

Allowed test files:

- `tests/test_phase3_config_validation.py`
- `tests/test_phase4_orchestrated_judge.py`
- `tests/test_judge_orchestration.py`
- `tests/test_cmg_judge.py`

Allowed verification artifacts:

- None required. The final response receipt is sufficient if exact commands and results are reported.

Forbidden files/directories:

- `data*/`, `results*/`, generated judge artifacts, DuckDB files, notebooks, dashboard files
- `gecco/diagnostic_store/tools.py` unless inspection proves direct store queries in `tool_judge.py` cannot satisfy the contracts
- Broad config restructuring outside `judge.capabilities`

Production surfaces and entrypoints in scope:

- Config loading: `config.schema.JudgeCapability`, `JudgeConfig.validate_capabilities`, `GeCCoConfig.validate_judge_dependencies`
- Judge runtime: `ToolUsingJudge.get_feedback_analysis`, `ToolUsingJudge.synthesize_for_persona`
- Prompt layers: `_build_judge_system_prompt`, `_JUDGE_USER_TEMPLATE` or any replacement user-message builder, `_build_synthesis_prompt`
- Persistence surface: `run_orchestrated_judge_pipeline` and `FeedbackArtifact.synthesized_feedback`; persisted JSON must reflect the narrowed output but does not need a schema migration
- Runtime configs: archived YAML capability lists under `config/archive/`

Non-goals:

- No new diagnostic-store public tool unless required by tests
- No backward-compatibility aliases for `citations`, `coverage`, or `mechanistic_coherence`
- No changes to LLM provider loading, orchestration launch behavior, result storage layout, or model evaluation
- No new judge personas or prompt rewrites beyond capability gating

## Contracts

| Contract | Code path / symbol | Positive proof | Negative proof |
| --- | --- | --- | --- |
| `attempted_models_overview` means names only | `ToolUsingJudge.get_feedback_analysis`, `synthesize_for_persona` | With duplicate model names in the current iteration, feedback lists each unique name once in first-seen order | Output contains no metric values, BIC text, counts, statuses, trajectory, recommendations, diagnostics, model IDs, or LLM-generated prose; fallback generation and structured verdict extraction are not called |
| `performance_summary` means metric-only performance | deterministic summary helper in `tool_judge.py` | Feedback reports only metric values needed for current-iteration and overall/trajectory performance | Output contains no model names, model IDs, total/success/failure counts, statuses, diagnostic terms, or recommendations |
| Narrow deterministic modes compose | `ToolUsingJudge.get_feedback_analysis`, `synthesize_for_persona` | `['attempted_models_overview', 'performance_summary']` emits exactly the names section plus metric-only section | Generic judge prompt, tool loop, fallback generation, and persona LLM synthesis are not used for this pair |
| `diagnostic_detail` replaces `coverage` | `JudgeCapability`, `_apply_capability_postprocessing`, prompt builders | Schema accepts `diagnostic_detail`; prompts and feedback may include PPC, residual, recovery, and individual-difference detail only when capability is enabled | Schema rejects `coverage`; no disabled diagnostic prompts or feedback sections survive when `diagnostic_detail` is absent |
| `citations` and `mechanistic_coherence` are removed | `JudgeCapability`, runtime configs, tests | Runtime configs and tests no longer declare these capabilities | Config loading rejects these names; prompt messages no longer include citation-specific instructions or `cited_models` requirements |
| Prompt layers obey capabilities | `_build_judge_system_prompt`, user message builder, `_build_synthesis_prompt` | Tests capture actual messages sent to the LLM for a non-deterministic judge with and without `diagnostic_detail` | Captured messages do not contain retired capability instructions or diagnostic-detail terms when disabled |

## Tests And Verification

Add or update tests:

- `tests/test_phase3_config_validation.py`
  - Accepts `diagnostic_detail` capability.
  - Rejects `citations`, `coverage`, and `mechanistic_coherence` with clear validation errors.
  - Tests deduped names-only output for `attempted_models_overview` using a store stub with duplicate names and hidden metrics/statuses.
  - Tests metric-only output for `performance_summary`; assert no model names, counts, statuses, diagnostics, or recommendations.
  - Tests composed deterministic output for `['attempted_models_overview', 'performance_summary']` and verifies LLM/fallback paths are not called.
  - Updates postprocessing tests from `coverage` to `diagnostic_detail`.
- `tests/test_phase4_orchestrated_judge.py`
  - Ensure persisted `FeedbackArtifact.synthesized_feedback` preserves deterministic narrow outputs without adding recommendations or persona prose.
- `tests/test_judge_orchestration.py` and `tests/test_cmg_judge.py`
  - Update only if existing fixture capability names fail after schema cleanup.

Commands to run with the `gecco_mh` conda environment:

- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py`
- `conda run -n gecco_mh pytest tests/test_phase4_orchestrated_judge.py tests/test_judge_orchestration.py tests/test_cmg_judge.py`
- If available and reasonably fast, run the repository's configured lint/type command discovered from project metadata; otherwise report that no lint/type command was run.

Manual review checks:

- `git status --short` final changed-file list is within allowed scope.
- Search changed files for retired capability strings: `citations`, `coverage`, `mechanistic_coherence`, and confirm only rejection tests or explanatory comments remain.
- Review captured LLM-message tests to ensure they assert actual system/user/synthesis messages, not just helper output.

## Implementation Steps

1. Worktree and branch setup
   - Run `git status --short` and classify any pre-existing changes as in-scope or unrelated.
   - Create `judge-capability-refactor` branch before edits.
   - Stop and ask if unrelated dirty files overlap the intended edit targets.

2. Schema tests first
   - In `tests/test_phase3_config_validation.py`, add validation tests for accepted `diagnostic_detail` and rejected retired names.
   - Update `config/schema.py` `JudgeCapability` to remove `citations`, `coverage`, `mechanistic_coherence` and add `diagnostic_detail`.
   - Run the config validation subset until these tests fail/pass for the intended reason.

3. Deterministic attempted-models behavior
   - Add a helper in `tool_judge.py` to query current-iteration model names only, dedupe preserving order, and format the names-only section.
   - Add tests proving duplicate names are listed once and no forbidden terms leak.
   - Short-circuit `get_feedback_analysis()` before generic prompt construction for exact narrow deterministic sets containing only `attempted_models_overview` and/or `performance_summary`.

4. Metric-only performance summary
   - Refactor `_build_summary_only_feedback` or add a new helper so `performance_summary` contains metric values only.
   - Remove total/success/failure counts from `performance_summary` output.
   - Add tests that fail if model names, counts, statuses, diagnostics, or recommendations appear.

5. Composition and synthesis bypass
   - Update `synthesize_for_persona()` to return deterministic feedback unchanged for `attempted_models_overview`, `performance_summary`, or their combination.
   - Ensure structured verdict extraction and fallback generation are not called in these modes.
   - Verify orchestrated artifact persistence stores exactly the deterministic feedback map.

6. Diagnostic-detail gating and prompt cleanup
   - Rename capability checks from `coverage` to `diagnostic_detail`.
   - Remove citation-specific prompt additions and `cited_models` requirements from active prompt paths.
   - Make system, user, and synthesis prompt layers omit PPC/residual/recovery/individual-difference diagnostic instructions unless `diagnostic_detail` is enabled.
   - Add or update tests that capture actual LLM call messages for enabled and disabled diagnostic-detail cases.

7. Runtime config updates
   - Update only `judge.capabilities` lists in `config/archive/*.yaml` to remove retired names and use `diagnostic_detail` where the old config intended detailed diagnostics.
   - Do not change model/provider/evaluation/task settings.

8. Final verification
   - Run the targeted pytest commands.
   - Run final `git status --short` and compare changed files to the initial classification.
   - Search for retired names in changed runtime code/configs and resolve any remaining active use.

## Forbidden Patterns

- Do not leave narrow modes to rely on postprocessing after the generic judge prompt has already seen forbidden context.
- Do not mock away `get_feedback_analysis()` or `synthesize_for_persona()` in tests intended to prove output content.
- Do not hardcode config file names, result paths, branch names beyond the setup command, or model names in production code.
- Do not add compatibility aliases for retired capability names.
- Do not change diagnostic-store tools unless direct store queries are demonstrably insufficient.
- Do not introduce hidden global state, environment switches, or provider-specific behavior for capability semantics.
- Do not broaden this into a persona redesign, orchestration refactor, or result artifact migration.

## Acceptance Checklist

Contract: names-only attempted-models overview

- Test proves duplicate model names are listed once.
- Test proves no metrics, statuses, counts, diagnostics, recommendations, model IDs, or LLM prose appear.
- Test proves LLM/fallback paths are not called.

Contract: metric-only performance summary

- Test proves metric values appear.
- Test proves model names, counts, statuses, diagnostics, recommendations, and model IDs do not appear.

Contract: diagnostic-detail taxonomy

- Schema accepts `diagnostic_detail`.
- Schema rejects `citations`, `coverage`, and `mechanistic_coherence`.
- Prompt-capture tests prove diagnostic details are gated by `diagnostic_detail`.

Final checks

- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py` passes.
- `conda run -n gecco_mh pytest tests/test_phase4_orchestrated_judge.py tests/test_judge_orchestration.py tests/test_cmg_judge.py` passes.
- Final `git status --short` contains only allowed files.
- Changed-file review confirms no unrelated pre-existing changes were reverted or modified.
- Any unverified risk, skipped command, or scope expansion is reported in the final response.
