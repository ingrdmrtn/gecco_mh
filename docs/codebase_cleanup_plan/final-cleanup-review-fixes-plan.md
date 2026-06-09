# Final Cleanup Review Fixes Plan

## Summary

Fix the remaining review findings from `final-cleanup-genuine-issues-review.md`: dict-backed nested `naive_ideation` must work through the actual generation path, README must stop referencing deleted demo script surfaces, and final verification evidence must be durable enough to review. The main risk this prevents is accepting a dispatch-only fix while nested dict config still fails at runtime.

## Scope

Allowed implementation files:

- `gecco/candidate_generation.py` for nested dict/object access inside `generate_models_naive()` and related naive-ideation logic.
- `README.md` only to remove stale deleted script references and “demo scripts” wording from active documentation.

Allowed test files:

- `tests/test_cmg_runtime.py` for a regression test that exercises the actual naive generation path with dict-backed nested config.
- Existing focused README/docs test only if one already asserts public docs command surfaces; otherwise use grep/manual review rather than adding a broad docs test.

Allowed verification artifacts:

- `docs/codebase_cleanup_plan/final-cleanup-review-fixes-receipt.md` with baseline/final status, exact commands, grep outputs, dashboard diff audit, and skipped-check rationale.

Forbidden files/directories:

- `gecco-mh-dashboard/**`.
- `scripts/**`, `bash/**`, `.claude/**`.
- `config/*.yaml`.
- Runtime files outside `gecco/candidate_generation.py`.
- Test files outside the allowed test surface unless the user approves.

Explicit non-goals:

- Do not revisit client-profile fixes already made in `gecco/coordination.py`, `gecco/run_gecco.py`, or `gecco/construct_feedback/orchestrated.py` unless the new focused test proves `candidate_generation.py` cannot be fixed alone.
- Do not address judge JSON persistence, utility script deletion, dashboard behavior, provider loading, or CLI packaging.
- Do not create compatibility wrappers for deleted scripts.

Worktree rules:

- Run `git status --short` before implementation and record it in the receipt.
- Briefly classify pre-existing dirty tracked/untracked files, especially dashboard files.
- Do not revert or modify unrelated pre-existing changes.
- Scope audits must include tracked and untracked files.
- A changed file outside this scope means the implementation is incomplete unless the user approves the scope change.

## Contracts

| Contract | Code Path Or Symbol | Positive Proof | Negative Proof |
| --- | --- | --- | --- |
| Dict-backed nested `naive_ideation` works through actual naive generation. | `CandidateGenerator.generate_models_naive()` and any helper used to read `persona`, `translation_preamble`, and `enabled`. | A test passes a dict-backed `cfg.clients["generator"]["naive_ideation"]` with `enabled`, `persona`, and `translation_preamble`, calls the real naive generation path, and observes those values in the generated prompt flow without `AttributeError`. | The test must not mock `generate_models_naive()` itself; it should fail if code uses `naive_cfg.persona` or `getattr(naive_cfg, ...)` as the only dict path. |
| README active docs do not point users at deleted demo script surfaces. | `README.md` repository structure and usage sections. | Grep/manual review shows no `decision_making_demo.py`, `two_step_demo.py`, or “Quick start with demo scripts” wording remains in active docs. | Do not replace stale references with `python scripts/...` commands or new shim instructions. |
| Verification evidence is reviewable after handoff. | `final-cleanup-review-fixes-receipt.md`. | Receipt contains pre-flight/final status, final changed files, targeted pytest output, README grep output, dashboard diff audit, and untracked-file audit. | Final claims must not depend only on transient terminal output or unstated assumptions about pre-existing dashboard changes. |

## Tests And Verification

Tests to add or update:

- Update `tests/test_cmg_runtime.py` so dict-backed nested `naive_ideation` is exercised through the real `CandidateGenerator.generate_models_naive()` path. Mock only external LLM/generation dependencies, not `generate_models_naive()`.
- The test should assert the prompt-building or generation-call input includes the configured `persona` and `translation_preamble` from a dict-backed nested config.

Commands to run:

- `conda run -n gecco_mh pytest tests/test_cmg_runtime.py -q`
- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py tests/test_cmg_runtime.py tests/test_phase4_orchestrated_judge.py -q`

Manual review checks to record in the receipt:

- `git status --short` before and after implementation.
- `git diff --name-only` and final untracked-file audit from `git status --short`.
- `git diff --name-only -- gecco-mh-dashboard`.
- README grep checks for `decision_making_demo.py`, `two_step_demo.py`, `Quick start with demo scripts`, `python scripts/two_step_demo.py`, `python scripts/decision_making_demo.py`, `shared_registry.json`, and `JSON registry`.

## Implementation Steps

1. Baseline and receipt setup.
   - Run `git status --short`.
   - Create `docs/codebase_cleanup_plan/final-cleanup-review-fixes-receipt.md` and record baseline status plus pre-existing dirty-file classification.

2. Write the failing runtime test.
   - In `tests/test_cmg_runtime.py`, add or strengthen the dict-backed nested `naive_ideation` test.
   - Ensure the test calls `generate_models_naive()` through the actual `CandidateGenerator` flow and would fail on `naive_cfg.persona`.

3. Fix nested config reads minimally.
   - In `gecco/candidate_generation.py`, read `enabled`, `persona`, and `translation_preamble` through the existing dict/object helper or a small local equivalent.
   - Do not introduce broad config normalization or mutate `cfg.clients`.

4. Remove stale README surfaces.
   - Delete or rewrite repository-structure references to nonexistent `decision_making_demo.py` and `two_step_demo.py`.
   - Change “Quick start with demo scripts” to active `gecco` CLI wording.
   - Keep README edits limited to stale script/DuckDB cleanup wording.

5. Verify and audit.
   - Run the focused and targeted pytest commands.
   - Run the README grep checks and dashboard diff audit.
   - Record exact outputs and final changed-file list in the receipt.

## Forbidden Patterns

- Do not mock `generate_models_naive()` in the regression test for this bug.
- Do not use object-only access for nested dict-backed config values.
- Do not add global config conversion, hidden defaults, or compatibility wrappers.
- Do not restore deleted demo scripts or add script shims.
- Do not edit dashboard files or classify implementation-owned dashboard edits as pre-existing without evidence.
- Do not perform opportunistic README rewrites beyond the stale surfaces named here.

## Acceptance Checklist

Nested `naive_ideation` contract:

- `tests/test_cmg_runtime.py` includes a real-path regression test for dict-backed nested `naive_ideation`.
- `conda run -n gecco_mh pytest tests/test_cmg_runtime.py -q` passes.
- The broader targeted pytest command passes or any unrelated failures are recorded with evidence.

README contract:

- Receipt records grep outputs showing stale deleted-script references and JSON registry wording are absent from active README surfaces.
- README replacement wording points to the active `gecco` CLI, not `scripts/*.py` shims.

Verification/scope contract:

- `docs/codebase_cleanup_plan/final-cleanup-review-fixes-receipt.md` exists and contains exact baseline/final status, command outputs, grep outputs, dashboard diff audit, changed-file list, and skipped-check rationale.
- Final `git diff --name-only` is limited to allowed implementation/test/verification files unless user approval is recorded.
- Final `git diff --name-only -- gecco-mh-dashboard` is empty for implementation-owned changes, or remaining dashboard diffs are explicitly classified as pre-existing with supporting status evidence.
