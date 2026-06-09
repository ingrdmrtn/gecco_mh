# Final Cleanup Genuine Issues Plan

## Summary

Fix the confirmed final cleanup issues: validated config client profiles must work with dict-backed `clients`, dashboard changes must be removed or explicitly classified as pre-existing, and public docs must stop pointing users at deleted script entrypoints or JSON registry state. The main risk this prevents is accepting the cleanup while launch/runtime profiles are broken and review scope still includes out-of-scope or stale surfaces.

## Scope

Allowed implementation files:

- `gecco/coordination.py` for `apply_client_profile()` dict/object handling.
- `gecco/candidate_generation.py` for client profile lookup during generation.
- `gecco/run_gecco.py` for CMG generator client profile lookup.
- `gecco/construct_feedback/orchestrated.py` for persona/client config lookup and suffix extraction.
- `README.md` only to replace stale deleted-script commands and JSON registry wording with the active `gecco ...` CLI and DuckDB state wording.

Allowed test files:

- `tests/test_phase3_config_validation.py` for validated-config/client profile tests if an existing config contract test is the narrowest fit.
- `tests/test_cmg_runtime.py` for CMG generator profile/runtime behavior.
- `tests/test_phase4_orchestrated_judge.py` for persona synthesis dict-backed `clients` behavior.
- Add one focused test file under `tests/` only if these files are too crowded.

Allowed verification artifacts:

- None required. The final response may summarize command results.

Forbidden files/directories:

- `gecco-mh-dashboard/**` except to revert implementation-owned cleanup changes back to the pre-cleanup baseline, with user approval if ownership is unclear.
- `scripts/**`, `bash/**`, `.claude/**`.
- `config/*.yaml` unless a focused test proves the profile bug is caused by invalid test fixture shape rather than code.
- Broad docs outside `README.md` and this plan.

Explicit non-goals:

- Do not resolve the ambiguous judge JSON trace/DuckDB persistence question in this step.
- Do not delete or migrate remaining utility scripts such as `scripts/test_fit_model.py`.
- Do not redesign config models, provider loading, prompt behavior, or dashboard behavior.
- Do not add compatibility wrappers for old entrypoints.

Worktree rules:

- Run `git status --short` before implementation and briefly classify pre-existing dirty tracked/untracked files, especially dashboard files.
- Do not revert or modify unrelated pre-existing changes.
- Include tracked and untracked files in the final scope audit.
- A changed file outside the allowed scope means the implementation is incomplete unless the user approves the scope change.

## Contracts

| Contract | Code Path Or Symbol | Positive Proof | Negative Proof |
| --- | --- | --- | --- |
| Validated dict-backed client profiles work everywhere they are consumed. | `apply_client_profile()`, `CandidateGenerator.generate_iteration()`, `CandidateGenerator.generate_non_cmg_iteration()`, `GeCCoModelSearch._run_cmg_generator_iteration()`, `_resolve_synthesis_personas()`, `_persona_suffix()`. | Tests load a real validated config with `clients: {generator: ...}` and show the generator profile/suffix is found and applied. | Tests fail if implementation uses only `getattr(cfg.clients, name, None)` or `vars(cfg.clients)` for dict clients. |
| Public docs point to active production launch surfaces. | `README.md` command examples and distributed-state wording. | Review/grep shows no `python scripts/two_step_demo.py`, `python scripts/decision_making_demo.py`, or “JSON registry” wording remains in active usage sections. | Do not replace deleted scripts with new shim commands or mention old script paths as supported paths. |
| Dashboard remains out of final implementation scope. | `gecco-mh-dashboard/**`. | Final `git diff --name-only -- gecco-mh-dashboard` is empty for implementation-owned changes, or the final response clearly classifies remaining dashboard diffs as pre-existing/user-owned. | Do not add dashboard tests or behavior changes to satisfy cleanup acceptance. |

## Tests And Verification

Tests to add or update:

- Add a focused test for `apply_client_profile(load_config("config/two_step_factors_cmg.yaml"), "generator")` proving `cfg.llm.models_per_iteration`, `temperature`, and `system_prompt_suffix`-driven prompt text are applied without `TypeError`.
- Add or update a CMG generation/runtime test proving dict-backed `cfg.clients` enables generator `naive_ideation` or profile lookup when `client_id="generator"`.
- Add or update a persona synthesis test proving `_resolve_synthesis_personas()` and `_persona_suffix()` handle validated dict-backed client configs and pass non-empty persona suffix to `synthesize_for_persona()`.

Commands to run:

- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py tests/test_cmg_runtime.py tests/test_phase4_orchestrated_judge.py -q`
- If a new focused test file is added, include it in the command above or run `conda run -n gecco_mh pytest <new-test-file> -q`.

Manual review checks:

- `git status --short` before and after implementation.
- `git diff --name-only` plus final untracked-file audit from `git status --short`.
- `git diff --name-only -- gecco-mh-dashboard`.
- `grep`/search in `README.md` for `python scripts/two_step_demo.py`, `python scripts/decision_making_demo.py`, `shared_registry.json`, and `JSON registry`.

## Implementation Steps

1. Baseline the worktree.
   - Run `git status --short`.
   - Classify existing dashboard diffs as pre-existing/user-owned or implementation-owned before editing.

2. Write failing profile lookup tests.
   - Add the `apply_client_profile()` validated-config test first.
   - Add the CMG generator/profile lookup test and persona suffix test before changing runtime code.

3. Fix dict/object config access minimally.
   - Add one small local helper only if needed, such as `_mapping_get(obj, key)`, in files that need it; do not introduce a broad config abstraction.
   - Update `apply_client_profile()` to support dict and object clients, and dict/object nested `llm` profile sections.
   - Update CMG generation and orchestrated persona lookup sites to use the same semantics.

4. Update public README examples.
   - Replace deleted demo script commands with active `gecco` CLI examples already supported by `gecco/cli/__init__.py`.
   - Replace JSON registry wording with DuckDB canonical state wording.
   - Keep edits surgical; do not rewrite unrelated README sections.

5. Handle dashboard scope.
   - If dashboard diffs are implementation-owned cleanup edits, remove those edits only after confirming they are not user-owned.
   - If they are pre-existing or user-owned, leave them untouched and classify them in the final response.

6. Verify and audit.
   - Run the targeted pytest command.
   - Run the manual README and dashboard scope checks.
   - Compare final changed files against the allowed scope.

## Forbidden Patterns

- Do not use `getattr(cfg.clients, name, None)` as the only client-profile lookup path.
- Do not use `vars(clients)` unless the value is known not to be a dict.
- Do not mutate unrelated config structure to make tests pass.
- Do not add script shims or restore deleted script entrypoints.
- Do not edit dashboard behavior or add dashboard tests as part of this final cleanup.
- Do not broaden this task into judge JSON trace persistence or utility script deletion.

## Acceptance Checklist

Client-profile contract:

- Targeted tests pass under `conda run -n gecco_mh`.
- Validated `config/two_step_factors_cmg.yaml` generator profile applies without error.
- CMG and persona synthesis tests fail if dict-backed `clients` lookup regresses.

Docs contract:

- `README.md` no longer references deleted demo scripts or JSON registry state in active usage sections.
- Replacement commands use the existing `gecco` CLI surface.

Scope contract:

- Final `git diff --name-only` is limited to allowed files unless the user approved otherwise.
- Final `git diff --name-only -- gecco-mh-dashboard` is empty for implementation-owned changes, or remaining dashboard diffs are explicitly classified as pre-existing/user-owned.
- Final status includes tracked and untracked files in the audit.
