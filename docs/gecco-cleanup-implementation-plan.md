# GeCCo Cleanup Implementation Plan

## Summary

Implement the agreed conservative cleanup against the current working-tree architecture after the user prepares a clean branch. The change removes unused modules and compatibility-only APIs, introduces small typed/helper boundaries for duplicated code, replaces ad-hoc launcher submission logic with a shared launch-plan executor, and refreshes current docs. The main risk this plan prevents is accidentally changing distributed launch/runtime behaviour, persisted artefacts, or registry/DuckDB state flow while deleting stale code.

## Scope

Before implementation, run `git status --short`. If the worktree is dirty, stop and ask the user to prepare the clean branch; do not edit around dirty files.

Allowed production files:

- `gecco/utils.py`
- `gecco/candidate_generation.py`
- `gecco/cli/launcher_utils.py` (new)
- `gecco/cli/launch_distributed.py`
- `gecco/cli/launch_cmg_distributed.py`
- `gecco/cli/run_test_evaluation.py`
- `gecco/construct_feedback/tool_judge.py`
- Deletions: `gecco/prompt_builder/guardrails.py`, `gecco/offline_evaluation/data_structures.py`, `gecco/diagnostic_store/populate.py`, `gecco/diagnostic_store/rebuild.py`
- Targeted removals in `gecco/offline_evaluation/utils.py` and `gecco/structured_output.py`

Allowed test files:

- `tests/test_cli_launcher_utils.py` (new)
- `tests/test_utils.py` or closest existing utility test file
- `tests/test_phase6_candidate_generator.py`
- `tests/test_phase2_cli_contract.py`
- `tests/test_phase3_config_validation.py`
- `tests/test_phase4_orchestrated_judge.py`
- `tests/test_cmg_judge.py`
- Delete rebuild-only test files or test cases, including `tests/test_rebuild_fallback.py` if it only covers `diagnostic_store/rebuild.py`

Allowed docs:

- `README.md`
- `docs/judge_implementation_record.md`
- `docs/centralised_judge_implementation.md`
- `docs/tool_judge_split_note.md` (new)
- Delete existing `docs/codebase_cleanup_plan/*` files and the folder if empty

Allowed verification artifacts:

- No persisted receipt is required. The final response can list commands run and unverified risks.

Forbidden files/directories:

- `gecco-mh-dashboard/`
- `config/` files
- `bash/` scripts
- LLM prompt text in `gecco/prompt_builder/prompt.py` unless a failing test proves it is required
- Runtime persistence schema files except tests/docs explicitly require a compile/import fix

Production surfaces in scope:

- CLI launch behaviour: `python -m gecco run distributed`, `python -m gecco run cmg-distributed`
- Candidate generation payloads: `CandidateGenerator.generate_iteration()` and `generate_non_cmg_iteration()`
- Judge public surface: remove direct `ToolUsingJudge.get_feedback()` wrapper while preserving orchestrated paths
- Test-evaluation user guidance when no DuckDB store is written

Non-goals:

- Do not move registry writes out of generation/evaluation services.
- Do not change JSON/TXT sidecar output behaviour under `results/.../models`, `bics`, `feedback`, `judge`, or `reviews`.
- Do not require `RunContext` everywhere or remove `ArtifactStore` fallback path support.
- Do not split `tool_judge.py` in this pass.
- Do not add compatibility stubs for deleted APIs.

## Contracts

| Contract | Code path | Positive proof | Negative proof |
| --- | --- | --- | --- |
| Dead APIs are gone | Deleted modules/functions and stale imports | Import/reference grep no longer finds active uses; relevant tests pass | Importing `gecco.diagnostic_store.rebuild` or calling `ToolUsingJudge.get_feedback` is not supported by tests or docs |
| Launcher behaviour is preserved | `launch_distributed.py`, `launch_cmg_distributed.py`, `launcher_utils.py` | Existing CLI contract tests still observe the same effective sbatch commands, dry-run/local semantics, and job-id dependency chaining | Tests fail if launchers hide command construction behind changed flag order, changed script paths, or mocked-away executor behaviour |
| Candidate payload shape is unchanged | `CandidatePayload`, `CandidateGenerationResult.candidates` | Generator tests show returned/published candidates are still dicts with existing fields | Tests fail if evaluator/registry must understand `CandidatePayload` objects |
| Runtime state flow is unchanged | `SharedRegistry`, `DiagnosticStore`, `ArtifactStore` callers | Existing phase 4/5/6/7 tests covering persistence, registry publication, and judge feedback still pass | No new fallback rebuild/import path or JSON source-of-truth path is introduced |
| Docs reflect current architecture | README and current judge docs | Docs state DuckDB/registry are canonical and JSON sidecars are inspection/audit artefacts | Docs contain no `rebuild_from_artifacts`, `judge.orchestrated`, or direct `ToolUsingJudge.get_feedback()` guidance |

## Tests And Verification

Add or update tests:

- `tests/test_cli_launcher_utils.py`: test `LaunchExecutor` job-id parsing, dry-run no-submit behaviour, and label-based dependency builders using real `LaunchPlan`/`LaunchCommand` objects with a fake runner.
- `tests/test_phase2_cli_contract.py`: preserve launcher route tests and assert effective command strings for distributed/CMG launchers through the new abstraction.
- `tests/test_phase6_candidate_generator.py`: test `CandidatePayload` normalises one parsed model and that generation results still expose `list[dict]` with existing keys.
- Utility test: test `mapping_get()` for `None`, dict, and attribute containers.
- Judge tests: remove or rewrite `ToolUsingJudge.get_feedback()` callers to use `run_orchestrated_judge_pipeline()` or lower-level analysis/synthesis methods.
- CMG/orchestrator tests: remove stale `rebuild_from_artifacts` patches and assert the current DuckDB-source path instead.
- Delete rebuild-only tests tied to `gecco.diagnostic_store.rebuild`.

Run with the project environment:

- `conda run -n gecco_mh pytest tests/test_cli_launcher_utils.py tests/test_phase2_cli_contract.py tests/test_phase6_candidate_generator.py`
- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py tests/test_phase4_orchestrated_judge.py tests/test_cmg_judge.py`
- `conda run -n gecco_mh pytest tests/test_phase5_duckdb_canonical_state.py tests/test_phase6_candidate_evaluator.py tests/test_phase6_parallel_extraction_subtracks.py`
- `conda run -n gecco_mh python -m gecco --help`
- Reference checks: use repo search to confirm no active code/docs mention `rebuild_from_artifacts`, `judge.orchestrated`, `judge.mode`, or `ToolUsingJudge.get_feedback` except historical deletion notes if any remain outside current docs.

Manual review checks:

- Compare `git status --short` before and after; every changed file must be in allowed scope or user-approved.
- Review generated launcher commands for exact script paths and flag ordering compatibility.
- Review docs to ensure current architecture decisions were preserved after deleting `docs/codebase_cleanup_plan/*`.

## Implementation Steps

1. Baseline gate: run `git status --short`. If dirty, stop. If clean, record the allowed-scope file list and continue.
2. Tests first for helpers: add `tests/test_cli_launcher_utils.py`, utility `mapping_get()` tests, and `CandidatePayload` tests. Keep them failing only for missing implementation.
3. Add `mapping_get()` to `gecco/utils.py`; replace local `_mapping_get()` copies in allowed files only where import cycles do not appear. Run the utility test.
4. Add `CandidatePayload` inside `gecco/candidate_generation.py` with a classmethod for one parsed model and a private batch helper for count validation. Keep `CandidateGenerationResult.candidates` as `list[dict]`. Run candidate generator tests.
5. Add `gecco/cli/launcher_utils.py` with `LaunchCommand`, `LaunchPlan`, `LaunchExecutor`, and `SubmissionResult`. Implement dependency builders over prior label results.
6. Refactor `launch_distributed.py` and `launch_cmg_distributed.py` to construct and execute launch plans while preserving printed behaviour and effective command strings. Run CLI helper and CLI contract tests.
7. Remove dead modules and unused functions. Update imports/tests immediately after each deletion. Do not leave stubs.
8. Delete `gecco/diagnostic_store/rebuild.py`, rewrite stale rebuild tests/patches to current DuckDB-source behaviour, and update `run_test_evaluation.py` guidance to point to `--write-store`.
9. Remove `ToolUsingJudge.get_feedback()` and migrate tests to the orchestrated pipeline or analysis/synthesis methods. Do not split `tool_judge.py`.
10. Docs pass: update `README.md`, `docs/judge_implementation_record.md`, and `docs/centralised_judge_implementation.md`; add `docs/tool_judge_split_note.md`; delete existing `docs/codebase_cleanup_plan/*` files and folder if empty.
11. Run targeted tests and smoke checks. Finish with `git status --short` and changed-file review.

## Forbidden Patterns

- Do not create hidden compatibility wrappers for deleted modules, functions, config fields, or judge methods.
- Do not make JSON sidecars canonical, rebuildable runtime state, or required inputs.
- Do not mock away `LaunchExecutor` in tests that are meant to prove dependency/job-id behaviour.
- Do not change SLURM script paths, command argument order, or dry-run/local semantics unless a test is intentionally updated to capture an approved behaviour change.
- Do not pass `CandidatePayload` objects into evaluator or registry code in this pass.
- Do not introduce global launch state, hardcoded absolute paths, or environment-dependent defaults beyond existing launcher inputs.
- Do not opportunistically refactor dashboard, config, bash scripts, prompt content, or persistence schemas.

## Acceptance Checklist

Dead-code deletion:

- Deleted files/functions are absent with no compatibility stubs.
- Active code/docs/tests have no references to deleted APIs.

Launcher abstraction:

- New launcher utility contract tests pass.
- Existing CLI route and launcher command tests pass.
- Manual command review confirms effective sbatch strings and dependency behaviour are preserved.

Candidate payloads:

- `CandidatePayload` tests pass.
- Generator results and registry payloads remain dict-shaped with existing fields.

Runtime state and judge flow:

- Rebuild path is removed; user guidance points to `--write-store`.
- `ToolUsingJudge.get_feedback()` is removed; orchestrated judge tests pass.
- No runtime sidecar output behaviour was intentionally changed.

Docs:

- Current docs preserve canonical DuckDB/registry decisions and describe JSON sidecars as inspection/audit artefacts.
- Existing cleanup-plan docs are deleted; `docs/tool_judge_split_note.md` exists.

Final verification:

- Required `conda run -n gecco_mh ...` test commands were run and results reported.
- Final `git status --short` contains only allowed-scope changes.
- Any unverified risk is explicitly stated in the final response.
