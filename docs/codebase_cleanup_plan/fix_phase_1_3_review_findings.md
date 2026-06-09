# Fix: Phase 1 / Phase 3 review findings

## Purpose

Close the remaining cleanup gaps identified while verifying
`docs/codebase_cleanup_plan/phase_1_docs_reset.md`,
`docs/codebase_cleanup_plan/phase_2_cli_consolidation.md`,
`docs/codebase_cleanup_plan/phase_3_config_validation.md`, and
`docs/codebase_cleanup_plan/phase_3_config_validation_remediation_plan.md`.

This follow-up is narrower than the phase files above: Phase 3’s core schema
and capability work is already in place, but the repository still has stale
entrypoint references and one pytest collection hazard that prevent the wider
suite from being clean.

## Relationship to prior plans

- `docs/codebase_cleanup_plan/phase_1_docs_reset.md`
  - Establishes the docs-reset goal: remove references to deleted entrypoints.
- `docs/codebase_cleanup_plan/phase_2_cli_consolidation.md`
  - Establishes the unified `gecco ...` CLI surface that docs should point to.
- `docs/codebase_cleanup_plan/phase_3_config_validation.md`
  - Establishes validated config loading and explicit capability lists.
- `docs/codebase_cleanup_plan/phase_3_config_validation_remediation_plan.md`
  - Establishes the Phase 3 follow-up behaviour for real configs, lesion configs,
    and dead-code removal.
- `docs/codebase_cleanup_plan/fix_phase_1_2_review_findings.md`
  - Establishes the prior cleanup style for stale references and script removal.

## Current gaps to fix

1. **Stale script-entrypoint references remain in active config/docs comments.**
   - The Phase 1 docs-reset tests still fail on config header comments that
     mention removed script paths such as `scripts/launch_distributed.py` and
     `bash/run_gecco_distributed.sh`.
   - These references should point at the unified `python -m gecco ...`
     command family instead.

2. **`scripts/or_test.py` is unsafe for pytest collection.**
   - It executes external API logic at import/collection time and causes the
     full suite to fail during collection.
   - It should either be deleted (preferred if unused) or moved behind a safe
     non-test entrypoint with no import-time side effects.

3. **Phase 3 config-loading coverage should stay scoped to real runtime configs.**
   - Complete runtime configs should continue to load through `load_config()`.
   - Partial/example fragments such as `config/judge_tool_example.yaml` should
     remain documented as examples, not treated as complete runtime fixtures.

## Non-goals

- Do not redesign the judge pipeline.
- Do not introduce a broad capability-permission framework.
- Do not touch dashboard code.
- Do not add compatibility wrappers for removed entrypoints.
- Do not call external services during unit test collection.

## TDD-first implementation plan

### Chunk A: Remove stale entrypoint references

1. Extend or keep the Phase 1 docs-reset coverage so config YAML header comments
   are treated as active documentation and stale entrypoint strings are caught.
2. Update the remaining config comments and docs to reference the unified CLI
   commands introduced in `phase_2_cli_consolidation.md`.
3. Confirm the fixes cover both markdown docs and YAML comment blocks.

### Chunk B: Eliminate the pytest collection hazard

1. Add a regression test that ensures `pytest tests` does not collect any
   top-level script that performs network access or other side effects at import
   time.
2. Delete `scripts/or_test.py` if it is not part of the supported runtime
   surface.
3. If a local experiment file must remain, move it out of pytest’s collection
   path and ensure it cannot execute network calls on import.

### Chunk C: Keep Phase 3 config-loading coverage precise

1. Preserve the production-config loading tests from
   `phase_3_config_validation.md`.
2. Ensure test fixtures clearly separate full runtime configs from example or
   partial fragments.
3. If any complete runtime config is still missing explicit capabilities,
   update the YAML to match the established full capability list from the Phase
   3 remediation plan.

## Files to inspect

- `tests/test_phase1_docs_reset.py`
- `config/*.yaml`
- `README.md`
- `scripts/or_test.py`
- `tests/test_phase3_config_validation.py`

## Tests to add or update

1. A docs-reset regression test that covers config YAML comment blocks.
2. A broad-scan test ensuring removed script paths do not appear in active docs
   or configs.
3. A regression test ensuring `pytest tests` does not collect side-effectful
   script files.
4. Keep the Phase 3 production-config load tests green.

## Acceptance criteria

- `pytest tests/test_phase1_docs_reset.py` passes.
- `pytest tests` passes.
- `pytest` no longer fails during collection because of `scripts/or_test.py`.
- Active docs and config comments no longer point at removed script entrypoints.
- Phase 3 config-loading tests still pass for complete runtime configs.

## Suggested verification commands

Use the `gecco_mh` conda environment for all Python commands:

- `conda run -n gecco_mh python -m pytest tests/test_phase1_docs_reset.py`
- `conda run -n gecco_mh python -m pytest tests/test_phase3_config_validation.py`
- `conda run -n gecco_mh python -m pytest tests`

## Suggested implementation order

1. Fix stale entrypoint references in docs/config comments.
2. Remove or quarantine `scripts/or_test.py`.
3. Re-run the Phase 1 and Phase 3 test files.
4. Run the full `tests/` suite.
