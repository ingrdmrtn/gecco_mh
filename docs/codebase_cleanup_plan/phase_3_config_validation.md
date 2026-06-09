# Phase 3: Config validation

## Purpose

Make runtime loading respect the schema.

## Depends on

- Phase 2.

## Can prepare in parallel with

- Later prep for Phases 4 and 6.

## Likely files/modules to inspect

- `config/schema.py`
- `gecco/run_gecco.py`
- `gecco/load_llms/model_loader.py`

## Tests to write first

- Config contract tests for valid configs/capability sets.
- Failure tests for invalid configs.
- Tests for explicit `judge.capabilities` placement and capability validation.

## Implementation tasks

- Load configs through Pydantic first.
- Make runtime fields explicit.
- Replace lesion-first config with additive `judge.capabilities` lists.
- Validate capability dependencies strictly and fail fast on invalid combinations.
- Delete old lesion config names (`complete`, `noise`, `no_tools`, `no_recommendations`, `no_citations`, `no_diagnostics`) after migration, with no compatibility wrappers.
- Keep built-in profiles as docs/example presets only; the runtime schema should accept explicit capability lists.

## Deletion tasks

- Remove old config names once replacement capability sets exist.
- Remove config shims that only support old naming.

## Acceptance criteria

- Explicit `judge.capabilities` load consistently.
- Invalid capability sets fail fast.
- Empty `judge.capabilities: []` yields an explicit empty feedback trace with no substantive feedback.
