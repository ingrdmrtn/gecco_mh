# Phase 0: Freeze current behavior

## Purpose

Capture the current public behavior before changing architecture.

## Depends on

- None.

## Can run in parallel with

- Nothing else yet.

## Likely files/modules to inspect

- `gecco/run_gecco.py`
- `gecco/coordination.py`
- `config/schema.py`
- `gecco/construct_feedback/*`
- `gecco/diagnostic_store/*`
- `scripts/*`
- `tests/*`

## Tests to write first

- CLI invocation characterization tests.
- Config load/validation characterization tests.
- Judge-path characterization tests.
- Artifact naming/output characterization tests.

## Implementation tasks

- Inventory current CLI scripts, configs, judge modes, and artifact paths.
- Record the current script names and output shapes that matter.
- Create tiny fixtures that capture the current behavior.

## Deletion tasks

- None yet; this phase is only for freezing behavior.

## Acceptance criteria

- Small fixtures reproduce the current behavior, or the differences are documented.
