# Phase 1: Docs reset

## Purpose

Make the active surface obvious to developers.

## Depends on

- Phase 0.

## Can run in parallel with

- Phase 2.

## Likely files/modules to inspect

- `README.md`
- `docs/centralised_judge_implementation.md`
- `docs/judge_implementation_record.md`
- `docs/codebase_cleanup_plan.md`

## Tests to write first

- Docs/navigation checks that search for stale command names or obsolete paths.

## Implementation tasks

- Update README and docs to point only at active commands and current file paths.
- Remove stale references to old scripts and dead examples.
- Keep this cleanup plan as the source of truth for the rewrite.

## Deletion tasks

- Remove docs links to removed entrypoints where safe.

## Acceptance criteria

- Docs do not direct users to removed entrypoints.
