# Phase 2: CLI consolidation

## Purpose

Collapse user entrypoints into one command family.

## Depends on

- Phase 0.

## Can run in parallel with

- Phase 1.

## Likely files/modules to inspect

- `gecco/run_gecco.py`
- `scripts/run_gecco_distributed.py`
- `scripts/run_judge_orchestrator.py`
- `scripts/launch_distributed.py`
- `scripts/reset_distributed.py`
- `scripts/monitor_distributed.py`

## Tests to write first

- CLI contract tests for command routing.
- Negative tests for removed old names.

## Implementation tasks

- Route through `gecco` subcommands.
- Keep worker logic out of CLI handlers.
- Move behavior into importable modules behind the CLI.

## Deletion tasks

- Delete old commands instead of wrapping them.
- Remove thin compatibility scripts once routing is covered.

## Acceptance criteria

- New CLI routes reach the right runtime paths.
- No thin compatibility wrappers remain.
