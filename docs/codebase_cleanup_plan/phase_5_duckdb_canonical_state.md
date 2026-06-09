# Phase 5: DuckDB canonical state

## Purpose

Move runtime state and coordination to DuckDB.

## Depends on

- Phase 3.

## Can run in parallel with

- Phase 4 after schema decisions.

## Likely files/modules to inspect

- `gecco/diagnostic_store/store.py`
- `gecco/diagnostic_store/schema.py`
- `gecco/diagnostic_store/rebuild.py`
- `gecco/coordination.py`
- `tests/test_cmg_runtime.py`
- `tests/test_diagnostic_store_tools.py`
- `tests/test_rebuild_fallback.py`

## Tests to write first

- DuckDB state contract tests.
- Writer/reader round-trip tests.
- Concurrent client tests.
- Restart/reload tests.
- View consistency tests for status/coordination views.

## Implementation tasks

- Make DuckDB the canonical runtime state store.
- Remove runtime JSON source-of-truth behavior.
- Add status and coordination views backed by DuckDB.
- Keep future export/report work separate from runtime state.
- Validate concurrency with a clear write strategy.

## Deletion tasks

- Remove code paths that treat JSON as state.
- Remove assumptions that runtime JSON files are canonical.

## Concurrency/locking validation

- Treat locking as a validation item, not a known blocker.
- Verify either a single-writer rule or explicit transactions.
- Add tests for concurrent clients and restart/reload behavior.

## Acceptance criteria

- DuckDB is the source of truth.
- Runtime JSON no longer owns state.
