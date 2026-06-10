# Centralised Judge Implementation

Centralised judge runs are built from the runtime DuckDB state and the shared registry.

- Evidence is loaded from `diagnostics*.duckdb` sources.
- The orchestrator reuses the same analysis/synthesis pipeline as local runs.
- JSON artefacts remain auxiliary outputs only.
