# Judge Implementation Record

The current judge path is orchestrated and capability-driven.

- Analysis and persona synthesis are separate steps.
- DuckDB-backed runtime state is canonical.
- JSON files in `results/...` are kept for inspection and audit.
- Diagnostic persistence for post-processing uses the explicit `--write-store` flag.
