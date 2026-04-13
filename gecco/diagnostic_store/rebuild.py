"""
Rebuild a DiagnosticStore from the canonical JSON artifacts in
``results/{task}/bics/``.

Use this when:
* The ``.duckdb`` file is missing or corrupted.
* The schema was changed and a full rebuild is needed.

Usage::

    from gecco.diagnostic_store.rebuild import rebuild_from_artifacts
    store = rebuild_from_artifacts("results/two_step_factors")
"""

from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import orjson

from gecco.diagnostic_store.store import DiagnosticStore


_ITER_FILENAME_RE = re.compile(
    r"iter(?P<iteration>\d+)(?:_client(?P<client_id>.+?))?_run(?P<run_idx>\d+)"
    r"(?:_participant\w+)?\.json$"
)


def _detect_split(filename: str) -> str:
    """Detect split from filename: _val -> 'val', _test -> 'test', else 'train'."""
    if "_val" in filename:
        return "val"
    if "_test" in filename:
        return "test"
    return "train"


def _load_json_file(args: tuple[Path, int, str | None, str]) -> dict | None:
    """Load and parse a single JSON file.

    Parameters
    ----------
    args:
        Tuple of (json_file_path, iteration, client_id, tag)

    Returns
    -------
    dict with parsed data or None on error
    """
    json_file, iteration, client_id, tag = args
    try:
        with open(json_file, "rb") as f:
            raw = f.read()

        # Try orjson first (faster), fall back to json if it fails
        # (json handles Python's non-standard Infinity/NaN literals)
        try:
            iteration_results = orjson.loads(raw)
        except orjson.JSONDecodeError:
            import json

            iteration_results = json.loads(raw.decode("utf-8"))

        # Extract PPC data from results
        ppc_results_map = {
            r["function_name"]: r["ppc"]
            for r in iteration_results
            if r.get("function_name") and r.get("ppc")
        }

        return {
            "json_file": json_file,
            "iteration": iteration,
            "client_id": client_id,
            "tag": tag,
            "iteration_results": iteration_results,
            "ppc_results_map": ppc_results_map or None,
        }
    except Exception as e:
        print(f"[rebuild] Error loading {json_file.name}: {e}")
        return None


def rebuild_from_artifacts(
    results_dir: str | Path,
    db_path: str | None = None,
    overwrite: bool = True,
    iterations: list[int] | None = None,
) -> DiagnosticStore:
    """Rebuild the diagnostic store from JSON artifact files.

    Parameters
    ----------
    results_dir:
        Path to the task results directory (contains ``bics/``).
    db_path:
        Where to write the rebuilt ``.duckdb`` file.  Defaults to
        ``{results_dir}/diagnostics.duckdb``.
    overwrite:
        If True, delete an existing database before rebuilding.
    iterations:
        If provided, only process these iteration numbers.
        If None, process all iterations found (original behavior).

    Returns
    -------
    DiagnosticStore
        The newly rebuilt store (left open).
    """
    results_dir = Path(results_dir)
    bics_dir = results_dir / "bics"

    if not bics_dir.exists():
        raise FileNotFoundError(f"bics directory not found: {bics_dir}")

    if db_path is None:
        db_path = str(results_dir / "diagnostics.duckdb")

    if overwrite and Path(db_path).exists():
        Path(db_path).unlink()

    store = DiagnosticStore(db_path)

    json_files = sorted(bics_dir.glob("iter*.json"))
    if not json_files:
        print(f"[rebuild] No iter*.json files found in {bics_dir}")
        return store

    iteration_set = set(iterations) if iterations is not None else None
    processed_count = 0

    print(f"[rebuild] Processing {len(json_files)} artifact files from {bics_dir}")

    # Prepare file list with metadata for parallel parsing
    files_to_parse = []
    for json_file in json_files:
        m = _ITER_FILENAME_RE.search(json_file.name)
        if not m:
            print(f"[rebuild] Skipping unrecognised filename: {json_file.name}")
            continue

        iteration = int(m.group("iteration"))

        # Skip if not in the requested iterations set
        if iteration_set is not None and iteration not in iteration_set:
            continue

        run_idx = int(m.group("run_idx"))
        client_id = m.group("client_id") or None
        tag = f"_client{client_id}" if client_id else ""

        files_to_parse.append((json_file, iteration, run_idx, client_id, tag))

    if not files_to_parse:
        if iteration_set is not None:
            print(
                f"[rebuild] Warning: No files found for iterations: {sorted(iteration_set)}"
            )
        return store

    # Parallel JSON parsing (Opt-E)
    # Parse JSON files in parallel using ProcessPoolExecutor
    parse_args = [
        (json_file, iteration, client_id, tag)
        for json_file, iteration, run_idx, client_id, tag in files_to_parse
    ]

    parsed_results = []
    if len(parse_args) > 1:
        # Use parallel parsing for multiple files
        with ProcessPoolExecutor() as executor:
            parsed_results = list(executor.map(_load_json_file, parse_args))
    else:
        # Single file - parse sequentially
        parsed_results = [_load_json_file(args) for args in parse_args]

    # Process parsed results sequentially (DuckDB is single-writer)
    for i, parsed in enumerate(parsed_results):
        if parsed is None:
            continue

        json_file, iteration, run_idx, client_id, tag = files_to_parse[i]
        split = _detect_split(json_file.name)

        store.write_iteration(
            iteration=parsed["iteration"],
            run_idx=run_idx,
            iteration_results=parsed["iteration_results"],
            ppc_results=parsed["ppc_results_map"],
            tag=parsed["tag"],
            client_id=parsed["client_id"],
        )
        processed_count += 1
        ppc_count = len(parsed["ppc_results_map"] or {})
        print(
            f"[rebuild]   iter={parsed['iteration']} run={run_idx} client='{parsed['client_id']}' split='{split}' "
            f"({len(parsed['iteration_results'])} models, {ppc_count} with PPC)"
        )

    top_models_test_file = results_dir / "bics" / "top_models_test.json"
    if top_models_test_file.exists():
        print(f"[rebuild] Processing top_models_test.json")
        with open(top_models_test_file, "rb") as f:
            try:
                top_models = orjson.loads(f.read())
            except Exception as e:
                print(f"[rebuild] JSON parse error in top_models_test.json: {e}")
            else:
                for entry in top_models:
                    store.write_top_model_test(entry)
                print(f"[rebuild]   Wrote {len(top_models)} test entries")

    if iteration_set is not None and processed_count == 0:
        print(
            f"[rebuild] Warning: No files found for iterations: {sorted(iteration_set)}"
        )
    else:
        print(f"[rebuild] Done. Processed {processed_count} files.")
    return store
