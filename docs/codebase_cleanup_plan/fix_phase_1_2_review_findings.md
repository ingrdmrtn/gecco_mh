# Fix: Phase 1 & Phase 2 review findings

## Purpose

Close the gaps identified during code review of the Phase 1 (Docs reset) and
Phase 2 (CLI consolidation) implementation on branch `cleanup/phase1-2`.

The routing layer and primary doc updates are correct and all 132 tests pass.
The remaining work falls into two categories:

1. **Phase 1 gaps** — stale references that the docs-reset test does not
   cover, so the acceptance criterion *"Docs do not direct users to removed
   entrypoints"* is not fully met.
2. **Phase 2 gaps** — the CLI delegates to old scripts via `sys.argv`
   reconstruction instead of calling importable module functions, and the old
   scripts have not been deleted. The plan explicitly requires *"Delete old
   commands instead of wrapping them"* and *"No thin compatibility wrappers
   remain."*

## Depends on

- Phase 0 (already complete).
- The existing Phase 1 and Phase 2 routing work on `cleanup/phase1-2`.

## Can run in parallel with

- Nothing — this fix plan should be completed before starting Phase 3.

---

## Part A: Phase 1 — remaining stale references

### Tests to write first

- Extend `tests/test_phase1_docs_reset.py`:
  - Add `scripts/two_step_psychiatry_group.py` to `STALE_ENTRYPOINTS` (the
    script does not exist on disk but is still referenced in `README.md`
    lines 365 and 379).
  - Expand `ACTIVE_DOCS` to also cover `config/*.yaml` header comments, so
    that config-file references to old entrypoints are caught by the same
    test.
  - Add a broad-scan test that walks all `.md`, `.yaml`, and `.sh` files
    under the project root (excluding `docs/codebase_cleanup_plan/`,
    `tests/fixtures/`, and `.git/`) and fails if any of the six stale
    entrypoint paths appear.

### Implementation tasks

1. **Fix `README.md` lines 365 and 379.**
   Replace `python scripts/two_step_psychiatry_group.py --config
   two_step_vllm_example.yaml` with the correct current command. If the
   script was renamed, use the new name; if it was removed, replace the
   vLLM quick-start example with `python scripts/two_step_demo.py` or
   another working demo script.

2. **Update config YAML header comments.**
   - `config/two_step_factors_gemini3flash_generic_lesion_no_tools_cmg.yaml`
     lines 2 and 6: replace `python scripts/launch_cmg_distributed.py` with
     `python -m gecco run cmg-distributed`, and replace
     `scripts/launch_distributed.py` with `python -m gecco run distributed`.
   - `config/two_step_factors_gemini3flash_generic_lesion_no_tools_cmg_smoke.yaml`
     lines 2 and 8: same replacements.

3. **Update `.claude/settings.json` line 4.**
   Replace `python scripts/reset_distributed.py
   config/two_step_factors_distributed.yaml --dry-run` with
   `python -m gecco reset config/two_step_factors_distributed.yaml --dry-run`.

4. **Update or annotate `docs/codebase_deepdive_deck.html` lines 350, 359.**
   Either update the architecture diagram references to use the CLI command
   names, or add a note at the top of the file marking it as a historical
   snapshot that may not reflect current architecture.

### Deletion tasks

- None for Part A — these are all reference updates.

### Acceptance criteria

- The extended `test_phase1_docs_reset.py` passes with the broader scan.
- No file outside `docs/codebase_cleanup_plan/` and `tests/fixtures/`
  references any of the six stale entrypoint paths or
  `scripts/two_step_psychiatry_group.py`.

---

## Part B: Phase 2 — extract modules and delete old scripts

The current CLI handlers reconstruct `sys.argv` and call each script's
`main()` via `gecco/cli/_legacy.py:invoke_legacy_main`. This is a reasonable
intermediate step but does not satisfy the plan's requirements:

> *"Move behavior into importable modules behind the CLI."*
> *"Delete old commands instead of wrapping them."*
> *"No thin compatibility wrappers remain."*

The fix is to split each script's `main()` into **(1) argument parsing** and
**(2) a pure entrypoint function** that takes typed parameters, then have the
CLI handler call the entrypoint function directly. Once every script has been
converted, delete the old `scripts/*.py` files.

### Tests to write first

- Add contract tests for the two untested CLI routes:
  - `run local-client` → `gecco.cli.run_local_client:main`
  - `internal test-evaluation` → `gecco.cli.run_test_evaluation:main`
- Add an argument-fidelity test for at least one route: verify that
  `gecco run distributed --config x.yaml --vllm-tp 4 --launch-vllm`
  produces the correct typed arguments when the handler calls the
  entrypoint function (not reconstructed `sys.argv`).
- After extraction, add a smoke test that imports each entrypoint function
  directly (e.g. `from gecco.cli.monitor_distributed import run_monitor`)
  and verifies it is callable with typed keyword arguments.

### Implementation tasks

Work one script at a time. For each script:

1. **Extract the entrypoint function.**
   In the corresponding `gecco/cli/<name>.py` module, add a `run_<name>(...)`
   function that accepts typed keyword arguments and contains (or imports and
   calls) the real business logic. The existing `main(args)` handler should
   become a thin adapter: extract fields from `argparse.Namespace` and call
   `run_<name>(...)`.

   Suggested extraction order (simplest to most complex):

   | Order | Script | Reason |
   |-------|--------|--------|
   | 1 | `scripts/reset_distributed.py` | Smallest (~193 lines), fewest args, no SLURM |
   | 2 | `scripts/monitor_distributed.py` | Self-contained, read-only, no config loading |
   | 3 | `scripts/run_test_evaluation.py` | Small, called internally |
   | 4 | `scripts/run_gecco_distributed.py` | Core worker, but well-isolated |
   | 5 | `scripts/run_judge_orchestrator.py` | Larger, barrier logic |
   | 6 | `scripts/launch_distributed.py` | Largest, SLURM submission logic |
   | 7 | `scripts/launch_cmg_distributed.py` | Similar to above, CMG variant |

2. **Update the CLI handler to call the entrypoint function directly.**
   Remove the `invoke_legacy_main` call and the `sys.argv` reconstruction.
   The handler becomes:

   ```python
   def main(args: argparse.Namespace) -> int | None:
       return run_<name>(
           config=args.config,
           task=args.task,
           ...
       )
   ```

3. **Verify tests still pass after each script conversion.**
   Run `pytest tests/ -v` after each extraction. The existing contract tests
   should continue to pass because they test routing, not internals.

4. **Delete `gecco/cli/_legacy.py`** once no handler imports it.

5. **Delete the old `scripts/*.py` entrypoints.**
   After all seven scripts have been converted and the full test suite
   passes:
   - Delete `scripts/launch_distributed.py`
   - Delete `scripts/run_gecco_distributed.py`
   - Delete `scripts/run_judge_orchestrator.py`
   - Delete `scripts/reset_distributed.py`
   - Delete `scripts/monitor_distributed.py`
   - Delete `scripts/launch_cmg_distributed.py`
   - Delete `scripts/run_test_evaluation.py` (if it was also routed through
     the CLI)

6. **Update bash scripts to use `python -m gecco` exclusively.**
   The bash scripts (`bash/run_gecco_distributed.sh`,
   `bash/run_judge_orchestrator.sh`, `bash/run_cmg_generator.sh`,
   `bash/run_test_evaluation.sh`) already use `python -m gecco ...`
   commands. Verify they still work after the old scripts are deleted.

7. **Update `scripts/run_local_client.py`.**
   This script already calls `gecco.cli.main(...)` for its in-process path
   (line 188). Verify it still works after the old scripts are deleted.
   Consider whether this script should also become a CLI route or be deleted.

### Deletion tasks

- Delete `gecco/cli/_legacy.py` after step 4 above.
- Delete the seven `scripts/*.py` files listed in step 5 above.
- Remove the `invoke_legacy_main` import from every `gecco/cli/*.py` handler.

### Acceptance criteria

- Every CLI handler calls a typed entrypoint function directly — no
  `sys.argv` reconstruction, no `invoke_legacy_main`.
- `gecco/cli/_legacy.py` does not exist.
- The six old `scripts/*.py` entrypoints listed in Phase 2 do not exist.
- `test_legacy_script_names_are_not_valid_cli_commands` still passes (it
  should, since the old scripts are gone).
- All 132+ existing tests still pass.
- The new argument-fidelity and untested-route contract tests pass.

---

## Commit strategy

Follow the plan README rules: *"Use staged commits to keep the work grouped
into small, reviewable units."*

Suggested commit sequence:

1. `fix(phase1): broaden stale-reference test and fix remaining docs`
   — Part A test extensions + all reference fixes.
2. `fix(phase2): add missing CLI contract tests`
   — New tests for `run local-client`, `internal test-evaluation`, and
   argument fidelity.
3. `refactor(phase2): extract reset_distributed into importable module`
   — First script conversion.
4. `refactor(phase2): extract monitor_distributed into importable module`
5. `refactor(phase2): extract run_test_evaluation into importable module`
6. `refactor(phase2): extract run_gecco_distributed into importable module`
7. `refactor(phase2): extract run_judge_orchestrator into importable module`
8. `refactor(phase2): extract launch_distributed into importable module`
9. `refactor(phase2): extract launch_cmg_distributed into importable module`
10. `chore(phase2): delete old scripts and _legacy.py`
    — Final deletion commit.

Each commit should pass the full test suite independently.

---

## Risk notes

- **SLURM integration**: The launch scripts (`launch_distributed.py`,
  `launch_cmg_distributed.py`) contain SLURM job-submission logic (building
  `sbatch` commands, writing shell wrappers). This logic must be preserved
  exactly during extraction. The `--dry-run` flag provides a safe way to
  verify output parity before and after refactoring.
- **`sys.path` manipulation**: Several scripts do
  `sys.path.insert(0, ...)` at module level. The extracted modules should
  not need this because they will be imported through the `gecco` package.
  Remove these during extraction.
- **`configure_temp_dirs` call at import time**: `run_gecco_distributed.py`
  calls `configure_temp_dirs(project_root, ...)` at module level (line 14).
  This side effect must be moved into the entrypoint function or into
  package-level initialization, not left at import time.
