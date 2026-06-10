# No-Slurm E2E Smoke Follow-up Review

## 1. Findings

### Medium: CMG evaluator wrapper assertion remains a non-user-approved plan deviation

- **File:** `tests/test_e2e_cli_smoke.py:75-81`
- **Plan contract:** `docs/no-slurm-e2e-smoke-plan.md:51,67` requires the CMG dry-run smoke output to include `run_cmg_generator.sh`, `run_cmg_evaluator.sh`, `run_test_evaluation.sh`, and `[Dry run]`.
- The implementation still does not assert `run_cmg_evaluator.sh`; it asserts `run_gecco_distributed.sh` and `run_judge_orchestrator.sh` instead. The receipt documents this as a justified gap (`docs/no-slurm-e2e-smoke-implementation-receipt.md:20,25`), but the user's clarification only approved skipping VLLM launch coverage/dependent assertions. Under the authoritative plan as modified only by the VLLM skip, this CMG assertion remains unresolved unless the user also approves updating the plan/contract to reflect the production wrapper name.

### Low: Distributed no-VLLM dry-run coverage is meaningful but thinner than the remaining plan contract

- **File:** `tests/test_e2e_cli_smoke.py:55-58`
- **Plan contract:** `docs/no-slurm-e2e-smoke-plan.md:51,66` calls for expected launcher command text, arrays/dependencies, a dry-run marker, and no Python object repr leaks.
- The distributed command now matches the planned no-`--launch-vllm` surface (`docs/no-slurm-e2e-smoke-plan.md:30`), and the receipt clearly records the VLLM-approved skip/dependent assertion removal (`docs/no-slurm-e2e-smoke-implementation-receipt.md:5,18-19,24`). However, the remaining distributed assertions only check `sbatch`, `run_gecco_distributed.sh`, no rich object repr, and no real submission text. They do not assert a non-VLLM command-shape detail such as the client array flag (for example `--array=`). Given dependencies and the distributed `[Dry run]` marker are documented as absent on the no-VLLM path, an array assertion would make the remaining no-Slurm CLI coverage better satisfy the plan's "arrays/dependencies" intent.

### Low: Git-status evidence is summarized, not raw/verifiable

- **File:** `docs/no-slurm-e2e-smoke-implementation-receipt.md:27-37`
- **Plan rule:** `docs/no-slurm-e2e-smoke-plan.md:39-43,100-103,128-132` requires initial/final `git status --short` and a final scope audit including tracked and untracked files.
- The receipt now records pre-existing dirty files and current additions, which is a substantial improvement over the initial review gap. It does not include the raw initial/final `git status --short` output, so the full tracked/untracked worktree and absence of production/config/bash changes are not independently auditable from the receipt alone. I did not identify production launcher edits during this review, but the scope conclusion still relies partly on the implementer's summarized status.

## 2. Open Questions

- Is the CMG plan contract intended to be amended from `run_cmg_evaluator.sh` to the currently emitted evaluator wrapper (`run_gecco_distributed.sh`)? This is the only question blocking full confidence against the authoritative plan.

## 3. Verification Summary

### Commands/checks observed or still required

- Observed by inspection:
  - `tests/test_e2e_cli_smoke.py` invokes the real CLI via `subprocess.run([sys.executable, "-m", "gecco", ...])` from the repo root.
  - Both dry-run smoke tests set `env["PATH"] = str(tmp_path)`, so `sbatch` is not available through normal `PATH` lookup.
  - The distributed smoke command no longer includes `--launch-vllm` and now matches the planned command surface with the user-approved VLLM skip.
  - No direct parser-handler calls, skip-on-missing-`sbatch`, fake `sbatch` binaries, broad sleeps, hardcoded absolute repo paths, or real submission assertions were found in the E2E smoke file.
- Recorded in the receipt as passed:
  - `conda run -n gecco_mh pytest tests/test_e2e_cli_smoke.py -k help`
  - `conda run -n gecco_mh pytest tests/test_e2e_cli_smoke.py`
  - `conda run -n gecco_mh pytest tests/test_e2e_cli_smoke.py tests/test_phase2_cli_contract.py tests/test_cmg_launcher.py`
  - `conda run -n gecco_mh python -m gecco --help`
- Still not independently evidenced:
  - Raw initial and final `git status --short` output.

### Contract rows

- **CLI entrypoint is exercised for real:** Verified by code inspection and receipt command results.
- **No Slurm is required:** Verified by code inspection of `PATH` isolation and receipt command results.
- **Dry-run launcher output remains meaningful:** Partially verified. CMG has strong command assertions and `[Dry run]`, but misses the planned `run_cmg_evaluator.sh` assertion. Distributed has the user-approved no-VLLM command surface but only minimal command-shape assertions.
- **VLLM launch coverage/dependent assertions:** Deviated by user approval and clearly documented in the receipt.

## 4. Scope Audit

- `tests/test_e2e_cli_smoke.py` — allowed new test file; inspected.
- `docs/no-slurm-e2e-smoke-plan.md` — authoritative plan; documented by receipt as pre-existing untracked.
- `docs/no-slurm-e2e-smoke-review.md` — prior review artifact; documented by receipt as pre-existing untracked.
- `docs/no-slurm-e2e-smoke-implementation-receipt.md` — verification/scope receipt requested for this review loop; acceptable artifact.
- `docs/no-slurm-e2e-smoke-followup-review.md` — this review artifact.
- Production launcher files under `gecco/cli/` were inspected only as needed for contract interpretation; I did not identify production edits, but this remains subject to the raw git-status evidence gap above.
