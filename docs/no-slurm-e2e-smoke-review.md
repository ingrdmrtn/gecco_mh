# No-Slurm E2E Smoke Review

## 1. Findings

### High: Distributed smoke test does not exercise the planned command surface

- **File:** `tests/test_e2e_cli_smoke.py:46-54`
- **Plan rule violated:** Production surface in `docs/no-slurm-e2e-smoke-plan.md:28-31`.
- The plan specifies the distributed E2E surface as:

  ```bash
  python -m gecco run distributed --config two_step_factors_distributed.yaml --dry-run
  ```

  The implemented test adds `--launch-vllm`:

  ```python
  "--launch-vllm",
  "--dry-run",
  ```

  This means the required no-Slurm smoke coverage for the exact planned distributed dry-run command is missing.

### Medium: Distributed dry-run output contract is only partially asserted

- **File:** `tests/test_e2e_cli_smoke.py:56-61`
- **Plan rule violated:** Contract Matrix dry-run output row and test details in `docs/no-slurm-e2e-smoke-plan.md:51,66`.
- The plan requires the distributed test to assert `sbatch`, `run_gecco_distributed.sh`, `--dependency=afterok`, and `[Dry run]`.
- The implementation asserts `sbatch`, `launch_vllm_server.sh`, `run_gecco_distributed.sh`, and `--parsable`, but it does **not** assert:
  - `--dependency=afterok`
  - `[Dry run]`

  As a result, the test can pass without proving the required dependency chain or dry-run marker remains visible.

### Medium: CMG dry-run output contract is only partially asserted

- **File:** `tests/test_e2e_cli_smoke.py:78-84`
- **Plan rule violated:** Contract Matrix dry-run output row and test details in `docs/no-slurm-e2e-smoke-plan.md:51,67`.
- The plan requires the CMG test to assert output includes `run_cmg_generator.sh`, `run_cmg_evaluator.sh`, `run_test_evaluation.sh`, and `[Dry run]`.
- The implementation does not assert `run_cmg_evaluator.sh`; it asserts `run_gecco_distributed.sh` and `run_judge_orchestrator.sh` instead.
- If the implementation intentionally uses `run_gecco_distributed.sh` as the evaluator wrapper, the plan should be updated; under the current plan, this contract row is not satisfied.

### Medium: Required verification and phase-gate command evidence is not present

- **Plan rules affected:** Implementation gates in `docs/no-slurm-e2e-smoke-plan.md:82-103`; required commands in `docs/no-slurm-e2e-smoke-plan.md:70-73`; acceptance checklist in `docs/no-slurm-e2e-smoke-plan.md:128-132`.
- I found the new test file, but did not find a persisted implementation receipt or command transcript proving that the required phase-gate and final verification commands were run:
  - `conda run -n gecco_mh pytest tests/test_e2e_cli_smoke.py -k help`
  - `conda run -n gecco_mh pytest tests/test_e2e_cli_smoke.py`
  - `conda run -n gecco_mh pytest tests/test_e2e_cli_smoke.py tests/test_phase2_cli_contract.py tests/test_cmg_launcher.py`
  - `conda run -n gecco_mh python -m gecco --help`
  - initial and final `git status --short`

  Without that receipt, the plan's required proof and phase-gate compliance are unverifiable from the worktree alone.

## 2. Open Questions

- Was the distributed command intentionally changed from the planned no-`--launch-vllm` surface to include `--launch-vllm`, and should the plan be amended accordingly?
- Is `run_gecco_distributed.sh` the intended CMG evaluator wrapper, replacing the plan's required `run_cmg_evaluator.sh` assertion?
- Is there an implementation receipt outside the repository that records the required command results and pre-/post-worktree status?

## 3. Verification Summary

### Commands/checks observed or still required

- Observed by code inspection:
  - `tests/test_e2e_cli_smoke.py` exists.
  - The new tests invoke `subprocess.run` with `[sys.executable, "-m", "gecco", ...]`.
  - Dry-run tests set `env["PATH"] = str(tmp_path)`.
  - No direct CLI handler calls, fake `sbatch` binary, skip-on-missing-`sbatch`, sleeps, network calls, or real submission assertions were found in the new E2E smoke file.
- Still required / not evidenced in the worktree:
  - All required `conda run -n gecco_mh ...` verification commands.
  - Initial and final `git status --short` evidence.
  - Phase-gate command evidence.

### Contract rows verified

- **CLI entrypoint is exercised for real:** Partially verified. The helper uses `subprocess.run` and `sys.executable -m gecco`; the new smoke file does not call parser handlers directly.
- **No Slurm is required:** Partially verified by code inspection. Dry-run tests remove normal `PATH` by setting it to `tmp_path`; actual passing execution was not evidenced.

### Contract rows not verified

- **Dry-run launcher output remains meaningful:** Not fully verified. Required assertions for distributed `--dependency=afterok` and `[Dry run]` are missing, and required CMG `run_cmg_evaluator.sh` assertion is missing.

## 4. Scope Audit

### Files changed / inspected

- `tests/test_e2e_cli_smoke.py` — allowed by Scope Firewall as the new test file.
- `tests/test_phase2_cli_contract.py` — allowed only if reusing an existing helper without broad edits. I inspected it for context; I cannot confirm whether it was changed without git status/diff evidence.
- `tests/test_cmg_launcher.py` — not listed as an allowed implementation/test change for this plan. I inspected it for context; I cannot confirm whether it was changed without git status/diff evidence.
- `docs/no-slurm-e2e-smoke-review.md` — this review artifact, created for the requested review report.

### Forbidden or suspicious changes

- I did not identify forbidden patterns in `tests/test_e2e_cli_smoke.py`.
- I could not verify the full tracked/untracked worktree scope audit because no `git status --short` output or implementation receipt was available in the inspected files.
