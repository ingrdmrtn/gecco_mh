# No-Slurm E2E Smoke Final Review

## 1. Findings

No findings.

Resolved/accepted items from prior reviews:

- The distributed smoke command now uses the planned no-`--launch-vllm` surface; VLLM launch coverage is an accepted user-approved deviation.
- Distributed no-VLLM smoke coverage now asserts the real CLI subprocess path, Slurm-free `PATH`, `sbatch`, `run_gecco_distributed.sh`, `--array=0-6`, no Rich object repr leak, and no real submission text.
- The CMG dry-run smoke now asserts `run_cmg_generator.sh`, `run_cmg_evaluator.sh`, `run_test_evaluation.sh`, `--array=0-1`, `[Dry run]`, and no real submission text.
- The CMG production launcher now emits a dedicated `bash/run_cmg_evaluator.sh` evaluator array command, and companion contract tests cover the command ordering and wrapper name.
- The implementation receipt now includes raw initial/final `git status --short`, commands run, pass results, approved deviations, and scope notes.

Residual risks:

- I reviewed by code/receipt inspection and did not independently re-run the recorded conda verification commands.
- The distributed no-VLLM path still does not assert a `[Dry run]` marker or dependency flag; this is acceptable for the approved VLLM-skip/no-VLLM scope because the test proves non-submission by running with `sbatch` absent from `PATH` and by asserting no `Submitted batch job` text.

## 2. Open Questions

None.

## 3. Verification Summary

- `tests/test_e2e_cli_smoke.py` invokes the real CLI with `subprocess.run([sys.executable, "-m", "gecco", ...])` from the repository root.
- Both dry-run smoke tests replace `PATH` with `tmp_path`, so they do not depend on a real `sbatch` binary.
- The distributed smoke test does not pass `--launch-vllm` and does not assert VLLM launcher output.
- The distributed smoke test has meaningful no-VLLM coverage through CLI execution, launcher script text, Slurm command construction, client array shape, and non-submission assertions.
- The CMG production change meaningfully satisfies the original CMG contract after the approved expansion: the launcher builds a distinct evaluator array using `bash/run_cmg_evaluator.sh`, and the new wrapper runs `python -m gecco internal distributed-client --client-id "$SLURM_ARRAY_TASK_ID"`.
- `tests/test_cmg_launcher.py` and `tests/test_phase2_cli_contract.py` include targeted assertions for the new CMG evaluator wrapper and command shape.
- The receipt records these verification commands as passed:
  - `conda run -n gecco_mh pytest tests/test_e2e_cli_smoke.py -k help`
  - `conda run -n gecco_mh pytest tests/test_e2e_cli_smoke.py`
  - `conda run -n gecco_mh pytest tests/test_e2e_cli_smoke.py tests/test_phase2_cli_contract.py tests/test_cmg_launcher.py`
  - `conda run -n gecco_mh python -m gecco --help`

## 4. Scope Audit

Accepted deviations/expansions:

- VLLM launch coverage is skipped/out of scope.
- CMG production functionality was expanded to expose/prove `run_cmg_evaluator.sh`.

Changed files recorded by the receipt:

- `gecco/cli/launch_cmg_distributed.py` — within approved CMG production expansion.
- `bash/run_cmg_evaluator.sh` — within approved CMG expansion to expose a dedicated evaluator wrapper.
- `tests/test_e2e_cli_smoke.py` — planned new smoke test.
- `tests/test_cmg_launcher.py` — targeted CMG contract coverage for the approved expansion.
- `tests/test_phase2_cli_contract.py` — targeted launcher command contract coverage.
- Documentation/review/receipt files under `docs/` — verification artifacts.

I found no scope violations beyond the user-approved CMG expansion and accepted VLLM skip. I also found no forbidden patterns in the E2E smoke test: no skip-on-missing-`sbatch`, fake scheduler shim, direct handler calls, hardcoded absolute repo paths, broad sleeps, network calls, GPU/model-server requirement, or real Slurm submission assertion.
