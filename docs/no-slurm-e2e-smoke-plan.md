# No-Slurm E2E Smoke Plan

## 1. Summary

Add a small end-to-end smoke test layer that runs real `python -m gecco` CLI entrypoints without requiring Slurm. The expected change is a new pytest file that executes help and dry-run launcher commands in subprocesses, proving the CLI can load, parse configs, and produce launcher commands on machines without `sbatch`.

The main risk this plan prevents is accidentally adding E2E tests that pass only on Slurm machines or mock away the CLI entrypoint being tested.

## 2. Scope

Allowed implementation files:
- None expected.
- If inspection shows a tiny production change is required to make dry-run execution avoid Slurm, stop and ask before editing production code.

Allowed test files:
- `tests/test_e2e_cli_smoke.py` as a new file.
- `tests/test_phase2_cli_contract.py` only if an existing helper can be reused without broad edits.

Allowed verification artifacts:
- No persisted artifact is required. The final response receipt is sufficient.

Forbidden files/directories:
- `gecco/**` production code unless the user approves a scope change.
- `config/**`; use existing config files only.
- `bash/**` Slurm wrapper scripts.
- Dashboard, prompt, persistence, migration, generated-output, and artifact directories.

Production surfaces under test:
- `python -m gecco --help`
- `python -m gecco run distributed --config two_step_factors_distributed.yaml --dry-run`
- `python -m gecco run cmg-distributed --config two_step_factors_cmg.yaml --dry-run`

Explicit non-goals:
- Do not add real Slurm E2E tests in this pass.
- Do not require `sbatch`, a scheduler, queued jobs, GPUs, model servers, or network access.
- Do not change production configs to make tests pass.
- Do not add a CI workflow unless separately requested.

Worktree rules:
- Run `git status --short` before implementation.
- Do not revert or modify unrelated pre-existing changes.
- Final scope audit must include tracked and untracked files.
- Any changed file outside the allowed scope means the implementation is incomplete unless the user approves the scope change.

## 3. Contracts

| Contract | Code path / symbol | Positive proof | Negative proof |
| --- | --- | --- | --- |
| CLI entrypoint is exercised for real | `python -m gecco` subprocesses | Tests run subprocess commands with `sys.executable -m gecco` and assert exit code `0` | Do not call parser handlers directly in the E2E smoke file |
| No Slurm is required | distributed and CMG dry-run launcher entrypoints | Tests run with `PATH` set to a temporary directory that does not contain `sbatch` and still pass | Tests fail if dry-run tries to execute `sbatch` |
| Dry-run launcher output remains meaningful | distributed and CMG launcher stdout | Tests assert output includes expected `sbatch` command text, job labels, arrays/dependencies, dry-run marker, and no Python object repr leaks | Tests assert no real submission success text is required, such as `Submitted batch job` |

## 4. Tests And Verification

Tests to add:
- `tests/test_e2e_cli_smoke.py`
- `test_cli_help_smoke_runs_real_entrypoint`
- `test_distributed_dry_run_smoke_does_not_require_slurm`
- `test_cmg_distributed_dry_run_smoke_does_not_require_slurm`

Test details:
- Use `subprocess.run([...], cwd=repo_root, env=env, text=True, capture_output=True, timeout=...)`.
- Build commands with `sys.executable`, not hardcoded `python`.
- Set `env["PATH"]` to a temporary directory for dry-run tests to prove `sbatch` is not required.
- Assert help output contains `GeCCo command line interface` and top-level commands.
- Assert distributed dry-run output contains `sbatch`, `run_gecco_distributed.sh`, `--dependency=afterok`, and `[Dry run]`.
- Assert CMG dry-run output contains `run_cmg_generator.sh`, `run_cmg_evaluator.sh`, `run_test_evaluation.sh`, and `[Dry run]`.
- Assert dry-run outputs do not contain `<rich.panel.Panel object` or `Submitted batch job`.

Commands to run:
- `conda run -n gecco_mh pytest tests/test_e2e_cli_smoke.py`
- `conda run -n gecco_mh pytest tests/test_e2e_cli_smoke.py tests/test_phase2_cli_contract.py tests/test_cmg_launcher.py`
- `conda run -n gecco_mh python -m gecco --help`

Manual review checks:
- Confirm the new tests use real subprocess CLI invocation rather than direct handler calls.
- Confirm no production files, configs, or Slurm scripts were changed.
- Confirm no test requires `sbatch` on `PATH`.

## 5. Implementation Steps

1. Pre-flight
- Run `git status --short` and note any pre-existing dirty files.
- Inspect `tests/test_cmg_launcher.py` and `tests/test_phase2_cli_contract.py` only for expected dry-run output terms.

2. Add subprocess helper
- Create `tests/test_e2e_cli_smoke.py`.
- Add a small helper that runs `sys.executable -m gecco ...` from the repository root with captured stdout/stderr and a timeout.
- Add a failure assertion that includes stdout/stderr when return code is non-zero.

3. Add real CLI help smoke test
- Add `test_cli_help_smoke_runs_real_entrypoint`.
- Run `conda run -n gecco_mh pytest tests/test_e2e_cli_smoke.py -k help` before continuing.

4. Add no-Slurm dry-run launcher tests
- Add distributed and CMG dry-run tests with `PATH` set to `tmp_path`.
- Assert dry-run command output proves launcher command construction without real submission.
- Run `conda run -n gecco_mh pytest tests/test_e2e_cli_smoke.py` before continuing.

5. Final verification
- Run all commands listed in Section 4.
- Run final `git status --short`.
- Review changed files against Section 2.

## 6. Forbidden Patterns

- Do not skip tests when `sbatch` is missing; the point is to pass without it.
- Do not add fake `sbatch` binaries, shell scripts, or scheduler shims.
- Do not call CLI handlers directly from the new E2E smoke tests.
- Do not hardcode absolute repo paths or machine-specific conda paths.
- Do not mutate production config files or write persistent result artifacts.
- Do not add broad sleeps, network calls, GPU/model-server requirements, or real Slurm submissions.

## 7. Acceptance Checklist

CLI entrypoint contract:
- `tests/test_e2e_cli_smoke.py` uses `subprocess.run` with `sys.executable -m gecco`.
- Help smoke test passes and asserts expected CLI text.

No-Slurm contract:
- Distributed and CMG dry-run tests set `PATH` to a temporary directory with no `sbatch`.
- Dry-run tests pass without Slurm installed or available.

Dry-run output contract:
- Tests assert expected command text and dry-run markers.
- Tests assert no `Submitted batch job` dependency on real submission output.

Final checks:
- `conda run -n gecco_mh pytest tests/test_e2e_cli_smoke.py` passes.
- `conda run -n gecco_mh pytest tests/test_e2e_cli_smoke.py tests/test_phase2_cli_contract.py tests/test_cmg_launcher.py` passes.
- `conda run -n gecco_mh python -m gecco --help` passes.
- Final `git status --short` shows only allowed files changed.
