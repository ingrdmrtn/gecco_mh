# No-Slurm E2E Smoke Implementation Receipt

## Scope notes

- User-approved deviation: skip VLLM launch coverage in the distributed smoke test.
- Scope expansion approved for CMG so the production launcher could emit `run_cmg_evaluator.sh`.

## Verification commands run

- `conda run -n gecco_mh pytest tests/test_e2e_cli_smoke.py -k help` — passed.
- `conda run -n gecco_mh pytest tests/test_e2e_cli_smoke.py` — passed.
- `conda run -n gecco_mh pytest tests/test_e2e_cli_smoke.py tests/test_phase2_cli_contract.py tests/test_cmg_launcher.py` — passed.
- `conda run -n gecco_mh python -m gecco --help` — passed.

## Findings handled

- Removed `--launch-vllm` from the distributed smoke test.
- Added a distributed no-VLLM array assertion (`--array=0-6`) to keep the command-shape coverage meaningful.
- Updated CMG production wiring to use `bash/run_cmg_evaluator.sh` and added smoke/contract assertions for it.

## Justified gaps / residual risks

- The distributed plan’s `--dependency=afterok` and `[Dry run]` assertions were not retained because VLLM launch coverage was user-skipped.
- No additional gaps are currently known.

## Git status

### Raw initial `git status --short`

```text
?? docs/no-slurm-e2e-smoke-followup-review.md
?? docs/no-slurm-e2e-smoke-implementation-receipt.md
?? docs/no-slurm-e2e-smoke-plan.md
?? docs/no-slurm-e2e-smoke-review.md
?? tests/test_e2e_cli_smoke.py
```

### Raw final `git status --short`

```text
 M gecco/cli/launch_cmg_distributed.py
 M tests/test_cmg_launcher.py
 M tests/test_phase2_cli_contract.py
?? bash/run_cmg_evaluator.sh
?? docs/no-slurm-e2e-smoke-followup-review.md
?? docs/no-slurm-e2e-smoke-implementation-receipt.md
?? docs/no-slurm-e2e-smoke-plan.md
?? docs/no-slurm-e2e-smoke-review.md
?? tests/test_e2e_cli_smoke.py
```

### Pre-existing dirty files carried into this pass

- `docs/no-slurm-e2e-smoke-followup-review.md`
- `docs/no-slurm-e2e-smoke-implementation-receipt.md`
- `docs/no-slurm-e2e-smoke-plan.md`
- `docs/no-slurm-e2e-smoke-review.md`
- `tests/test_e2e_cli_smoke.py`

### Files changed in this pass

- `bash/run_cmg_evaluator.sh`
- `gecco/cli/launch_cmg_distributed.py`
- `tests/test_cmg_launcher.py`
- `tests/test_phase2_cli_contract.py`
