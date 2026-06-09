# Cleanup Acceptance Gaps Review Fix Findings

## 1. Findings

### High — Required receipt artifact is missing and the implemented receipt is outside the allowed artifact path

- **Plan rule violated:** Scope Firewall / Contract row “Baseline and final verification evidence” (`cleanup-acceptance-gaps-review-fix-plan.md:21-24`, `:55`, `:139-142`).
- **Observed:** The required receipt path `docs/codebase_cleanup_plan/cleanup-acceptance-gaps-review-fix-receipt.md` does not exist. The implementation instead added/updated `docs/codebase_cleanup_plan/cleanup-acceptance-gaps-review-receipt.md` (`cleanup-acceptance-gaps-review-receipt.md:1`).
- **Why this matters:** The plan allowed exactly one verification artifact name. The actual receipt is a broad docs change outside that allowed filename, and reviewers looking for the source-of-truth receipt will not find it.

### High — Shared judge prompt still asks no-recommendations modes for next-iteration actionable feedback

- **Plan rule violated:** Capability prompt contract and forbidden pattern against broad shared prompts that request disabled recommendations (`cleanup-acceptance-gaps-review-fix-plan.md:52`, `:97-105`, `:114-115`, `:124-128`).
- **Observed:** `_JUDGE_SYSTEM_PROMPT` still unconditionally tells all capability modes to “synthesise actionable feedback for the next iteration” and to produce a `synthesized_feedback` paragraph “to improve the next iteration” (`gecco/construct_feedback/tool_judge.py:258-300`).
- **Why this matters:** For configurations without `recommendations`, the LLM is still prompted before generation to provide next-iteration improvement guidance. Post-processing then removes explicit recommendation sections, but the plan requires prompt-level prevention, not only output stripping.

### Medium — Receipt claims required final evidence but does not record it durably

- **Plan rule violated:** Receipt contract and verification requirements (`cleanup-acceptance-gaps-review-fix-plan.md:55`, `:71-78`, `:107-110`, `:139-143`).
- **Observed:** The receipt claims final `git status --short` and `git diff --name-only` coverage (`cleanup-acceptance-gaps-review-receipt.md:138`, `:147-153`) but only records summaries, not the final command outputs. It also does not include the exact output of `git diff --name-only -- gecco-mh-dashboard`.
- **Why this matters:** The receipt is not durable enough to verify final scope, untracked files, or dashboard diff audit without transient terminal state.

### Medium — Focused tool-judge verification command is not separately recorded or justified as skipped

- **Plan rule violated:** Required verification commands (`cleanup-acceptance-gaps-review-fix-plan.md:65-69`, `:107-109`).
- **Observed:** The receipt records only `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py tests/test_phase4_orchestrated_judge.py -q` with `72 passed` (`cleanup-acceptance-gaps-review-receipt.md:128-132`). It does not record the required focused tool-judge command or justify why it was omitted after placing prompt tests in `tests/test_phase3_config_validation.py`.
- **Why this matters:** The prompt contract may be covered by the combined command, but the receipt does not explicitly prove or justify the required focused check.

### Medium — Lesion-free production config discovery does not start from all `config/*.yaml`

- **Plan rule violated:** Production config contract and forbidden pattern against narrow representative lists (`cleanup-acceptance-gaps-review-fix-plan.md:54`, `:61-62`, `:92-95`, `:118`).
- **Observed:** `PRODUCTION_CONFIGS` is built from `CONFIG_DIR.glob("two_step_factors_*.yaml")` (`tests/test_phase3_config_validation.py:31`), not from `config/*.yaml`. The parametrized test then loads only that subset (`tests/test_phase3_config_validation.py:197-205`).
- **Why this matters:** The plan required dynamic discovery of the production config surface from `config/*.yaml`, with exclusions/classification explicit if needed. This test can miss YAML files outside the `two_step_factors_*.yaml` prefix.

## 2. Open Questions

- None blocking review confidence.

## 3. Verification Summary

- **Observed recorded command:** `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py tests/test_phase4_orchestrated_judge.py -q` → receipt says `72 passed in 4.18s`.
- **Still required or not durably proven:** final `git status --short`, final `git diff --name-only`, final `git diff --name-only -- gecco-mh-dashboard`, and the focused tool-judge pytest command or an explicit skip rationale.
- **Contract rows verified:** DuckDB-first orchestrator test contains guards for `rebuild_from_artifacts` and JSON/bics `Path.glob` scanning (`tests/test_phase4_orchestrated_judge.py:632-643`) and constructs evidence from `diagnostics*.duckdb` (`:645-680`).
- **Contract rows not fully verified:** capability-limited prompts, lesion-free config discovery, durable baseline/final evidence.

## 4. Scope Audit

- `gecco/construct_feedback/tool_judge.py` — allowed implementation file, but prompt contract remains incomplete.
- `tests/test_phase3_config_validation.py` — allowed test file, but config discovery is narrower than planned.
- `tests/test_phase4_orchestrated_judge.py` — allowed test file.
- `docs/codebase_cleanup_plan/cleanup-acceptance-gaps-review-receipt.md` — **not the allowed receipt filename**; required `...review-fix-receipt.md` is missing.
- No implementation-owned dashboard changes are claimed in the receipt, but the final dashboard diff output is not durably recorded.
