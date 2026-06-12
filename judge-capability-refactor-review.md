# Judge Capability Refactor Review

## 1. Findings

### High: `diagnostic_detail` gating is incomplete in active prompt paths

- **Violation:** Contract row `diagnostic_detail replaces coverage`; phase 6; plan lines 62 and 124 require PPC, residual, recovery, and individual-difference detail only when `diagnostic_detail` is enabled.
- **Evidence:** `gecco/construct_feedback/tool_judge.py:295-303` keeps parameter recovery instructions in `_CORE_ANGLES`, which are used when `diagnostic_detail` is absent (`_build_judge_system_prompt`, lines 400-406). The default synthesis profile also keeps diagnostic terms without the capability: `parameter recovery correlations (r), and R²` at `tool_judge.py:1115`.
- **Impact:** Non-deterministic judges without `diagnostic_detail` can still be instructed to inspect/report recovery/R²-style diagnostic details, violating the narrowed public capability contract.

### Medium: `performance_summary` is not strictly metric-only

- **Violation:** Contract row `performance_summary means metric-only performance`; plan lines 60 and 112-114.
- **Evidence:** `_build_summary_only_feedback()` includes `BIC trajectory so far: {trajectory_str}` (`tool_judge.py:196-207`), and `_format_trajectory()` appends non-metric status labels such as `(improving)`, `(regressed)`, and `(plateaued)` (`tool_judge.py:1361-1369`).
- **Impact:** `performance_summary` can emit qualitative trajectory/status prose rather than only metric values. Existing tests only assert that `BIC` appears and do not reject these labels.

### Medium: Narrow deterministic modes are short-circuited too late relative to the plan gate

- **Violation:** Phase 3 line 109 requires short-circuiting before generic prompt construction.
- **Evidence:** `get_feedback_analysis()` builds the full `_JUDGE_USER_TEMPLATE` message and previous-verdict context at `tool_judge.py:1634-1689` before checking `_is_narrow_deterministic_set()` at `tool_judge.py:1691`.
- **Impact:** Although the current implementation does not send that prompt to the LLM for narrow modes, it does not satisfy the stated gate and keeps forbidden status/count/recommendation context in the narrow-mode path.

### Medium: Required persistence proof is weakened by mocking the behavior under review

- **Violation:** Required test in plan lines 77-78 and forbidden pattern line 139.
- **Evidence:** `tests/test_phase4_orchestrated_judge.py:288-300` mocks both `judge.get_feedback_analysis` and `judge.synthesize_for_persona` in the deterministic persistence test.
- **Impact:** The test proves the orchestrator persists an arbitrary mocked string, not that the real deterministic narrow judge output is preserved without recommendations or persona prose.

### Low: Prompt-capture tests miss disabled recovery/detail terms in actual messages

- **Violation:** Contract row `Prompt layers obey capabilities`; manual review check line 92.
- **Evidence:** The disabled-diagnostics prompt assertions at `tests/test_phase3_config_validation.py:763-777` check PPC/residual/individual-difference terms but do not check `parameter recovery`, `recovery`, or `R²`, despite the plan naming recovery as gated diagnostic detail.
- **Impact:** The test suite allows the prompt-gating regression described above to pass.

## 2. Open Questions

- I could not observe `git status --short`, branch creation, or test-command output with the available tools. If an implementation receipt exists outside the worktree, it is needed to verify setup/final status and command results.

## 3. Verification Summary

### Observed by file review

- `JudgeCapability` accepts `diagnostic_detail` and no longer includes `citations`, `coverage`, or `mechanistic_coherence` in `config/schema.py:16-25`.
- Narrow deterministic bypass exists in `get_feedback_analysis()` / `synthesize_for_persona()` for `attempted_models_overview` and `performance_summary` (`tool_judge.py:1691-1720`, `1800-1810`).
- Retired capability names were found in tests and comments/config names; no active runtime capability aliases were observed in the reviewed runtime code.

### Commands/checks not observed and still required

- `git status --short` initial/final and branch verification.
- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py`.
- `conda run -n gecco_mh pytest tests/test_phase4_orchestrated_judge.py tests/test_judge_orchestration.py tests/test_cmg_judge.py`.
- Lint/type command or justified skip.

### Contract rows verified / not verified

- Verified partially: retired capability names removed from schema; deterministic bypass implemented.
- Not fully verified: names-only attempted-model output proof, metric-only performance output proof, diagnostic-detail prompt gating, prompt-layer message capture, persistence of real deterministic output.

## 4. Scope Audit

### Files observed with relevant changes/content

- `config/schema.py` — allowed.
- `gecco/construct_feedback/tool_judge.py` — allowed.
- `gecco/construct_feedback/orchestrated.py` — allowed only for small artifact assembly adjustment; no suspicious change identified in reviewed section.
- `tests/test_phase3_config_validation.py` — allowed.
- `tests/test_phase4_orchestrated_judge.py` — allowed.
- `config/archive/*.yaml` — allowed only for `judge.capabilities` updates; grep showed `diagnostic_detail` capability entries and no active retired capability entries, though some filenames/comments still contain `citations`.

### Forbidden/suspicious changes

- No changes to `gecco/diagnostic_store/tools.py` were required by this plan; grep still finds pre-existing explanatory `coverage` wording there, but this file is outside allowed implementation scope and was not reviewed as changed.
- Worktree dirtiness and pre-existing-vs-plan changes could not be distinguished without git status/diff output.
