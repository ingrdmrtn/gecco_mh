# Final Cleanup Genuine Issues Review

## 1. Findings

### High: Dict-backed `naive_ideation` still fails at runtime

- **Files:** `gecco/candidate_generation.py:489-510`, `tests/test_cmg_runtime.py:75-112`
- **Violated plan contract:** "Validated dict-backed client profiles work everywhere they are consumed" for `CandidateGenerator.generate_iteration()` / `generate_non_cmg_iteration()`.
- **Issue:** The implementation looks up `naive_ideation` with `_mapping_get()`, but then dereferences it as an object:

```python
persona = naive_cfg.persona
translation_preamble = getattr(naive_cfg, "translation_preamble", None)
```

If `cfg.clients` is dict-backed and contains:

```python
{"generator": {"naive_ideation": {"enabled": True, "persona": "..."}}
```

this will raise `AttributeError: 'dict' object has no attribute 'persona'`.

- **Test gap:** `test_cmg_generator_profile_lookup_supports_dict_backed_clients()` mocks `generate_models_naive`, so it proves dispatch only, not that dict-backed nested `naive_ideation` works through the actual generation path.

### Medium: `README.md` still points at deleted/stale script surfaces

- **File:** `README.md:80-82`, `README.md:597`
- **Violated plan contract:** Public docs must stop pointing users at deleted script entrypoints / active production launch surfaces.
- **Issue:** The README still lists:

```text
scripts/
    decision_making_demo.py
    two_step_demo.py
```

and the usage section still says:

```markdown
Quick start with demo scripts:
```

even though the commands below were changed to `python -m gecco ...`.

- **Why this matters:** The plan summary explicitly requires public docs to stop pointing users at deleted script entrypoints, not only to remove exact `python scripts/...` commands.

### Medium: Required verification proof is not available in the worktree

- **Violated plan sections:** Tests And Verification, Worktree rules, Acceptance Checklist.
- **Issue:** I found no final-cleanup implementation receipt or durable command-output artifact. The plan allowed no artifact, but this review requires verifying performed commands/checks. Without the implementer's final response or captured outputs, I could not verify:
  - pre/post `git status --short`
  - `git diff --name-only`
  - final untracked audit
  - `git diff --name-only -- gecco-mh-dashboard`
  - targeted pytest command result
  - README grep outputs

## 2. Open Questions

- None blocking code review beyond the missing durable verification evidence noted above.

## 3. Verification Summary

Observed by file review:

- `apply_client_profile()` now supports dict lookup for `clients` and nested `llm`.
- README no longer contains exact:
  - `python scripts/two_step_demo.py`
  - `python scripts/decision_making_demo.py`
  - `shared_registry.json`
  - `JSON registry`

Not observed / still required:

- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py tests/test_cmg_runtime.py tests/test_phase4_orchestrated_judge.py -q`
- pre/post `git status --short`
- `git diff --name-only`
- `git diff --name-only -- gecco-mh-dashboard`
- final untracked-file audit

Contract rows:

- Dict-backed profiles: **partially verified; failing gap remains for dict-backed `naive_ideation`.**
- Public docs: **partially verified; exact commands removed, stale script references remain.**
- Dashboard scope: **not verified due to lack of git diff/status output.**

## 4. Scope Audit

Files reviewed as implementation-relevant:

- `gecco/coordination.py` - allowed.
- `gecco/candidate_generation.py` - allowed.
- `gecco/run_gecco.py` - allowed.
- `gecco/construct_feedback/orchestrated.py` - allowed.
- `README.md` - allowed.
- `tests/test_phase3_config_validation.py` - allowed.
- `tests/test_cmg_runtime.py` - allowed.
- `tests/test_phase4_orchestrated_judge.py` - allowed.

Potential/suspicious:

- Could not conclusively distinguish pre-existing dirty files from implementation-owned changes because git status/diff outputs were unavailable.
- `README.md` still contains stale deleted script paths in the repository structure.
