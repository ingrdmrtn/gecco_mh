"""Contract tests for judge mode/context/output config validation."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from config.schema import EvaluationConfig, GeCCoConfig, JudgeConfig, load_config
from gecco.construct_feedback.tool_judge import (
    JudgeVerdict,
    ToolUsingJudge,
    _RANDOM_FEEDBACK_TEXT,
    _cap_tool_result,
    _build_summary_only_feedback,
    _apply_capability_postprocessing,
)
from gecco.construct_feedback.orchestrated import run_orchestrated_judge_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config" / "archive"


def _write_config(tmp_path: Path, body: str) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(body, encoding="utf-8")
    return config_path


def _q(s: str) -> str:
    """Quote a YAML value if it could be parsed as a bool."""
    return f'"{s}"'


def _minimal_config(judge_block: str) -> str:
    """Build a minimal valid config document with a custom judge block.
    Note: mode values must be quoted in YAML to avoid boolean parsing (off -> False).
    """
    return f"""
task:
  name: "test_task"
  description: "Test"
  goal: "Validate config loading"

data:
  path: "data.csv"
  id_column: "participant"
  input_columns: ["choice"]

llm:
  provider: "openai"
  base_model: "gpt-test"
  temperature: 0.1
  max_tokens: 128
  system_prompt: "Be concise"
  models_per_iteration: 1
  guardrails: []

evaluation:
  metric: "bic"
  fit_type: "group"

loop:
  max_iterations: 1

judge:
{judge_block}
"""


def _minimal_config_with_provider(provider: str, judge_block: str) -> str:
    return _minimal_config(judge_block).replace('provider: "openai"', f'provider: "{provider}"')


class DummyStore:
    def fetchone(self, query, params=None):
        if "COUNT(*) AS n_total" in query:
            return {"n_total": 4, "n_ok": 3, "n_failed": 1, "best_iter": 98.5}
        return None

    def fetchall(self, query, params=None):
        if "GROUP BY m.iteration" in query:
            return [
                {"iteration": 0, "best_metric": 110.0, "n_models_total": 2, "n_ok": 2},
                {"iteration": 1, "best_metric": 98.5, "n_models_total": 4, "n_ok": 3},
            ]
        return []


# ======================================================================
# Contract A: Config Schema Is Explicit And Fails Fast
# ======================================================================


def test_evaluation_config_defaults_to_two_way_split():
    """EvaluationConfig should default to a 70/30 train/test split."""
    cfg = EvaluationConfig()

    assert cfg.train_ratio == pytest.approx(0.7)
    assert cfg.test_ratio == pytest.approx(0.3)


def test_evaluation_config_accepts_valid_two_way_ratios():
    """Explicit train/test ratios that sum to one should validate."""
    cfg = EvaluationConfig(train_ratio=0.6, test_ratio=0.4)

    assert cfg.train_ratio == pytest.approx(0.6)
    assert cfg.test_ratio == pytest.approx(0.4)


def test_evaluation_config_rejects_invalid_two_way_ratios():
    """Train/test ratios that do not sum to one should fail validation."""
    with pytest.raises(ValidationError, match="must sum to 1.0"):
        EvaluationConfig(train_ratio=0.6, test_ratio=0.5)


def test_schema_accepts_valid_mode_off(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "off"
  context:
    attempted_models: false
    performance: false
    best_model_code: false
    diagnostic: false
  output:
    persona_synthesis: false
"""))
    cfg = load_config(str(config_path))
    assert cfg.judge is not None
    assert cfg.judge.mode == "off"


def test_barrier_defaults_let_orchestrator_write_before_client_timeout(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "llm"
  context:
    attempted_models: true
    performance: false
    best_model_code: false
    diagnostic: false
"""))

    cfg = load_config(config_path)

    assert cfg.judge.barrier.client_wait_seconds > (
        cfg.judge.barrier.orchestrator_wait_seconds + cfg.judge.barrier.retry_wait_seconds
    )


def test_schema_accepts_valid_mode_random(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "random"
  context:
    attempted_models: false
    performance: false
    best_model_code: false
    diagnostic: false
  output:
    persona_synthesis: false
"""))
    cfg = load_config(str(config_path))
    assert cfg.judge is not None
    assert cfg.judge.mode == "random"


def test_schema_accepts_valid_mode_static(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "static"
  context:
    attempted_models: true
    performance: true
    best_model_code: false
    diagnostic: false
"""))
    cfg = load_config(str(config_path))
    assert cfg.judge.mode == "static"


def test_schema_accepts_valid_mode_llm(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "llm"
  context:
    attempted_models: true
    performance: true
    best_model_code: false
    diagnostic: false
"""))
    cfg = load_config(str(config_path))
    assert cfg.judge.mode == "llm"


def test_schema_accepts_valid_mode_agent(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "agent"
  context:
    attempted_models: true
    performance: true
    best_model_code: false
    diagnostic: true
"""))
    cfg = load_config(str(config_path))
    assert cfg.judge.mode == "agent"


def test_schema_parses_individual_differences_eval_section(tmp_path):
    config_path = _write_config(
        tmp_path,
        _minimal_config("""  mode: "off"
  context:
    attempted_models: false
    performance: false
    best_model_code: false
    diagnostic: false
  output:
    persona_synthesis: false
""")
        + """
individual_differences_eval:
  data_path: "self_report.csv"
  id_column: "subj"
  behavioral_id_column: "subject_id"
  predictors: ["Factor1", "Factor2"]
  covariates: ["age"]
""",
    )

    cfg = load_config(str(config_path))

    assert cfg.individual_differences_eval is not None
    assert cfg.individual_differences_eval.data_path == "self_report.csv"
    assert cfg.individual_differences_eval.predictors == ["Factor1", "Factor2"]
    assert cfg.individual_differences_eval.covariates == ["age"]


def test_schema_rejects_random_with_context(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "random"
  context:
    attempted_models: true
    performance: false
    best_model_code: false
    diagnostic: false
"""))
    with pytest.raises(ValidationError, match="random"):
        load_config(str(config_path))


def test_schema_rejects_off_with_context(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "off"
  context:
    attempted_models: false
    performance: true
    best_model_code: false
    diagnostic: false
"""))
    with pytest.raises(ValidationError, match="off"):
        load_config(str(config_path))


def test_schema_rejects_random_with_persona_synthesis(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "random"
  context:
    attempted_models: false
    performance: false
    best_model_code: false
    diagnostic: false
  output:
    persona_synthesis: true
"""))
    with pytest.raises(ValidationError, match="random"):
        load_config(str(config_path))


def test_schema_rejects_off_with_persona_synthesis(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "off"
  context:
    attempted_models: false
    performance: false
    best_model_code: false
    diagnostic: false
  output:
    persona_synthesis: true
"""))
    with pytest.raises(ValidationError, match="off"):
        load_config(str(config_path))


def test_schema_rejects_agent_without_context(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "agent"
  context:
    attempted_models: false
    performance: false
    best_model_code: false
    diagnostic: false
"""))
    with pytest.raises(ValidationError, match="agent"):
        load_config(str(config_path))


def test_schema_rejects_unknown_mode(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "unknown_mode"
"""))
    with pytest.raises(ValidationError):
        load_config(str(config_path))


def test_schema_rejects_old_capabilities(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  capabilities:
    - tools
"""))
    with pytest.raises(ValidationError, match="capabilities has been retired"):
        load_config(str(config_path))


def test_schema_rejects_old_capabilities_with_full_config(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "llm"
  capabilities:
    - tools
"""))
    with pytest.raises(ValidationError, match="capabilities has been retired"):
        load_config(str(config_path))


def test_schema_rejects_retired_judge_mode_manual(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "manual"
"""))
    with pytest.raises(ValidationError, match="retired"):
        load_config(str(config_path))


def test_schema_rejects_retired_judge_mode_tool_using(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "tool_using"
"""))
    with pytest.raises(ValidationError, match="retired"):
        load_config(str(config_path))


def test_schema_rejects_orchestrated_field(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "llm"
  orchestrated: true
"""))
    with pytest.raises(ValidationError, match="orchestrated"):
        load_config(str(config_path))


def test_schema_rejects_persona_synthesis_without_personas(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "llm"
  context:
    attempted_models: true
  output:
    persona_synthesis: true
"""))
    with pytest.raises(ValidationError, match="persona_synthesis"):
        load_config(str(config_path))


def test_schema_accepts_individual_differences_with_static(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "static"
  context:
    attempted_models: false
    performance: false
    best_model_code: false
    diagnostic: false
    individual_differences: true
"""))
    cfg = load_config(str(config_path))
    assert cfg.judge.mode == "static"
    assert cfg.judge.context.individual_differences is True


def test_schema_accepts_individual_differences_with_llm(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "llm"
  context:
    attempted_models: false
    performance: false
    best_model_code: false
    diagnostic: false
    individual_differences: true
"""))
    cfg = load_config(str(config_path))
    assert cfg.judge.mode == "llm"
    assert cfg.judge.context.individual_differences is True


def test_schema_accepts_individual_differences_with_agent(tmp_path):
    """Agent is valid with only individual_differences enabled."""
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "agent"
  context:
    attempted_models: false
    performance: false
    best_model_code: false
    diagnostic: false
    individual_differences: true
"""))
    cfg = load_config(str(config_path))
    assert cfg.judge.mode == "agent"
    assert cfg.judge.context.individual_differences is True


def test_schema_rejects_off_with_individual_differences(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "off"
  context:
    attempted_models: false
    performance: false
    best_model_code: false
    diagnostic: false
    individual_differences: true
"""))
    with pytest.raises(ValidationError, match="off"):
        load_config(str(config_path))


def test_schema_rejects_random_with_individual_differences(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "random"
  context:
    attempted_models: false
    performance: false
    best_model_code: false
    diagnostic: false
    individual_differences: true
"""))
    with pytest.raises(ValidationError, match="random"):
        load_config(str(config_path))


def test_schema_rejects_unknown_judge_context_key(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "llm"
  context:
    attempted_models: true
    performance: false
    best_model_code: false
    diagnostic: false
    unexpected_flag: true
"""))
    with pytest.raises(ValidationError, match="unexpected_flag"):
        load_config(str(config_path))


def test_schema_rejects_unknown_judge_output_key(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "llm"
  context:
    attempted_models: true
    performance: false
    best_model_code: false
    diagnostic: false
  output:
    persona_synthesis: false
    extra_output: true
"""))
    with pytest.raises(ValidationError, match="extra_output"):
        load_config(str(config_path))


def test_schema_allows_persona_synthesis_with_explicit_profiles(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "llm"
  context:
    attempted_models: true
  output:
    persona_synthesis: true
  persona_profiles:
    default:
      metric_language: "plain language"
    explore:
      metric_language: "curious language"
"""))
    cfg = load_config(str(config_path))
    assert cfg.judge is not None
    assert set(cfg.judge.persona_profiles.keys()) == {"default", "explore"}


def test_empty_judge_section_produces_no_feedback(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "off"
"""))
    cfg = load_config(str(config_path))
    judge = ToolUsingJudge(
        cfg=cfg,
        diagnostic_store=object(),
        model=object(),
        tokenizer=None,
        results_dir=tmp_path,
    )
    artifact = run_orchestrated_judge_pipeline(
        judge=judge,
        cfg=cfg,
        results_dir=tmp_path,
        iteration=1,
        run_idx=0,
        tag="",
        best_model=None,
        best_metric=None,
        recovery_failures=None,
        prev_had_success=True,
    )
    assert artifact.synthesized_feedback == {"default": ""}
    trace_payload = (tmp_path / "judge" / "iter1_run0.json").read_text(encoding="utf-8")
    assert '"synthesized_feedback": {' in trace_payload
    assert '"default": ""' in trace_payload


# ======================================================================
# Random mode tests
# ======================================================================


def test_random_mode_returns_deterministic_generic_feedback(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "random"
  context:
    attempted_models: false
    performance: false
    best_model_code: false
    diagnostic: false
  output:
    persona_synthesis: false
"""))
    cfg = load_config(str(config_path))
    judge = ToolUsingJudge(
        cfg=cfg,
        diagnostic_store=object(),
        model=object(),
        tokenizer=None,
        results_dir=tmp_path,
    )
    artifact = run_orchestrated_judge_pipeline(
        judge=judge,
        cfg=cfg,
        results_dir=tmp_path,
        iteration=2,
        run_idx=0,
        tag="",
        best_model=None,
        best_metric=None,
        recovery_failures=None,
        prev_had_success=True,
    )
    assert artifact.synthesized_feedback == {"default": _RANDOM_FEEDBACK_TEXT}
    trace_payload = (tmp_path / "judge" / "iter2_run0.json").read_text(encoding="utf-8")
    assert '"random_feedback_only": true' in trace_payload
    assert '"tool_call_trace": []' in trace_payload


def test_random_mode_short_circuits_analysis(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "random"
  context:
    attempted_models: false
    performance: false
    best_model_code: false
    diagnostic: false
  output:
    persona_synthesis: false
"""))
    cfg = load_config(str(config_path))
    judge = ToolUsingJudge(
        cfg=cfg,
        diagnostic_store=object(),
        model=object(),
        tokenizer=None,
        results_dir=tmp_path,
    )
    analysis = judge.get_feedback_analysis(iteration=2, run_idx=0, tag="")
    assert analysis["short_circuit"] is True
    assert analysis["random_feedback_only"] is True
    assert analysis["analysis_text"] == _RANDOM_FEEDBACK_TEXT
    assert analysis["trace"] == []


# ======================================================================
# Static mode tests
# ======================================================================


class _NamesOnlyStore:
    def fetchone(self, query, params=None):
        return {"n_total": 3, "n_ok": 2, "n_failed": 1, "best_iter": 100.0}

    def fetchall(self, query, params=None):
        if "GROUP BY m.iteration" in query:
            return [
                {"iteration": 0, "best_metric": 110.0, "n_models_total": 2, "n_ok": 2},
            ]
        if "iteration =" in query:
            return [
                {"name": "alpha_model"},
                {"name": "beta_model"},
                {"name": "gamma_model"},
            ]
        return []


def test_static_attempted_only_has_names_only(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "static"
  context:
    attempted_models: true
    performance: false
    best_model_code: false
    diagnostic: false
"""))
    cfg = load_config(str(config_path))
    judge = ToolUsingJudge(cfg=cfg, diagnostic_store=_NamesOnlyStore(), model=object(), tokenizer=None, results_dir=tmp_path)
    analysis = judge.get_feedback_analysis(iteration=1, run_idx=0, tag="")
    text = analysis["analysis_text"]
    assert "alpha_model" in text
    assert "beta_model" in text
    assert "gamma_model" in text
    forbidden = ["BIC", "bic", "metric", "ok", "failed", "status", "diagnostic", "ppc", "recommend", "trajectory", "recovery", "r²"]
    for term in forbidden:
        assert term.lower() not in text.lower()


def test_static_attempted_and_performance(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "static"
  context:
    attempted_models: true
    performance: true
    best_model_code: false
    diagnostic: false
"""))
    cfg = load_config(str(config_path))
    judge = ToolUsingJudge(cfg=cfg, diagnostic_store=_NamesOnlyStore(), model=object(), tokenizer=None, results_dir=tmp_path)
    analysis = judge.get_feedback_analysis(iteration=1, run_idx=0, tag="")
    text = analysis["analysis_text"]
    assert "alpha_model" in text
    assert "BIC" in text or "bic" in text.lower()
    assert "recommend" not in text.lower()


def test_static_does_not_call_llm_or_tools(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "static"
  context:
    attempted_models: true
    performance: true
    best_model_code: false
    diagnostic: false
"""))
    cfg = load_config(str(config_path))
    judge = ToolUsingJudge(cfg=cfg, diagnostic_store=_NamesOnlyStore(), model=object(), tokenizer=None, results_dir=tmp_path)
    monkeypatch.setattr(judge, "_fallback_generate", lambda *args: pytest.fail("fallback should not be called"))
    analysis = judge.get_feedback_analysis(iteration=1, run_idx=0, tag="")
    assert analysis["analysis_text"] != ""
    assert analysis["trace"] == []


# ======================================================================
# Summary-only feedback helper tests
# ======================================================================


def test_summary_only_feedback_is_metric_only():
    analysis_data = {
        "n_total": 4,
        "n_ok": 3,
        "n_failed": 1,
        "best_iter_bic": 98.5,
        "best_bic": 95.0,
        "trajectory_str": "110.00 → 98.50",
    }
    feedback = _build_summary_only_feedback(analysis_data, capabilities=["performance_summary"])
    assert "Performance summary" in feedback
    assert "Best BIC this iteration: 98.50" in feedback
    for forbidden in ["recommend", "ppc", "residual", "diagnostic", "improving", "regressed", "plateaued"]:
        assert forbidden not in feedback.lower()
    assert "iter 0" not in feedback.lower()


def test_cap_tool_result_uses_neutral_truncation_hint():
    result = "x" * 40001

    capped = _cap_tool_result(result, raw_result=[{"row": 1}, {"row": 2}])

    assert "[truncated: result too large" in capped
    assert "2 rows total" in capped
    assert "use narrower allowed filters to reduce result size" in capped
    for forbidden in ["status=", "code_contains=", "param_contains="]:
        assert forbidden not in capped


# ======================================================================
# LLM/compatibility layer tests
# ======================================================================


def test_summary_only_synthesis_preserves_analysis_text(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "static"
  context:
    attempted_models: true
    performance: true
    best_model_code: false
    diagnostic: false
"""))
    cfg = load_config(str(config_path))
    judge = ToolUsingJudge(cfg=cfg, diagnostic_store=_NamesOnlyStore(), model=object(), tokenizer=None, results_dir=tmp_path)
    monkeypatch.setattr(judge, "_request_structured_verdict_with_suffix", lambda *args, **kwargs: pytest.fail("should not call verdict extraction"))
    monkeypatch.setattr(judge, "_fallback_generate", lambda *args: pytest.fail("fallback should not be called"))
    analysis = judge.get_feedback_analysis(iteration=1, run_idx=0, tag="")
    feedback, verdict_dict = judge.synthesize_for_persona(analysis, persona_name="default")
    assert feedback == analysis["analysis_text"]
    assert verdict_dict["synthesized_feedback"] == analysis["analysis_text"]
    assert "alpha_model" in feedback


def test_empty_mode_off_produces_no_substantive_feedback(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "off"
"""))
    cfg = load_config(str(config_path))
    judge = ToolUsingJudge(cfg=cfg, diagnostic_store=object(), model=object(), tokenizer=None, results_dir=tmp_path)
    artifact = run_orchestrated_judge_pipeline(
        judge=judge, cfg=cfg, results_dir=tmp_path, iteration=1, run_idx=0, tag="",
        best_model=None, best_metric=None, recovery_failures=None, prev_had_success=True,
    )
    assert artifact.synthesized_feedback == {"default": ""}


def test_context_postprocessing_preserves_recommendations_for_llm_mode(tmp_path):
    """Mode-aware postprocessing should keep recommendations in LLM mode."""
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "llm"
  context:
    attempted_models: true
    performance: false
    best_model_code: false
    diagnostic: false
"""))
    cfg = load_config(str(config_path))
    verdict = JudgeVerdict(
        iteration=1, per_angle=[], key_recommendations=["Try a simpler model next."],
        synthesized_feedback=(
            "Performance summary: best BIC improved.\n\n"
            "Recommendations: Try a simpler model next.\n\n"
            "PPC diagnostics showed residual misfit."
        ),
        tool_call_count=0, wall_time_seconds=0.0, best_bic=2847.0,
    )
    processed = _apply_capability_postprocessing(verdict, cfg.judge)
    assert processed.key_recommendations == ["Try a simpler model next."]
    assert "Recommendations:" in processed.synthesized_feedback
    assert "PPC diagnostics" not in processed.synthesized_feedback
    assert "BIC=2847" not in processed.synthesized_feedback


def test_diagnostic_detail_preserved_in_postprocessing(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "llm"
  context:
    attempted_models: true
    performance: false
    best_model_code: false
    diagnostic: false
    individual_differences: true
"""))
    cfg = load_config(str(config_path))
    verdict = JudgeVerdict(
        iteration=1, per_angle=[], key_recommendations=[],
        synthesized_feedback=(
            "PPC diagnostics showed residual misfit and parameter recovery issues.\n\n"
            "Individual differences showed mean R²=0.42 and r2 evidence for heterogeneity.\n\n"
            "Self-report heterogeneity supported the regression pattern."
        ),
        tool_call_count=0, wall_time_seconds=0.0, best_bic=100.0,
    )
    processed = _apply_capability_postprocessing(verdict, cfg.judge)
    assert "PPC diagnostics" not in processed.synthesized_feedback
    assert "residual misfit" not in processed.synthesized_feedback
    assert "parameter recovery" not in processed.synthesized_feedback
    assert "Individual differences" in processed.synthesized_feedback
    assert "R²=0.42" in processed.synthesized_feedback
    assert "r2 evidence" in processed.synthesized_feedback.lower()
    assert "heterogeneity" in processed.synthesized_feedback.lower()
    assert "self-report" in processed.synthesized_feedback.lower()


def test_individual_differences_postprocessing_removes_generic_heterogeneity_bullets(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "llm"
  context:
    attempted_models: true
    performance: false
    best_model_code: false
    diagnostic: false
    individual_differences: false
"""))
    cfg = load_config(str(config_path))
    verdict = JudgeVerdict(
        iteration=1,
        per_angle=[],
        key_recommendations=[],
        synthesized_feedback=(
            "Findings:\n"
            "- Model fit is stable.\n"
            "- Heterogeneity was high across participants."
        ),
        tool_call_count=0,
        wall_time_seconds=0.0,
        best_bic=100.0,
    )
    processed = _apply_capability_postprocessing(verdict, cfg.judge)
    assert "Model fit is stable." in processed.synthesized_feedback
    assert "Heterogeneity was high across participants." not in processed.synthesized_feedback


def test_individual_differences_postprocessing_preserves_diagnostic_heterogeneity_lines(tmp_path):
    config_path = _write_config(tmp_path, _minimal_config("""  mode: "llm"
  context:
    attempted_models: true
    performance: false
    best_model_code: false
    diagnostic: true
    individual_differences: false
"""))
    cfg = load_config(str(config_path))
    verdict = JudgeVerdict(
        iteration=1,
        per_angle=[],
        key_recommendations=[],
        synthesized_feedback=(
            "Findings:\n"
            "- Heterogeneity was high across participants.\n"
            "- PPC heterogeneity across posterior predictive checks was reviewed.\n"
            "- Model fit is stable."
        ),
        tool_call_count=0,
        wall_time_seconds=0.0,
        best_bic=100.0,
    )
    processed = _apply_capability_postprocessing(verdict, cfg.judge)
    assert "Heterogeneity was high across participants." not in processed.synthesized_feedback
    assert "PPC heterogeneity across posterior predictive checks was reviewed." in processed.synthesized_feedback
    assert "Model fit is stable." in processed.synthesized_feedback
