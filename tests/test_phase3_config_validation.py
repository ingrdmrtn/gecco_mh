"""Contract tests for Phase 3 config validation remediation."""

from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from config.schema import GeCCoConfig, JudgeConfig, load_config
from gecco.coordination import apply_client_profile
from gecco.construct_feedback.tool_judge import (
    JudgeVerdict,
    ToolUsingJudge,
    _OpenAIToolLoop,
    _RANDOM_FEEDBACK_TEXT,
    _apply_capability_postprocessing,
    _build_summary_only_feedback,
)
from gecco.construct_feedback.orchestrated import run_orchestrated_judge_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config" / "archive"
NON_PRODUCTION_CONFIGS = {"judge_tool_example.yaml", "test_orchestrator.yaml"}
FULL_CAPABILITIES = [
    "attempted_models_overview",
    "performance_summary",
    "best_model_code",
    "diagnostic_detail",
    "recommendations",
    "tools",
]
PRODUCTION_CONFIGS = sorted(
    p.name for p in CONFIG_DIR.glob("*.yaml") if p.name not in NON_PRODUCTION_CONFIGS
)


def _has_lesion_surface(text: str) -> bool:
    """Return whether a filename or task name still exposes lesion wording."""
    lowered = text.lower()
    return (
        "_lesion" in lowered
        or "lesion_" in lowered
        or re.search(r"(?<![a-z0-9])lesion(?![a-z0-9])", lowered) is not None
    )


class DummyStore:
    """Minimal store stub for deterministic judge tests."""

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


def _write_config(tmp_path: Path, body: str) -> Path:
    """Write a temporary YAML config file for validation tests."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(body, encoding="utf-8")
    return config_path


def _minimal_config(judge_block: str) -> str:
    """Build a minimal valid config document with a custom judge block."""
    return f"""
task:
  name: "phase3_task"
  description: "Phase 3 test"
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
    """Build a minimal valid config document for a specific provider."""
    return _minimal_config(judge_block).replace('provider: "openai"', f'provider: "{provider}"')


def _load_real_config(name: str) -> GeCCoConfig:
    """Load a production config from the repository config directory."""
    return load_config(CONFIG_DIR / name)


def _build_judge(config_name: str, tmp_path: Path, store=None) -> ToolUsingJudge:
    """Instantiate a tool judge from a real config without loading an LLM."""
    cfg = _load_real_config(config_name)
    return ToolUsingJudge(
        cfg=cfg,
        diagnostic_store=store or DummyStore(),
        model=object(),
        tokenizer=None,
        results_dir=tmp_path,
    )


def _make_openai_response(content: str, tool_calls=None) -> SimpleNamespace:
    """Build a minimal OpenAI chat-completions response stub."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ]
    )


def test_load_config_returns_validated_model_with_explicit_capabilities(tmp_path):
    """Explicit judge capabilities should load through the validated schema."""
    config_path = _write_config(
        tmp_path,
        _minimal_config(
            """  \n  capabilities:
    - tools
    - performance_summary
    - best_model_code
  diagnostic_store:
    enabled: true
"""
        ),
    )

    cfg = load_config(str(config_path))

    assert isinstance(cfg, GeCCoConfig)
    assert cfg.task.instructions == ""
    assert cfg.judge is not None
    assert cfg.judge.capabilities == [
        "tools",
        "performance_summary",
        "best_model_code",
    ]


def test_llm_config_accepts_registered_provider_keys(tmp_path):
    """Registered provider keys should pass validation unchanged."""
    config_path = _write_config(
        tmp_path,
        _minimal_config_with_provider(
            "opencode-go",
            """  \n  capabilities:
    - tools
""",
        ),
    )

    cfg = load_config(str(config_path))

    assert cfg.llm.provider == "opencode-go"


def test_llm_config_rejects_unknown_or_substring_provider_key(tmp_path):
    """Typos and substring provider names should fail before model loading."""
    config_path = _write_config(
        tmp_path,
        _minimal_config_with_provider(
            "my-openrouter-proxy",
            """  \n  capabilities:
    - tools
""",
        ),
    )

    with pytest.raises(ValidationError, match="Registered LLM providers"):
        load_config(str(config_path))


def test_llm_config_rejects_case_variant_provider_key(tmp_path):
    """Provider validation must reject case variants of registered keys."""
    config_path = _write_config(
        tmp_path,
        _minimal_config_with_provider(
            "OpenAI",
            """  \n  capabilities:
    - tools
""",
        ),
    )

    with pytest.raises(ValidationError, match="Registered LLM providers"):
        load_config(str(config_path))


@pytest.mark.parametrize("config_name", PRODUCTION_CONFIGS)
def test_production_configs_load_through_validated_schema(config_name):
    """Every discovered production config should load through load_config()."""
    cfg = _load_real_config(config_name)

    assert isinstance(cfg, GeCCoConfig)
    assert cfg.task.goal
    assert not _has_lesion_surface(config_name)
    assert not _has_lesion_surface(cfg.task.name)


def test_full_generic_config_declares_explicit_full_capability_set():
    """The generic full config should represent the normal non-lesioned judge."""
    cfg = _load_real_config("two_step_factors_gemini3flash_generic.yaml")

    assert cfg.judge is not None
    assert cfg.judge.capabilities == FULL_CAPABILITIES


def test_cmg_config_declares_explicit_full_capability_set():
    """The CMG config should also use the explicit full capability set."""
    cfg = _load_real_config("two_step_factors_cmg.yaml")

    assert cfg.judge is not None
    assert cfg.judge.capabilities == FULL_CAPABILITIES


def test_apply_client_profile_supports_dict_backed_clients():
    """Validated configs should apply dict-backed generator profiles cleanly."""
    cfg = _load_real_config("two_step_factors_cmg.yaml")

    apply_client_profile(cfg, "generator")

    assert cfg.llm.models_per_iteration == 2
    assert cfg.llm.temperature == 0.3
    assert (
        "Focus on proposing diverse, creative candidate models for parallel evaluation."
        in cfg.llm.system_prompt
    )
    assert "Ensure all {n_models} candidate functions are valid and structurally different." in cfg.llm.system_prompt


def test_example_judge_fragment_is_not_treated_as_runtime_config():
    """Example judge fragments should stay separate from full runtime configs."""
    assert "judge_tool_example.yaml" not in PRODUCTION_CONFIGS
    assert "test_orchestrator.yaml" not in PRODUCTION_CONFIGS

    with pytest.raises(ValidationError):
        load_config(CONFIG_DIR / "judge_tool_example.yaml")


def test_load_config_rejects_legacy_lesion_block(tmp_path):
    """Legacy lesion-first judge settings should fail validation."""
    config_path = _write_config(
        tmp_path,
        _minimal_config(
            """  \n  capabilities:
    - tools
  lesion:
    enabled: true
    lesion_type: "no_tools"
"""
        ),
    )

    with pytest.raises(ValidationError, match="lesion"):
        load_config(str(config_path))


@pytest.mark.parametrize("orchestrated_value", [False, True])
def test_load_config_rejects_explicit_judge_orchestrated_field(tmp_path, orchestrated_value):
    """Explicit judge.orchestrated should fail validation regardless of value."""
    config_path = _write_config(
        tmp_path,
        _minimal_config(
            f"""  orchestrated: {str(orchestrated_value).lower()}
  capabilities:
    - performance_summary
"""
        ),
    )

    with pytest.raises(ValidationError, match=r"judge\.orchestrated has been retired"):
        load_config(str(config_path))


def test_load_config_rejects_unknown_judge_capability(tmp_path):
    """Unknown capabilities should fail fast during config loading."""
    config_path = _write_config(
        tmp_path,
        _minimal_config(
            """  \n  capabilities:
    - tools
    - definitely_not_real
"""
        ),
    )

    with pytest.raises(ValidationError, match="definitely_not_real"):
        load_config(str(config_path))


def test_load_config_rejects_duplicate_judge_capabilities(tmp_path):
    """Duplicate capabilities should be rejected during validation."""
    config_path = _write_config(
        tmp_path,
        _minimal_config(
            """  \n  capabilities:
    - tools
    - tools
"""
        ),
    )

    with pytest.raises(ValidationError, match="must not contain duplicates"):
        load_config(str(config_path))


def test_load_config_rejects_invalid_random_feedback_combination(tmp_path):
    """random_feedback should be an exclusive capability mode."""
    config_path = _write_config(
        tmp_path,
        _minimal_config(
            """  \n  capabilities:
    - random_feedback
    - performance_summary
"""
        ),
    )

    with pytest.raises(ValidationError, match="random_feedback"):
        load_config(str(config_path))


def test_load_config_rejects_persona_synthesis_without_personas(tmp_path):
    """persona_synthesis requires multiple personas or explicit profiles."""
    config_path = _write_config(
        tmp_path,
        _minimal_config(
            """  \n  capabilities:
    - persona_synthesis
"""
        ),
    )

    with pytest.raises(ValidationError, match="persona_synthesis"):
        load_config(str(config_path))


def test_load_config_rejects_retired_judge_mode_manual(tmp_path):
    """judge.mode=manual should fail with a clear retirement message."""
    config_path = _write_config(
        tmp_path,
        _minimal_config(
            """  mode: "manual"\n"""
        ),
    )

    with pytest.raises(ValidationError, match=r"judge\.mode has been retired"):
        load_config(str(config_path))


def test_load_config_rejects_retired_judge_mode_tool_using(tmp_path):
    """judge.mode=tool_using should also fail with the retirement message."""
    config_path = _write_config(
        tmp_path,
        _minimal_config(
            """  mode: "tool_using"\n"""
        ),
    )

    with pytest.raises(ValidationError, match=r"judge\.mode has been retired"):
        load_config(str(config_path))


def test_judge_config_public_contract_no_longer_exposes_mode_field():
    """The validated judge schema should no longer advertise judge.mode."""
    assert "mode" not in JudgeConfig.model_fields


def test_runtime_yaml_configs_do_not_set_retired_judge_mode():
    """Repository runtime configs should not keep stale judge.mode keys."""
    config_files = sorted(CONFIG_DIR.glob("*.yaml"))
    assert config_files

    for config_path in config_files:
        text = config_path.read_text(encoding="utf-8")
        assert 'mode: "tool_using"' not in text
        assert 'mode: "manual"' not in text
        assert "orchestrated:" not in text


def test_load_config_allows_persona_synthesis_with_explicit_profiles(tmp_path):
    """Explicit persona profiles should satisfy persona_synthesis validation."""
    config_path = _write_config(
        tmp_path,
        _minimal_config(
            """  capabilities:
    - persona_synthesis
  persona_profiles:
    default:
      metric_language: "plain language"
    explore:
      metric_language: "curious language"
"""
        ),
    )

    cfg = load_config(str(config_path))

    assert cfg.judge is not None
    assert set(cfg.judge.persona_profiles.keys()) == {"default", "explore"}


def test_empty_capabilities_yield_explicit_empty_feedback_trace(tmp_path):
    """An empty capability list should produce no substantive feedback."""
    config_path = _write_config(
        tmp_path,
        _minimal_config("""  capabilities: []
"""),
    )
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
    assert artifact.key_recommendations == []

    trace_path = tmp_path / "judge" / "iter1_run0.json"
    assert trace_path.exists()
    trace_payload = trace_path.read_text(encoding="utf-8")
    assert '"synthesized_feedback": {' in trace_payload
    assert '"default": ""' in trace_payload
    assert '"tool_call_trace": []' in trace_payload
    assert '"no_substantive_feedback": true' in trace_payload


def test_run_test_evaluation_guides_write_store_when_disabled(tmp_path, monkeypatch, capsys):
    """The test-evaluation CLI should tell users how to persist the diagnostic store."""
    from gecco.cli.run_test_evaluation import run_test_evaluation

    results_dir = tmp_path / "results" / "demo"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "shared_registry.duckdb").write_text("", encoding="utf-8")

    cfg = SimpleNamespace(
        task=SimpleNamespace(name="demo-task"),
        data=SimpleNamespace(
            path="data.csv",
            input_columns=["choice"],
            id_column="participant",
            splits={},
        ),
        evaluation=SimpleNamespace(n_test_models=1),
    )

    monkeypatch.setattr("gecco.cli.run_test_evaluation.configure_temp_dirs", lambda *args, **kwargs: None)
    monkeypatch.setattr("gecco.cli.run_test_evaluation.load_config", lambda config_path: cfg)
    monkeypatch.setattr(
        "gecco.cli.run_test_evaluation.SharedRegistry.open_existing",
        lambda path: SimpleNamespace(read=lambda: {}),
    )
    monkeypatch.setattr("gecco.cli.run_test_evaluation.load_splits", lambda loaded_cfg: [1])
    monkeypatch.setattr(
        "gecco.cli.run_test_evaluation.collect_candidates",
        lambda registry: [
            {
                "client_id": 0,
                "iteration": 1,
                "function_name": "candidate_model",
                "code": "def candidate_model():\n    return 1\n",
                "val_mean_nll": 1.23,
                "param_names": [],
            }
        ],
    )
    monkeypatch.setattr(
        "gecco.cli.run_test_evaluation.fit_one_on_test",
        lambda candidate, df_test, cfg, id_eval_data=None: {
            "model_name": candidate["function_name"],
            "val_nll": candidate["val_mean_nll"],
            "test_mean_BIC": 2.0,
            "test_mean_NLL": 3.0,
            "test_individual_BIC": [],
            "test_individual_NLL": [],
            "test_individual_differences": None,
        },
    )

    run_test_evaluation(config="demo.yaml", results_dir=str(results_dir), write_store=False)

    output = capsys.readouterr().out
    assert "--write-store" in output
    assert "Diagnostic store persistence is disabled" in output


def test_random_feedback_only_returns_deterministic_generic_feedback(tmp_path):
    """Noise mode should return fixed feedback without tool use or synthesis."""
    judge = _build_judge(
        "two_step_factors_gemini3flash_capabilities_random_feedback.yaml",
        tmp_path,
        store=object(),
    )

    artifact = run_orchestrated_judge_pipeline(
        judge=judge,
        cfg=judge.cfg,
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
    assert artifact.key_recommendations == []
    trace_payload = (tmp_path / "judge" / "iter2_run0.json").read_text(encoding="utf-8")
    assert '"random_feedback_only": true' in trace_payload
    assert '"tool_call_trace": []' in trace_payload


def test_random_feedback_only_short_circuits_orchestrated_analysis(tmp_path):
    """Noise mode should short-circuit the orchestrated analysis path as well."""
    judge = _build_judge(
        "two_step_factors_gemini3flash_capabilities_random_feedback.yaml",
        tmp_path,
        store=object(),
    )

    analysis = judge.get_feedback_analysis(iteration=2, run_idx=0, tag="")

    assert analysis["short_circuit"] is True
    assert analysis["random_feedback_only"] is True
    assert analysis["analysis_text"] == _RANDOM_FEEDBACK_TEXT
    assert analysis["trace"] == []


def test_summary_only_feedback_is_metric_only():
    """Summary-only helper should emit deterministic metric-only feedback."""
    analysis_data = {
        "n_total": 4,
        "n_ok": 3,
        "n_failed": 1,
        "best_iter_bic": 98.5,
        "best_bic": 95.0,
        "trajectory_str": "iter 0: 110.00 → iter 1: 98.50",
    }

    feedback = _build_summary_only_feedback(
        analysis_data, capabilities=["performance_summary"]
    )

    assert "Performance summary" in feedback
    assert "Best BIC this iteration: 98.50" in feedback
    for forbidden in [
        "recommend",
        "ppc",
        "residual",
        "diagnostic",
        "improving",
        "regressed",
        "plateaued",
        "total",
        "status",
        "alpha_model",
    ]:
        assert forbidden not in feedback.lower()
    assert "iter 0" not in feedback.lower()


def test_summary_only_config_short_circuits_persona_synthesis(tmp_path, monkeypatch):
    """Summary-only configs should bypass normal persona synthesis."""
    judge = _build_judge(
        "two_step_factors_gemini3flash_capabilities_summary_only.yaml",
        tmp_path,
    )
    monkeypatch.setattr(
        judge,
        "_request_structured_verdict_with_suffix",
        lambda *args, **kwargs: pytest.fail("summary-only synthesis should not call verdict extraction"),
    )
    analysis = {
        "iteration": 1,
        "analysis_text": "unused analysis text",
        "trace": [],
        "full_trace": [],
        "best_bic": 95.0,
        "best_iter_bic": 98.5,
        "n_total": 4,
        "n_ok": 3,
        "n_failed": 1,
        "trajectory": [],
        "trajectory_str": "iter 0: 110.00 → iter 1: 98.50",
        "is_stuck": False,
    }

    feedback, verdict_dict = judge.synthesize_for_persona(analysis, persona_name="default")

    assert feedback == analysis["analysis_text"]
    assert verdict_dict["synthesized_feedback"] == analysis["analysis_text"]
    assert verdict_dict["key_recommendations"] == []
    assert feedback == analysis["analysis_text"]


def test_summary_only_config_preserves_analysis_text_in_composed_synthesis(tmp_path, monkeypatch):
    """Composed narrow synthesis should preserve both deterministic sections unchanged."""
    judge = _build_judge(
        "two_step_factors_gemini3flash_capabilities_summary_only.yaml",
        tmp_path,
    )
    monkeypatch.setattr(
        judge,
        "_request_structured_verdict_with_suffix",
        lambda *args, **kwargs: pytest.fail("summary-only synthesis should not call verdict extraction"),
    )
    analysis = judge.get_feedback_analysis(iteration=1, run_idx=0, tag="")

    feedback, verdict_dict = judge.synthesize_for_persona(analysis, persona_name="default")

    assert feedback == analysis["analysis_text"]
    assert verdict_dict["synthesized_feedback"] == analysis["analysis_text"]
    assert "Models attempted" in feedback or "No models attempted" in feedback
    assert "Performance summary" in feedback


def test_summary_only_analysis_skips_fallback_generation(tmp_path, monkeypatch):
    """Summary-only configs should not perform unnecessary fallback generation."""
    judge = _build_judge(
        "two_step_factors_gemini3flash_capabilities_summary_only.yaml",
        tmp_path,
    )
    monkeypatch.setattr(
        judge,
        "_fallback_generate",
        lambda user_message: pytest.fail("summary-only analysis should not call fallback generation"),
    )

    analysis = judge.get_feedback_analysis(iteration=1, run_idx=0, tag="")

    assert analysis["narrow_deterministic"] is True
    assert analysis["trace"] == []


def test_no_tools_mode_uses_fallback_without_diagnostic_tool_loop(tmp_path, monkeypatch):
    """No-tools configs should skip diagnostic tool calls entirely."""
    judge = _build_judge(
        "two_step_factors_gemini3flash_capabilities_no_tools.yaml",
        tmp_path,
    )

    assert judge._tool_loop is None

    monkeypatch.setattr(judge, "_fallback_generate", lambda user_message: "analysis text")

    analysis = judge.get_feedback_analysis(iteration=1, run_idx=0, tag="")

    assert analysis["analysis_text"] == "analysis text"
    assert analysis["trace"] == []
    assert analysis["full_trace"] == []


def test_get_feedback_analysis_captures_no_recommendations_analysis_messages(
    tmp_path,
):
    """Analysis-phase prompts should omit recommendation wording before tool use."""
    judge = _build_judge(
        "two_step_factors_gemini3flash_capabilities_no_recommendations.yaml",
        tmp_path,
    )
    judge.provider = "openai"
    judge.model_name = "gpt-test"
    chat_create = MagicMock(
        side_effect=[
            _make_openai_response("Planning questions."),
            _make_openai_response("Analysis text."),
        ]
    )
    judge.model = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat_create))
    )
    judge._tool_loop = _OpenAIToolLoop(
        client=judge.model,
        model_name=judge.model_name,
        max_tokens=judge.max_tokens,
        temperature=judge.temperature,
        verbose=False,
    )

    analysis = judge.get_feedback_analysis(iteration=1, run_idx=0, tag="")

    assert analysis["analysis_text"] == "Analysis text."
    first_call_messages = chat_create.call_args_list[0].kwargs["messages"]
    system_prompt = first_call_messages[0]["content"].lower()
    user_prompt = first_call_messages[1]["content"].lower()

    for forbidden in [
        "recommendations",
        "what to try next",
        "key_recommendations",
        "actionable feedback",
        "improve the next iteration",
        "suggestions",
        "next-step ideas",
    ]:
        assert forbidden not in system_prompt
        assert forbidden not in user_prompt

    assert "synthesise feedback from the evidence you gathered" in system_prompt
    assert "please query the diagnostic database" in user_prompt
    assert "before calling any tools" in user_prompt


@pytest.mark.parametrize(
    ("config_name", "prompt_absent"),
    [
        (
            "two_step_factors_gemini3flash_capabilities_no_recommendations.yaml",
            [
                "recommendations",
                "what to try next",
                "key_recommendations",
                "actionable feedback",
                "improve the next iteration",
                "suggestions",
                "next-step ideas",
                "coverage",
                "citations",
                "cited_models",
                "cited models",
                "mechanistic_coherence",
                "mechanistic coherence",
                "six angles",
                "all six angles",
            ],
        ),
        (
            "two_step_factors_gemini3flash_capabilities_no_diagnostics.yaml",
            [
                "predictive adequacy",
                "ppc",
                "posterior predictive",
                "residual",
                "individual differences",
                "parameter recovery",
                "recovery",
                "r²",
                "r2",
                "coverage",
                "citations",
                "cited_models",
                "cited models",
                "mechanistic_coherence",
                "mechanistic coherence",
                "six angles",
                "all six angles",
            ],
        ),
    ],
)
def test_orchestrated_persona_synthesis_captures_capability_limited_llm_messages(
    config_name, prompt_absent, tmp_path
):
    """Orchestrated persona synthesis should trim disabled prompt content before LLM calls."""
    judge = _build_judge(config_name, tmp_path)
    judge.provider = "openai"
    judge.model_name = "gpt-test"
    chat_create = MagicMock(
        return_value=_make_openai_response(
            '{"per_angle":[],"key_recommendations":["Try a simpler model next."],'
            '"synthesized_feedback":"Performance summary: the model with separate '
            'learning rates (separate_lr_gain_loss, BIC=2847) improved over iteration '
            '3 run 1.\\n\\nRecommendations: Try a simpler model next.\\n\\nPPC '
            'diagnostics showed residual misfit in late trials."}'
        )
    )
    judge.model = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat_create))
    )
    analysis = {
        "iteration": 1,
        "analysis_text": "analysis text",
        "trace": [],
        "full_trace": [],
        "best_bic": 95.0,
        "trajectory": [],
        "is_stuck": False,
        "wall_time": 0.0,
    }

    feedback, verdict_dict = judge.synthesize_for_persona(analysis, persona_name="default")

    messages = chat_create.call_args.kwargs["messages"]
    system_prompt = messages[0]["content"]
    user_prompt = messages[2]["content"]

    for forbidden in prompt_absent:
        assert forbidden not in system_prompt.lower()
        assert forbidden not in user_prompt.lower()

    assert "what worked" in user_prompt.lower()
    assert "what partially worked" in user_prompt.lower()

    assert "Performance summary:" in feedback
    if "no_recommendations" in config_name:
        assert verdict_dict["key_recommendations"] == []
        assert "Recommendations:" not in feedback
    if "no_diagnostics" in config_name:
        assert "PPC diagnostics" not in feedback


def test_orchestrated_persona_synthesis_includes_diagnostic_detail_angles(tmp_path):
    """diagnostic_detail should keep core angles and add detailed diagnostics."""
    judge = _build_judge("two_step_factors_gemini3flash.yaml", tmp_path)
    judge.provider = "openai"
    judge.model_name = "gpt-test"
    chat_create = MagicMock(
        return_value=_make_openai_response(
            '{"per_angle":[],"key_recommendations":["Try a simpler model next."],'
            '"synthesized_feedback":"Performance summary: the model with separate '
            'learning rates (separate_lr_gain_loss, BIC=2847) improved over iteration '
            '3 run 1.\\n\\nRecommendations: Try a simpler model next.\\n\\nPPC '
            'diagnostics showed residual misfit in late trials."}'
        )
    )
    judge.model = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat_create))
    )
    analysis = {
        "iteration": 1,
        "analysis_text": "analysis text",
        "trace": [],
        "full_trace": [],
        "best_bic": 95.0,
        "trajectory": [],
        "is_stuck": False,
        "wall_time": 0.0,
    }

    judge.synthesize_for_persona(analysis, persona_name="default")

    system_prompt = chat_create.call_args.kwargs["messages"][0]["content"].lower()

    for phrase in [
        "statistical fit quality",
        "parameter identifiability",
        "parameter recovery",
        "predictive adequacy",
        "ppc",
        "residual",
        "individual differences",
        "r²",
        "mechanistic interpretability",
        "search breadth",
    ]:
        assert phrase in system_prompt


def test_capability_postprocessing_suppresses_recommendations_and_diagnostics():
    """Capability post-processing should remove disallowed feedback content."""
    verdict = JudgeVerdict(
        iteration=1,
        per_angle=[],
        key_recommendations=["Try a simpler model next."],
        synthesized_feedback=(
            "Performance summary: the model with separate learning rates "
            "(separate_lr_gain_loss, BIC=2847) improved over iteration 3 run 1.\n\n"
            "Recommendations: Try a simpler model next.\n\n"
            "PPC diagnostics showed residual misfit in late trials.\n\n"
            "Parameter recovery diagnostics showed low r² for one parameter.\n\n"
            "Recovery was weak for alpha.\n\n"
            "Individual differences suggested subgroup variability."
        ),
        tool_call_count=0,
        wall_time_seconds=0.0,
        best_bic=2847.0,
    )

    processed = _apply_capability_postprocessing(
        verdict,
        ["attempted_models_overview", "performance_summary"],
    )

    assert processed.key_recommendations == []
    assert "Recommendations:" not in processed.synthesized_feedback
    assert "BIC=2847" not in processed.synthesized_feedback
    assert "iteration 3 run 1" not in processed.synthesized_feedback.lower()
    assert "PPC diagnostics" not in processed.synthesized_feedback
    assert "recovery" not in processed.synthesized_feedback.lower()
    assert "r²" not in processed.synthesized_feedback.lower()
    assert "individual differences" not in processed.synthesized_feedback.lower()


def test_postprocessing_preserves_diagnostics_when_diagnostic_detail_enabled():
    """diagnostic_detail should preserve diagnostic sections."""
    verdict = JudgeVerdict(
        iteration=1,
        per_angle=[],
        key_recommendations=[],
        synthesized_feedback=(
            "Performance summary: best BIC improved.\n\n"
            "PPC diagnostics showed residual misfit in late trials.\n\n"
            "Parameter recovery diagnostics showed low r² for one parameter.\n\n"
            "Recovery was weak for alpha.\n\n"
            "Individual differences suggested subgroup variability."
        ),
        tool_call_count=0,
        wall_time_seconds=0.0,
        best_bic=100.0,
    )

    processed = _apply_capability_postprocessing(
        verdict,
        ["attempted_models_overview", "diagnostic_detail"],
    )

    assert "PPC diagnostics" in processed.synthesized_feedback
    assert "recovery" in processed.synthesized_feedback.lower()
    assert "r²" in processed.synthesized_feedback.lower()
    assert "individual differences" in processed.synthesized_feedback.lower()


@pytest.mark.parametrize(
    ("config_name", "expected_absent"),
    [
        (
            "two_step_factors_gemini3flash_capabilities_no_recommendations.yaml",
            "Recommendations:",
        ),
        (
            "two_step_factors_gemini3flash_capabilities_no_diagnostics.yaml",
            "PPC diagnostics",
        ),
    ],
)
def test_real_capability_config_shapes_feedback(config_name, expected_absent):
    """Real capability configs should shape feedback according to their capability sets."""
    cfg = _load_real_config(config_name)
    verdict = JudgeVerdict(
        iteration=1,
        per_angle=[],
        key_recommendations=["Try a simpler model next."],
        synthesized_feedback=(
            "Performance summary: the model with separate learning rates "
            "(separate_lr_gain_loss, BIC=2847) improved over iteration 3 run 1.\n\n"
            "Recommendations: Try a simpler model next.\n\n"
            "PPC diagnostics showed residual misfit in late trials."
        ),
        tool_call_count=0,
        wall_time_seconds=0.0,
        best_bic=2847.0,
    )

    processed = _apply_capability_postprocessing(verdict, cfg.judge.capabilities)

    assert expected_absent not in processed.synthesized_feedback
    if "no_recommendations" in config_name:
        assert processed.key_recommendations == []


def test_schema_accepts_diagnostic_detail_capability(tmp_path):
    """diagnostic_detail should be accepted by the schema as a valid capability."""
    config_path = _write_config(
        tmp_path,
        _minimal_config(
            """  capabilities:
    - attempted_models_overview
    - diagnostic_detail
"""
        ),
    )

    cfg = load_config(str(config_path))

    assert cfg.judge is not None
    assert "diagnostic_detail" in cfg.judge.capabilities


def test_schema_rejects_retired_citations(tmp_path):
    """citations should be rejected with a validation error."""
    config_path = _write_config(
        tmp_path,
        _minimal_config(
            """  capabilities:
    - citations
"""
        ),
    )

    with pytest.raises(ValidationError, match="citations"):
        load_config(str(config_path))


def test_schema_rejects_retired_coverage(tmp_path):
    """coverage should be rejected with a validation error."""
    config_path = _write_config(
        tmp_path,
        _minimal_config(
            """  capabilities:
    - coverage
"""
        ),
    )

    with pytest.raises(ValidationError, match="coverage"):
        load_config(str(config_path))


def test_schema_rejects_retired_mechanistic_coherence(tmp_path):
    """mechanistic_coherence should be rejected with a validation error."""
    config_path = _write_config(
        tmp_path,
        _minimal_config(
            """  capabilities:
    - mechanistic_coherence
"""
        ),
    )

    with pytest.raises(ValidationError, match="mechanistic_coherence"):
        load_config(str(config_path))


class _NamesDedupeStore:
    """Store stub with duplicate model names and hidden metrics/statuses."""

    def fetchone(self, query, params=None):
        return {"n_total": 3, "n_ok": 2, "n_failed": 1, "best_iter": 100.0}

    def fetchall(self, query, params=None):
        name = params[0] if params else 0
        if "GROUP BY m.iteration" in query:
            return [
                {"iteration": name - 1, "best_metric": 110.0, "n_models_total": 2, "n_ok": 2},
            ]
        if "iteration =" in query:
            assert "ORDER BY model_id" in query
            return [
                {"name": "alpha_model"},
                {"name": "beta_model"},
                {"name": "alpha_model"},
                {"name": "gamma_model"},
                {"name": "beta_model"},
            ]
        return []


def test_attempted_models_overview_dedupes_names(tmp_path):
    """Attempted-models overview should list unique names in first-seen order."""
    from types import SimpleNamespace
    judge = ToolUsingJudge.__new__(ToolUsingJudge)
    judge.capabilities = ["attempted_models_overview"]
    judge.store = _NamesDedupeStore()
    judge.results_dir = tmp_path
    judge.max_tool_calls = 20
    judge.stuck_search_cfg = SimpleNamespace(tolerance=10.0, window=2)

    analysis = judge.get_feedback_analysis(iteration=1, run_idx=0, tag="")

    text = analysis["analysis_text"]
    assert "alpha_model" in text
    assert "beta_model" in text
    assert "gamma_model" in text
    assert text.index("alpha_model") < text.index("beta_model")
    assert text.index("beta_model") < text.index("gamma_model")
    assert text.count("alpha_model") == 1
    assert text.count("beta_model") == 1
    assert text.count("gamma_model") == 1


def test_attempted_models_overview_contains_no_forbidden_terms(tmp_path):
    """Attempted-models overview should contain no metrics, statuses, etc."""
    from types import SimpleNamespace
    judge = ToolUsingJudge.__new__(ToolUsingJudge)
    judge.capabilities = ["attempted_models_overview"]
    judge.store = _NamesDedupeStore()
    judge.results_dir = tmp_path
    judge.max_tool_calls = 20
    judge.stuck_search_cfg = SimpleNamespace(tolerance=10.0, window=2)

    analysis = judge.get_feedback_analysis(iteration=1, run_idx=0, tag="")

    text = analysis["analysis_text"]
    forbidden = ["BIC", "bic", "metric", "total", "ok", "failed", "status",
                 "diagnostic", "ppc", "residual", "recommend", "trajectory",
                 "recovery", "r²", "r2", "model_id", "id"]
    for term in forbidden:
        assert term.lower() not in text.lower()


def test_attempted_models_overview_synthesis_preserves_analysis_text(tmp_path, monkeypatch):
    """Synthesised attempted-models feedback should preserve the deterministic analysis text."""
    from types import SimpleNamespace
    judge = ToolUsingJudge.__new__(ToolUsingJudge)
    judge.capabilities = ["attempted_models_overview"]
    judge.store = _NamesDedupeStore()
    judge.results_dir = tmp_path
    judge.max_tool_calls = 20
    judge.stuck_search_cfg = SimpleNamespace(tolerance=10.0, window=2)
    monkeypatch.setattr(
        judge,
        "_request_structured_verdict_with_suffix",
        lambda *args, **kwargs: pytest.fail("narrow deterministic synthesis should not call verdict extraction"),
    )
    monkeypatch.setattr(
        judge,
        "_fallback_generate",
        lambda *args, **kwargs: pytest.fail("narrow deterministic synthesis should not call fallback generation"),
    )

    analysis = judge.get_feedback_analysis(iteration=1, run_idx=0, tag="")

    feedback, verdict_dict = judge.synthesize_for_persona(analysis, persona_name="default")

    assert feedback == analysis["analysis_text"]
    assert verdict_dict["synthesized_feedback"] == analysis["analysis_text"]
    assert "alpha_model" in feedback
    assert "beta_model" in feedback
    assert "gamma_model" in feedback
    assert feedback.count("alpha_model") == 1
    assert feedback.count("beta_model") == 1
    assert feedback.count("gamma_model") == 1


def test_performance_summary_contains_metric_values_only(tmp_path):
    """Performance summary should contain metric values but no model names, counts, etc."""
    from types import SimpleNamespace
    judge = ToolUsingJudge.__new__(ToolUsingJudge)
    judge.capabilities = ["performance_summary"]
    judge.store = _NamesDedupeStore()
    judge.results_dir = tmp_path
    judge.max_tool_calls = 20
    judge.stuck_search_cfg = SimpleNamespace(tolerance=10.0, window=2)

    analysis = judge.get_feedback_analysis(iteration=1, run_idx=0, tag="")

    text = analysis["analysis_text"]
    forbidden = ["alpha_model", "beta_model", "gamma_model", "total", "ok",
                 "failed", "diagnostic", "ppc", "residual", "recommend",
                 "model_id", "id", "improving", "regressed", "plateaued",
                 "recovery", "r²", "r2"]
    for term in forbidden:
        assert term.lower() not in text.lower()
    assert "BIC" in text or "bic" in text


def test_composed_deterministic_emits_both_sections(tmp_path):
    """Composed attempted_models_overview + performance_summary should emit both."""
    from types import SimpleNamespace
    judge = ToolUsingJudge.__new__(ToolUsingJudge)
    judge.capabilities = ["attempted_models_overview", "performance_summary"]
    judge.store = _NamesDedupeStore()
    judge.results_dir = tmp_path
    judge.max_tool_calls = 20
    judge.stuck_search_cfg = SimpleNamespace(tolerance=10.0, window=2)

    analysis = judge.get_feedback_analysis(iteration=1, run_idx=0, tag="")

    text = analysis["analysis_text"]
    assert "Models attempted" in text
    assert "alpha_model" in text
    assert "beta_model" in text
    assert "BIC" in text or "bic" in text.lower()
    assert "improving" not in text.lower()
    assert "regressed" not in text.lower()
    assert "plateaued" not in text.lower()


def test_composed_deterministic_does_not_call_llm_or_fallback(tmp_path, monkeypatch):
    """Composed mode should bypass LLM/fallback paths entirely."""
    from types import SimpleNamespace
    judge = ToolUsingJudge.__new__(ToolUsingJudge)
    judge.capabilities = ["attempted_models_overview", "performance_summary"]
    judge.store = _NamesDedupeStore()
    judge.results_dir = tmp_path
    judge.max_tool_calls = 20
    judge.stuck_search_cfg = SimpleNamespace(tolerance=10.0, window=2)
    monkeypatch.setattr(
        judge,
        "_fallback_generate",
        lambda *args: pytest.fail("fallback should not be called"),
    )

    analysis = judge.get_feedback_analysis(iteration=1, run_idx=0, tag="")

    assert "Models attempted" in analysis["analysis_text"]
