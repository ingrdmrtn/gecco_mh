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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
NON_PRODUCTION_CONFIGS = {"judge_tool_example.yaml", "test_orchestrator.yaml"}
FULL_CAPABILITIES = [
    "attempted_models_overview",
    "performance_summary",
    "best_model_code",
    "recommendations",
    "mechanistic_coherence",
    "tools",
    "citations",
    "coverage",
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

    verdict = judge.get_feedback(iteration=1, run_idx=0, tag="")

    assert verdict.synthesized_feedback == ""
    assert verdict.key_recommendations == []

    trace_path = tmp_path / "judge" / "iter1_run0.json"
    assert trace_path.exists()
    trace_payload = trace_path.read_text(encoding="utf-8")
    assert '"synthesized_feedback": {' in trace_payload
    assert '"default": ""' in trace_payload
    assert '"tool_call_trace": []' in trace_payload
    assert '"no_substantive_feedback": true' in trace_payload


def test_random_feedback_only_returns_deterministic_generic_feedback(tmp_path):
    """Noise mode should return fixed feedback without tool use or synthesis."""
    judge = _build_judge(
        "two_step_factors_gemini3flash_capabilities_random_feedback.yaml",
        tmp_path,
        store=object(),
    )

    verdict = judge.get_feedback(iteration=2, run_idx=0, tag="")

    assert verdict.synthesized_feedback == _RANDOM_FEEDBACK_TEXT
    assert verdict.key_recommendations == []
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


def test_summary_only_feedback_is_concise_and_quantitative():
    """Summary-only helper should emit deterministic progress feedback only."""
    analysis_data = {
        "n_total": 4,
        "n_ok": 3,
        "n_failed": 1,
        "best_iter_bic": 98.5,
        "best_bic": 95.0,
        "trajectory_str": "iter 0: 110.00 → iter 1: 98.50",
    }

    feedback = _build_summary_only_feedback(analysis_data)

    assert "Iteration summary" in feedback
    assert "Models evaluated this iteration: 4 total, 3 successful, 1 failed" in feedback
    assert "Best BIC this iteration: 98.50" in feedback
    assert "recommend" not in feedback.lower()
    assert "ppc" not in feedback.lower()


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

    assert feedback == verdict_dict["synthesized_feedback"]
    assert verdict_dict["key_recommendations"] == []
    assert "Iteration summary" in feedback
    assert "unused analysis text" not in feedback


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

    assert analysis["summary_only"] is True
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
            ],
        ),
        (
            "two_step_factors_gemini3flash_capabilities_no_citations.yaml",
            ["cite model names", "bic values", "r values", "cited_models"],
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
    if "no_citations" in config_name:
        assert "BIC=2847" not in feedback


def test_capability_postprocessing_suppresses_recommendations_citations_and_diagnostics():
    """Capability post-processing should remove disallowed feedback content."""
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

    processed = _apply_capability_postprocessing(
        verdict,
        ["attempted_models_overview", "performance_summary"],
    )

    assert processed.key_recommendations == []
    assert "Recommendations:" not in processed.synthesized_feedback
    assert "BIC=2847" not in processed.synthesized_feedback
    assert "iteration 3 run 1" not in processed.synthesized_feedback.lower()
    assert "PPC diagnostics" not in processed.synthesized_feedback


@pytest.mark.parametrize(
    ("config_name", "expected_absent"),
    [
        (
            "two_step_factors_gemini3flash_capabilities_no_recommendations.yaml",
            "Recommendations:",
        ),
        (
            "two_step_factors_gemini3flash_capabilities_no_citations.yaml",
            "BIC=2847",
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
