"""Contract tests for agent context-gated tool access."""

from __future__ import annotations

import json
import sys
import types as pytypes
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.schema import load_config
from gecco.construct_feedback.tool_judge import ToolUsingJudge
from gecco.diagnostic_store.tools import dispatch_tool, get_judge_tool_names


class _AgentStore:
    def fetchone(self, query, params=None):
        if "SELECT model_id FROM models WHERE iteration = ?" in query:
            return {"model_id": 11}
        if "COUNT(*) AS n_total" in query:
            return {"n_total": 4, "n_ok": 3, "n_failed": 1, "best_iter": 98.5}
        if "MIN(CASE WHEN status='ok' THEN metric_value END) AS best_iter" in query:
            return {"best_iter": 98.5}
        return None

    def fetchall(self, query, params=None):
        if "SELECT name FROM models WHERE iteration = ?" in query:
            return [
                {"name": "alpha_model"},
                {"name": "beta_model"},
                {"name": "alpha_model"},
            ]
        if "GROUP BY m.iteration" in query:
            return [
                {"iteration": 0, "best_metric": 110.0, "n_models_total": 2, "n_ok": 2},
                {"iteration": 1, "best_metric": 98.5, "n_models_total": 4, "n_ok": 3},
            ]
        return []


class _DiagnosticStore(_AgentStore):
    def fetchone(self, query, params=None):
        if "SELECT model_id FROM models WHERE iteration = ?" in query:
            return {"model_id": 11}
        if "SELECT m.model_id, m.name, m.code, i.iteration, i.run_idx" in query:
            return {
                "model_id": 11,
                "name": "best_model",
                "iteration": 0,
                "run_idx": 0,
                "code": "def safe_best_model():\n    return 1\n",
            }
        if "SELECT 1 FROM models WHERE model_id = ?" in query:
            return {"1": 1}
        if "SELECT 1 FROM ppc WHERE model_id = ? LIMIT 1" in query:
            return {"1": 1}
        if "SELECT 1 FROM block_residuals WHERE model_id = ? LIMIT 1" in query:
            return {"1": 1}
        if "COUNT(DISTINCT participant_id) AS n_participants" in query:
            return {"n_participants": 2}
        if "FROM parameter_recovery WHERE model_id = ?" in query:
            return {
                "mean_r": 0.42,
                "worst_params": [{"name": "alpha"}, {"name": "beta"}],
            }
        if "FROM individual_differences WHERE model_id = ?" in query:
            return {"mean_r2": 0.12}
        return super().fetchone(query, params)

    def fetchall(self, query, params=None):
        if "FROM models m" in query and "m.status = 'ok'" in query and "m.metric_value IS NOT NULL" in query:
            return [
                {
                    "model_id": 11,
                    "run_idx": 0,
                    "iteration": 0,
                    "name": "best_model",
                    "metric_name": "BIC",
                    "metric_value": 10.0,
                    "param_names": ["alpha"],
                    "status": "ok",
                },
                {
                    "model_id": 12,
                    "run_idx": 0,
                    "iteration": 1,
                    "name": "runner_up",
                    "metric_name": "BIC",
                    "metric_value": 12.5,
                    "param_names": ["beta"],
                    "status": "ok",
                },
            ]
        if "FROM ppc" in query:
            return [
                {
                    "statistic_name": "stay_rate",
                    "condition": "rewarded",
                    "n_participants": 2,
                    "n_outside_95ci": 1,
                    "frac_outside_95ci": 0.5,
                    "mean_observed": 0.7,
                    "mean_simulated_mean": 0.6,
                    "mean_abs_zscore": 1.1,
                }
            ]
        if "FROM block_residuals" in query and "GROUP BY block_idx" in query:
            return [
                {
                    "block_idx": 0,
                    "block_start": 0,
                    "block_end": 10,
                    "mean_nll_per_trial_mean": 1.2,
                    "mean_nll_per_trial_std": 0.1,
                    "mean_nll_per_trial_min": 1.1,
                    "mean_nll_per_trial_max": 1.3,
                    "n_participants": 2,
                }
            ]
        if "ROW_NUMBER() OVER" in query:
            return [
                {
                    "participant_idx": 0,
                    "model_id": 11,
                    "name": "best_model",
                    "iteration": 0,
                    "run_idx": 0,
                    "bic": 9.8,
                },
                {
                    "participant_idx": 1,
                    "model_id": 12,
                    "name": "runner_up",
                    "iteration": 1,
                    "run_idx": 0,
                    "bic": 12.1,
                },
            ]
        if "LEFT JOIN parameter_recovery pr" in query and "WHERE m.model_id IN" in query:
            return [
                {
                    "model_id": 11,
                    "name": "best_model",
                    "iteration": 0,
                    "metric_value": 10.0,
                    "recovery_mean_r": 0.42,
                }
            ]
        return super().fetchall(query, params)


def _make_cfg(provider: str, context: dict[str, bool], mode: str = "agent") -> SimpleNamespace:
    return SimpleNamespace(
        llm=SimpleNamespace(
            provider=provider,
            base_model="gpt-test",
            max_output_tokens=64,
            temperature=0.0,
        ),
        judge=SimpleNamespace(
            mode=mode,
            context=SimpleNamespace(**context),
            output=SimpleNamespace(persona_synthesis=False),
            max_tool_calls=1,
            verbose=False,
        ),
    )


def _write_real_config(tmp_path: Path, context: dict[str, bool], mode: str = "agent") -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
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
  mode: "{mode}"
  context:
    attempted_models: {str(context['attempted_models']).lower()}
    performance: {str(context['performance']).lower()}
    best_model_code: {str(context['best_model_code']).lower()}
    diagnostic: {str(context['diagnostic']).lower()}
    individual_differences: {str(context['individual_differences']).lower()}
""",
        encoding="utf-8",
    )
    return config_path


def _openai_response(content: str = "", tool_calls: list[SimpleNamespace] | None = None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls or [])
            )
        ]
    )


class _OpenAIClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _install_fake_gemini_modules(monkeypatch):
    types_mod = pytypes.ModuleType("google.genai.types")

    class FunctionDeclaration:
        def __init__(self, name, description, parameters):
            self.name = name
            self.description = description
            self.parameters = parameters

    class Tool:
        def __init__(self, function_declarations):
            self.function_declarations = function_declarations

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    types_mod.FunctionDeclaration = FunctionDeclaration
    types_mod.Tool = Tool
    types_mod.GenerateContentConfig = GenerateContentConfig

    genai_mod = pytypes.ModuleType("google.genai")
    genai_mod.types = types_mod

    google_mod = pytypes.ModuleType("google")
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.generativeai", genai_mod)
    return types_mod


def _gemini_response(text: str = "", function_call: SimpleNamespace | None = None):
    parts = []
    if text:
        parts.append(SimpleNamespace(text=text))
    if function_call is not None:
        parts.append(SimpleNamespace(function_call=function_call))
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))],
    )


class _GeminiClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self.responses.pop(0)


def _run_agent_judge(provider: str, context: dict[str, bool], responses, monkeypatch):
    cfg = _make_cfg(provider, context)
    if provider == "gemini":
        _install_fake_gemini_modules(monkeypatch)
        client = _GeminiClient(responses)
    else:
        client = _OpenAIClient(responses)

    judge = ToolUsingJudge(
        cfg=cfg,
        diagnostic_store=_AgentStore(),
        model=client,
        tokenizer=None,
        results_dir=None,
    )
    analysis = judge.get_feedback_analysis(iteration=0, run_idx=0, tag="")
    return analysis, client


def _tool_schema_dump(provider: str, client) -> str:
    if provider == "openai":
        payload = client.calls[0]["tools"]
    else:
        payload = [
            {
                "name": declaration.name,
                "description": declaration.description,
                "parameters": declaration.parameters,
            }
            for tool in client.calls[0]["config"].tools
            for declaration in tool.function_declarations
        ]
    return json.dumps(payload, default=lambda obj: getattr(obj, "__dict__", str(obj)), sort_keys=True)


_CONTEXT_CASES = [
    (
        "attempted_only",
        {
            "attempted_models": True,
            "performance": False,
            "best_model_code": False,
            "diagnostic": False,
            "individual_differences": False,
        },
        {"list_attempted_models"},
    ),
    (
        "performance_only",
        {
            "attempted_models": False,
            "performance": True,
            "best_model_code": False,
            "diagnostic": False,
            "individual_differences": False,
        },
        {
            "list_iterations",
            "get_best_models",
            "get_bic_trajectory",
            "get_per_participant_fit",
            "list_failed_models",
        },
    ),
    (
        "best_model_code_only",
        {
            "attempted_models": False,
            "performance": False,
            "best_model_code": True,
            "diagnostic": False,
            "individual_differences": False,
        },
        {"get_best_model_code"},
    ),
    (
        "diagnostic_only",
        {
            "attempted_models": False,
            "performance": False,
            "best_model_code": False,
            "diagnostic": True,
            "individual_differences": False,
        },
        {
            "get_recovery",
            "get_ppc",
            "get_block_residuals",
            "get_parameter_distribution",
        },
    ),
    (
        "performance_diagnostic",
        {
            "attempted_models": False,
            "performance": True,
            "best_model_code": False,
            "diagnostic": True,
            "individual_differences": False,
        },
        {
            "list_iterations",
            "get_best_models",
            "get_bic_trajectory",
            "get_per_participant_fit",
            "list_failed_models",
            "compare_models",
            "get_recovery",
            "get_ppc",
            "get_block_residuals",
            "get_parameter_distribution",
        },
    ),
    (
        "full_agent",
        {
            "attempted_models": True,
            "performance": True,
            "best_model_code": True,
            "diagnostic": True,
            "individual_differences": False,
        },
        {
            "list_attempted_models",
            "list_iterations",
            "get_best_models",
            "get_bic_trajectory",
            "get_per_participant_fit",
            "list_failed_models",
            "get_model",
            "search_models",
            "get_recovery",
            "get_ppc",
            "get_block_residuals",
            "compare_models",
            "get_parameter_distribution",
        },
    ),
    (
        "individual_differences_only",
        {
            "attempted_models": False,
            "performance": False,
            "best_model_code": False,
            "diagnostic": False,
            "individual_differences": True,
        },
        {
            "get_individual_differences",
            "get_participant_best_models",
        },
    ),
    (
        "diagnostic_and_id",
        {
            "attempted_models": False,
            "performance": False,
            "best_model_code": False,
            "diagnostic": True,
            "individual_differences": True,
        },
        {
            "get_recovery",
            "get_ppc",
            "get_block_residuals",
            "get_parameter_distribution",
            "get_individual_differences",
            "get_participant_best_models",
        },
    ),
    (
        "full_agent_with_id",
        {
            "attempted_models": True,
            "performance": True,
            "best_model_code": True,
            "diagnostic": True,
            "individual_differences": True,
        },
        {
            "list_attempted_models",
            "list_iterations",
            "get_best_models",
            "get_bic_trajectory",
            "get_per_participant_fit",
            "list_failed_models",
            "get_model",
            "search_models",
            "get_recovery",
            "get_ppc",
            "get_block_residuals",
            "compare_models",
            "get_parameter_distribution",
            "get_individual_differences",
            "get_participant_best_models",
        },
    ),
]


@pytest.mark.parametrize("provider", ["openai", "gemini"])
@pytest.mark.parametrize("case_name,context,expected_names", _CONTEXT_CASES)
def test_agent_tool_schemas_are_context_filtered(provider, case_name, context, expected_names, monkeypatch):
    if provider == "openai":
        responses = [_openai_response("planning"), _openai_response("done")]
    else:
        responses = [_gemini_response("planning"), _gemini_response("done")]

    analysis, client = _run_agent_judge(provider, context, responses, monkeypatch)

    if provider == "openai":
        system_prompt = client.calls[0]["messages"][0]["content"]
        user_prompt = client.calls[0]["messages"][1]["content"]
    else:
        system_prompt = client.calls[0]["config"].system_instruction
        user_prompt = client.calls[0]["contents"][0]["parts"][0]["text"]

    if not context["performance"]:
        for forbidden in ["BIC", "trajectory", "metric", "status"]:
            assert forbidden.lower() not in system_prompt.lower()
            assert forbidden.lower() not in user_prompt.lower()
    if not context["best_model_code"]:
        assert "code" not in system_prompt.lower()
        assert "code" not in user_prompt.lower()
    if not context["diagnostic"]:
        for forbidden in ["PPC", "recovery", "residual", "r²", "r2"]:
            assert forbidden.lower() not in system_prompt.lower()
            assert forbidden.lower() not in user_prompt.lower()
    if not context["diagnostic"] and not context.get("individual_differences", False):
        for forbidden in ["individual differences"]:
            assert forbidden.lower() not in system_prompt.lower()
            assert forbidden.lower() not in user_prompt.lower()
    assert "allowed judge tools" in user_prompt.lower()

    if provider == "openai":
        captured_names = [schema["function"]["name"] for schema in client.calls[0]["tools"]]
    else:
        captured_tools = client.calls[0]["config"].tools
        captured_names = [
            declaration.name
            for tool in captured_tools
            for declaration in tool.function_declarations
        ]

    assert set(captured_names) == expected_names
    if not context["performance"] and context["diagnostic"]:
        schema_dump = _tool_schema_dump(provider, client)
        for forbidden in [
            "compare_models",
            "comparison",
            "comparative",
            "performance",
            "BIC",
            "metric",
            "ranking",
            "status",
        ]:
            assert forbidden.lower() not in schema_dump.lower()
    assert analysis["analysis_text"] == "done"


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_forbidden_tool_dispatch_returns_error(provider, monkeypatch):
    context = {
        "attempted_models": True,
        "performance": False,
        "best_model_code": False,
        "diagnostic": False,
    }
    forbidden_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="get_best_models", arguments="{}"),
    )

    if provider == "openai":
        responses = [
            _openai_response("planning"),
            _openai_response("", [forbidden_call]),
            _openai_response("forbidden handled"),
        ]
    else:
        _install_fake_gemini_modules(monkeypatch)
        responses = [
            _gemini_response("planning"),
            _gemini_response("", SimpleNamespace(name="get_best_models", args={})),
            _gemini_response("forbidden handled"),
        ]

    analysis, client = _run_agent_judge(provider, context, responses, monkeypatch)

    assert analysis["trace"][0]["tool"] == "get_best_models"
    assert "Forbidden tool" in analysis["trace"][0]["result_summary"]

    if provider == "openai":
        captured_names = [schema["function"]["name"] for schema in client.calls[0]["tools"]]
    else:
        captured_tools = client.calls[0]["config"].tools
        captured_names = [
            declaration.name
            for tool in captured_tools
            for declaration in tool.function_declarations
        ]
    assert captured_names == ["list_attempted_models"]


@pytest.mark.parametrize("provider", ["openai", "gemini"])
@pytest.mark.parametrize("forbidden_tool", ["get_individual_differences", "get_participant_best_models"])
def test_agent_loop_blocks_forbidden_individual_differences_tools(provider, forbidden_tool, monkeypatch):
    context = {
        "attempted_models": True,
        "performance": False,
        "best_model_code": False,
        "diagnostic": False,
        "individual_differences": False,
    }

    def _fail_if_called(*args, **kwargs):
        pytest.fail(f"{forbidden_tool} should not execute when forbidden")

    monkeypatch.setattr("gecco.diagnostic_store.tools.get_individual_differences", _fail_if_called)
    monkeypatch.setattr("gecco.diagnostic_store.tools.get_participant_best_models", _fail_if_called)

    if provider == "openai":
        requested_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name=forbidden_tool, arguments="{}"),
        )
        responses = [
            _openai_response("planning"),
            _openai_response("", [requested_call]),
            _openai_response("forbidden handled"),
        ]
    else:
        _install_fake_gemini_modules(monkeypatch)
        responses = [
            _gemini_response("planning"),
            _gemini_response("", SimpleNamespace(name=forbidden_tool, args={})),
            _gemini_response("forbidden handled"),
        ]

    analysis, client = _run_agent_judge(provider, context, responses, monkeypatch)

    assert analysis["trace"][0]["tool"] == forbidden_tool
    assert "Forbidden tool" in analysis["trace"][0]["result_summary"]

    calls_dump = json.dumps(client.calls, default=lambda obj: getattr(obj, "__dict__", str(obj)), sort_keys=True)
    assert forbidden_tool in calls_dump
    assert "Forbidden tool" in calls_dump


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_best_model_code_only_uses_safe_code_tool_and_blocks_model_tools(provider, monkeypatch):
    context = {
        "attempted_models": False,
        "performance": False,
        "best_model_code": True,
        "diagnostic": False,
    }
    safe_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="get_best_model_code", arguments="{}"),
    )

    if provider == "openai":
        responses = [
            _openai_response("planning"),
            _openai_response("", [safe_call]),
            _openai_response("done"),
        ]
    else:
        _install_fake_gemini_modules(monkeypatch)
        responses = [
            _gemini_response("planning"),
            _gemini_response("", SimpleNamespace(name="get_best_model_code", args={})),
            _gemini_response("done"),
        ]

    analysis, client = _run_agent_judge(provider, context, responses, monkeypatch)

    if provider == "openai":
        captured_names = [schema["function"]["name"] for schema in client.calls[0]["tools"]]
    else:
        captured_tools = client.calls[0]["config"].tools
        captured_names = [
            declaration.name
            for tool in captured_tools
            for declaration in tool.function_declarations
        ]

    assert captured_names == ["get_best_model_code"]
    schema_dump = _tool_schema_dump(provider, client)
    for forbidden in ["BIC", "metric", "status", "ranking", "performance", "comparison", "trajectory"]:
        assert forbidden.lower() not in schema_dump.lower()

    assert analysis["trace"][0]["tool"] == "get_best_model_code"

    result = dispatch_tool(
        _DiagnosticStore(),
        "get_best_model_code",
        {},
        allowed_tool_names={"get_best_model_code"},
    )
    assert "code" in result
    assert result["code"].startswith("def safe_best_model")
    for forbidden_key in ["metric", "status", "ranking", "performance", "trajectory", "bic"]:
        assert forbidden_key not in {key.lower() for key in result}

    for forbidden_tool in ["get_model", "search_models"]:
        forbidden_result = dispatch_tool(
            _DiagnosticStore(),
            forbidden_tool,
            {"model_id": 11} if forbidden_tool == "get_model" else {},
            allowed_tool_names={"get_best_model_code"},
        )
        assert forbidden_result["error"] == f"Forbidden tool: {forbidden_tool}"


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_agent_tool_schemas_include_compare_models_when_performance_and_diagnostic_enabled(provider, monkeypatch):
    context = {
        "attempted_models": False,
        "performance": True,
        "best_model_code": False,
        "diagnostic": True,
    }
    responses = [_openai_response("planning"), _openai_response("done")] if provider == "openai" else [_gemini_response("planning"), _gemini_response("done")]

    analysis, client = _run_agent_judge(provider, context, responses, monkeypatch)

    if provider == "openai":
        system_prompt = client.calls[0]["messages"][0]["content"]
        captured_names = [schema["function"]["name"] for schema in client.calls[0]["tools"]]
    else:
        system_prompt = client.calls[0]["config"].system_instruction
        captured_names = [
            declaration.name
            for tool in client.calls[0]["config"].tools
            for declaration in tool.function_declarations
        ]

    assert "compare_models" in captured_names
    assert any(
        term in system_prompt.lower()
        for term in ["comparison", "comparative", "performance", "BIC", "metric", "ranking", "status"]
    )
    assert analysis["analysis_text"] == "done"


def test_llm_prompts_do_not_leak_disabled_context(monkeypatch):
    context = {
        "attempted_models": True,
        "performance": False,
        "best_model_code": False,
        "diagnostic": False,
    }
    cfg = _make_cfg("openai", context, mode="llm")
    client = _OpenAIClient(
        [
            _openai_response("analysis text"),
            _openai_response(
                '{"per_angle": [], "key_recommendations": ["Explore broader search."], "synthesized_feedback": "done"}'
            ),
        ]
    )

    judge = ToolUsingJudge(
        cfg=cfg,
        diagnostic_store=_AgentStore(),
        model=client,
        tokenizer=None,
        results_dir=None,
    )

    analysis = judge.get_feedback_analysis(iteration=0, run_idx=0, tag="")
    assert analysis["analysis_text"] == "analysis text"
    assert len(client.calls) == 1

    first_call = client.calls[0]
    system_prompt = first_call["messages"][0]["content"]
    user_prompt = first_call["messages"][1]["content"]
    for forbidden in [
        "comparison",
        "comparative",
        "performance",
        "tool",
        "diagnostic database",
        "BIC",
        "metric",
        "trajectory",
        "stuck",
        "improved",
        "regressed",
        "plateaued",
        "ranking",
        "status",
        "code",
        "failed",
    ]:
        assert forbidden.lower() not in system_prompt.lower()
        assert forbidden.lower() not in user_prompt.lower()
    assert "Models attempted this iteration" in user_prompt
    assert "have been tried" in system_prompt.lower()

    feedback, verdict = judge.synthesize_for_persona(analysis, persona_name="default")
    assert feedback == "done"
    assert verdict["key_recommendations"] == ["Explore broader search."]
    assert len(client.calls) == 2

    synth_system_prompt = client.calls[1]["messages"][0]["content"]
    synth_user_prompt = client.calls[1]["messages"][2]["content"]
    for forbidden in [
        "comparison",
        "comparative",
        "performance",
        "tool",
        "diagnostic database",
        "BIC",
        "metric",
        "trajectory",
        "stuck",
        "improved",
        "regressed",
        "plateaued",
        "PPC",
        "recovery",
        "residual",
        "individual differences",
        "ranking",
        "status",
        "code",
        "failed",
    ]:
        assert forbidden.lower() not in synth_system_prompt.lower()
        assert forbidden.lower() not in synth_user_prompt.lower()
    assert "What to try next" in synth_user_prompt


def test_llm_prompt_ignores_stuck_search_without_performance(monkeypatch):
    context = {
        "attempted_models": True,
        "performance": False,
        "best_model_code": False,
        "diagnostic": False,
    }
    cfg = _make_cfg("openai", context, mode="llm")
    client = _OpenAIClient(
        [
            _openai_response("analysis text"),
            _openai_response(
                '{"per_angle": [], "key_recommendations": ["Explore broader search."], "synthesized_feedback": "done"}'
            ),
        ]
    )

    stuck_trajectory = [
        {"best_metric": 120.0},
        {"best_metric": 119.2},
        {"best_metric": 119.1},
    ]
    called = {"value": False}

    def _fake_get_bic_trajectory(store):
        called["value"] = True
        return stuck_trajectory

    monkeypatch.setattr("gecco.diagnostic_store.tools.get_bic_trajectory", _fake_get_bic_trajectory)

    judge = ToolUsingJudge(
        cfg=cfg,
        diagnostic_store=_AgentStore(),
        model=client,
        tokenizer=None,
        results_dir=None,
    )

    analysis = judge.get_feedback_analysis(iteration=0, run_idx=0, tag="")
    assert analysis["analysis_text"] == "analysis text"
    assert called["value"] is False

    first_call = client.calls[0]
    system_prompt = first_call["messages"][0]["content"]
    user_prompt = first_call["messages"][1]["content"]
    for forbidden in ["comparison", "comparative", "performance", "stuck", "BIC", "trajectory", "improved", "regressed", "plateaued", "ranking", "status"]:
        assert forbidden.lower() not in system_prompt.lower()
        assert forbidden.lower() not in user_prompt.lower()

    feedback, verdict = judge.synthesize_for_persona(analysis, persona_name="default")
    assert feedback == "done"
    assert verdict["key_recommendations"] == ["Explore broader search."]
    synth_system_prompt = client.calls[1]["messages"][0]["content"]
    synth_user_prompt = client.calls[1]["messages"][2]["content"]
    for forbidden in ["comparison", "comparative", "performance", "stuck", "BIC", "trajectory", "improved", "regressed", "plateaued", "ranking", "status"]:
        assert forbidden.lower() not in synth_system_prompt.lower()
        assert forbidden.lower() not in synth_user_prompt.lower()


def test_llm_previous_verdict_context_does_not_leak_performance_when_disabled(tmp_path):
    prev_dir = tmp_path / "judge"
    prev_dir.mkdir()
    (prev_dir / "iter0_run0.json").write_text(
        json.dumps(
            {
                "short_circuit": False,
                "iteration": 0,
                "key_recommendations": ["BIC improved, but the regression remains."],
                "best_bic": 12.3,
            }
        )
    )

    context = {
        "attempted_models": True,
        "performance": False,
        "best_model_code": False,
        "diagnostic": False,
    }
    cfg = _make_cfg("openai", context, mode="llm")
    client = _OpenAIClient(
        [
            _openai_response("analysis text"),
            _openai_response(
                '{"per_angle": [], "key_recommendations": ["Explore broader search."], "synthesized_feedback": "done"}'
            ),
        ]
    )

    judge = ToolUsingJudge(
        cfg=cfg,
        diagnostic_store=_AgentStore(),
        model=client,
        tokenizer=None,
        results_dir=tmp_path,
    )

    analysis = judge.get_feedback_analysis(iteration=1, run_idx=0, tag="")
    assert analysis["analysis_text"] == "analysis text"

    user_prompt = client.calls[0]["messages"][1]["content"]
    for forbidden in ["Previous recommendations", "comparison", "comparative", "performance", "BIC", "improved", "regression", "trajectory", "stuck", "metric", "ranking", "status"]:
        assert forbidden.lower() not in user_prompt.lower()


def test_static_diagnostic_context_is_deterministic(tmp_path, monkeypatch):
    cfg = _make_cfg(
        "openai",
        {
            "attempted_models": False,
            "performance": False,
            "best_model_code": False,
            "diagnostic": True,
        },
        mode="static",
    )
    judge = ToolUsingJudge(
        cfg=cfg,
        diagnostic_store=_DiagnosticStore(),
        model=object(),
        tokenizer=None,
        results_dir=tmp_path,
    )
    monkeypatch.setattr(judge, "_fallback_generate", lambda *args, **kwargs: pytest.fail("fallback should not be called"))

    analysis = judge.get_feedback_analysis(iteration=0, run_idx=0, tag="")

    text = analysis["analysis_text"]
    assert "Diagnostic context:" in text
    assert "Parameter recovery: available" in text
    assert "PPC: available" in text
    assert "Block residuals: available" in text
    for forbidden in [
        "individual differences",
        "comparison",
        "comparative",
        "performance",
        "BIC",
        "metric",
        "rank",
        "top fitted",
        "improved",
        "regressed",
        "trajectory",
        "status",
    ]:
        assert forbidden.lower() not in text.lower()


def test_static_diagnostic_context_allows_richer_performance_details(tmp_path, monkeypatch):
    cfg = _make_cfg(
        "openai",
        {
            "attempted_models": False,
            "performance": True,
            "best_model_code": False,
            "diagnostic": True,
        },
        mode="static",
    )
    judge = ToolUsingJudge(
        cfg=cfg,
        diagnostic_store=_DiagnosticStore(),
        model=object(),
        tokenizer=None,
        results_dir=tmp_path,
    )
    monkeypatch.setattr(judge, "_fallback_generate", lambda *args, **kwargs: pytest.fail("fallback should not be called"))

    analysis = judge.get_feedback_analysis(iteration=0, run_idx=0, tag="")

    text = analysis["analysis_text"]
    assert "Top fitted models:" in text
    assert "BIC=" in text
    assert "Model comparison:" in text
    assert "Parameter recovery:" in text


def test_static_diagnostic_only_excludes_individual_differences(tmp_path, monkeypatch):
    """With diagnostic: true and individual_differences: false,
    recovery/PPC/residual remain but ID/heterogeneity/regression text is absent."""
    cfg = _make_cfg(
        "openai",
        {
            "attempted_models": False,
            "performance": False,
            "best_model_code": False,
            "diagnostic": True,
            "individual_differences": False,
        },
        mode="static",
    )
    judge = ToolUsingJudge(
        cfg=cfg,
        diagnostic_store=_DiagnosticStore(),
        model=object(),
        tokenizer=None,
        results_dir=tmp_path,
    )
    monkeypatch.setattr(judge, "_fallback_generate", lambda *args, **kwargs: pytest.fail("fallback should not be called"))

    analysis = judge.get_feedback_analysis(iteration=0, run_idx=0, tag="")

    text = analysis["analysis_text"]
    assert "Diagnostic context:" in text
    assert "Parameter recovery: available" in text
    assert "PPC: available" in text
    assert "Block residuals: available" in text
    for forbidden in [
        "individual differences",
        "r²",
        "r2",
        "self-report",
        "heterogeneity",
    ]:
        assert forbidden.lower() not in text.lower()


def test_static_individual_differences_only_includes_id_evidence(tmp_path, monkeypatch):
    """With diagnostic: false and individual_differences: true,
    static output still includes ID evidence but no diagnostic evidence."""
    cfg = _make_cfg(
        "openai",
        {
            "attempted_models": False,
            "performance": False,
            "best_model_code": False,
            "diagnostic": False,
            "individual_differences": True,
        },
        mode="static",
    )
    judge = ToolUsingJudge(
        cfg=cfg,
        diagnostic_store=_DiagnosticStore(),
        model=object(),
        tokenizer=None,
        results_dir=tmp_path,
    )
    monkeypatch.setattr(judge, "_fallback_generate", lambda *args, **kwargs: pytest.fail("fallback should not be called"))

    analysis = judge.get_feedback_analysis(iteration=0, run_idx=0, tag="")

    text = analysis["analysis_text"]
    assert "Individual differences context:" in text
    assert "Individual differences: mean R²=" in text
    assert "heterogeneity=" in text
    for forbidden in ["PPC", "recovery", "residual"]:
        assert forbidden.lower() not in text.lower()


def test_static_performance_diagnostic_with_id_includes_heterogeneity(tmp_path, monkeypatch):
    """With performance+diagnostic+ID enabled, heterogeneity and R² appear."""
    cfg = _make_cfg(
        "openai",
        {
            "attempted_models": False,
            "performance": True,
            "best_model_code": False,
            "diagnostic": True,
            "individual_differences": True,
        },
        mode="static",
    )
    judge = ToolUsingJudge(
        cfg=cfg,
        diagnostic_store=_DiagnosticStore(),
        model=object(),
        tokenizer=None,
        results_dir=tmp_path,
    )
    monkeypatch.setattr(judge, "_fallback_generate", lambda *args, **kwargs: pytest.fail("fallback should not be called"))

    analysis = judge.get_feedback_analysis(iteration=0, run_idx=0, tag="")

    text = analysis["analysis_text"]
    assert "Individual differences: mean R²=" in text


def test_llm_prompt_excludes_id_when_only_diagnostic_enabled(monkeypatch):
    """LLM mode with diagnostic: true, individual_differences: false
    should not include individual differences in prompts."""
    context = {
        "attempted_models": True,
        "performance": False,
        "best_model_code": False,
        "diagnostic": True,
        "individual_differences": False,
    }
    cfg = _make_cfg("openai", context, mode="llm")
    client = _OpenAIClient(
        [
            _openai_response("analysis text"),
            _openai_response(
                '{"per_angle": [], "key_recommendations": ["Explore broader search."], "synthesized_feedback": "done"}'
            ),
        ]
    )
    judge = ToolUsingJudge(
        cfg=cfg,
        diagnostic_store=_DiagnosticStore(),
        model=client,
        tokenizer=None,
        results_dir=None,
    )
    analysis = judge.get_feedback_analysis(iteration=0, run_idx=0, tag="")
    assert analysis["analysis_text"] == "analysis text"

    first_call = client.calls[0]
    system_prompt = first_call["messages"][0]["content"]
    user_prompt = first_call["messages"][1]["content"]
    assert "Diagnostic evidence" in system_prompt
    for forbidden in ["individual differences", "r²", "r2", "self-report", "heterogeneity"]:
        assert forbidden.lower() not in system_prompt.lower()
        assert forbidden.lower() not in user_prompt.lower()


def test_llm_prompt_includes_id_when_id_enabled(monkeypatch):
    """LLM mode with individual_differences: true includes ID angle and text."""
    context = {
        "attempted_models": False,
        "performance": False,
        "best_model_code": False,
        "diagnostic": False,
        "individual_differences": True,
    }
    cfg = _make_cfg("openai", context, mode="llm")
    client = _OpenAIClient(
        [
            _openai_response("analysis text"),
            _openai_response(
                '{"per_angle": [], "key_recommendations": ["Explore broader search."], "synthesized_feedback": "done"}'
            ),
        ]
    )
    judge = ToolUsingJudge(
        cfg=cfg,
        diagnostic_store=_DiagnosticStore(),
        model=client,
        tokenizer=None,
        results_dir=None,
    )
    analysis = judge.get_feedback_analysis(iteration=0, run_idx=0, tag="")
    assert analysis["analysis_text"] == "analysis text"

    first_call = client.calls[0]
    user_prompt = first_call["messages"][1]["content"]
    assert "Individual differences" in user_prompt
    assert "heterogeneity" in user_prompt.lower()
    for forbidden in ["PPC", "recovery", "residual"]:
        assert forbidden.lower() not in user_prompt.lower()


def test_forbidden_id_tool_dispatch_uses_real_allowlist(tmp_path):
    disabled_cfg = load_config(
        str(
            _write_real_config(
                tmp_path,
                {
                    "attempted_models": True,
                    "performance": False,
                    "best_model_code": False,
                    "diagnostic": False,
                    "individual_differences": False,
                },
                mode="agent",
            )
        )
    )
    disabled_tool_names = set(get_judge_tool_names(disabled_cfg))
    assert "list_attempted_models" in disabled_tool_names
    assert "get_individual_differences" not in disabled_tool_names
    assert "get_participant_best_models" not in disabled_tool_names

    for forbidden_tool in ["get_individual_differences", "get_participant_best_models"]:
        result = dispatch_tool(
            _DiagnosticStore(),
            forbidden_tool,
            {"model_id": 11} if forbidden_tool == "get_individual_differences" else {},
            allowed_tool_names=disabled_tool_names,
        )
        assert result["error"] == f"Forbidden tool: {forbidden_tool}"

    enabled_cfg = load_config(
        str(
            _write_real_config(
                tmp_path,
                {
                    "attempted_models": True,
                    "performance": False,
                    "best_model_code": False,
                    "diagnostic": False,
                    "individual_differences": True,
                },
                mode="agent",
            )
        )
    )
    enabled_tool_names = set(get_judge_tool_names(enabled_cfg))
    assert "get_individual_differences" in enabled_tool_names
    assert "get_participant_best_models" in enabled_tool_names


def test_llm_prompt_includes_diagnostic_context_without_performance_leak(monkeypatch):
    context = {
        "attempted_models": True,
        "performance": False,
        "best_model_code": False,
        "diagnostic": True,
    }
    cfg = _make_cfg("openai", context, mode="llm")
    client = _OpenAIClient(
        [
            _openai_response("analysis text"),
            _openai_response(
                '{"per_angle": [], "key_recommendations": ["Explore broader search."], "synthesized_feedback": "done"}'
            ),
        ]
    )

    judge = ToolUsingJudge(
        cfg=cfg,
        diagnostic_store=_DiagnosticStore(),
        model=client,
        tokenizer=None,
        results_dir=None,
    )

    analysis = judge.get_feedback_analysis(iteration=0, run_idx=0, tag="")
    assert analysis["analysis_text"] == "analysis text"

    first_call = client.calls[0]
    system_prompt = first_call["messages"][0]["content"]
    user_prompt = first_call["messages"][1]["content"]
    assert "Diagnostic context:" in user_prompt
    for forbidden in [
        "comparison",
        "comparative",
        "performance",
        "BIC",
        "metric",
        "rank",
        "top fitted",
        "improved",
        "regressed",
        "trajectory",
        "status",
    ]:
        assert forbidden.lower() not in system_prompt.lower()
        assert forbidden.lower() not in user_prompt.lower()

    feedback, verdict = judge.synthesize_for_persona(analysis, persona_name="default")
    assert feedback == "done"
    assert verdict["key_recommendations"] == ["Explore broader search."]
    synth_system_prompt = client.calls[1]["messages"][0]["content"]
    synth_user_prompt = client.calls[1]["messages"][2]["content"]
    for forbidden in [
        "comparison",
        "comparative",
        "performance",
        "BIC",
        "metric",
        "rank",
        "top fitted",
        "improved",
        "regressed",
        "trajectory",
        "status",
    ]:
        assert forbidden.lower() not in synth_system_prompt.lower()
        assert forbidden.lower() not in synth_user_prompt.lower()
