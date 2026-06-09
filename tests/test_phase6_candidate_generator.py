"""Direct contract tests for the candidate generation service."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gecco.artifacts import ArtifactStore
from gecco.candidate_generation import CandidateGenerator
from gecco.run_context import RunContext


def test_candidate_generator_generates_and_persists_models_without_monolith(tmp_path: Path):
    """The generator should work from explicit collaborators and persist artefacts."""

    cfg = SimpleNamespace(
        llm=SimpleNamespace(
            structured_output=True,
            analysis_scratchpad=True,
            reviewer=SimpleNamespace(enabled=False),
            provider="openai",
        ),
        validation=SimpleNamespace(retry_limit=1),
        task=SimpleNamespace(name="phase6_task"),
        evaluation=SimpleNamespace(fit_type="group"),
    )
    run_context = RunContext.from_cfg(cfg, project_root=tmp_path)
    artifact_store = ArtifactStore(run_context)
    generator = CandidateGenerator(artifact_store)

    response = json.dumps(
        {
            "models": [
                {
                    "name": "model_a",
                    "rationale": "A compact model with one parameter.",
                    "parameters": [
                        {"name": "alpha", "lower_bound": 0, "upper_bound": 1}
                    ],
                    "code": "@njit\ndef cognitive_model1(x, model_parameters):\n    alpha, = model_parameters\n    return alpha",
                    "analysis": "simple",
                }
            ]
        }
    )

    with patch("gecco.structured_output.validate_single_model") as validate_single_model:
        validate_single_model.return_value = SimpleNamespace(is_valid=True, errors=[], spec=None)
        raw_text, models = generator.generate_models(
            prompt="build one model",
            n_models=1,
            cfg=cfg,
            generate_text=MagicMock(return_value=response),
            model=object(),
            tokenizer=object(),
        )

    model_file = artifact_store.write_candidate_artifacts(
        iteration=0,
        run_idx=1,
        tag="",
        code_text=raw_text,
        parsed_models=models,
    )

    assert models[0]["name"] == "model_a"
    assert model_file.exists()
    assert not model_file.with_suffix(".json").exists()
    run_context.close()


def test_candidate_generator_persists_review_through_artifact_store(tmp_path: Path):
    """Review persistence should stay inside the service and artefact boundary."""

    cfg = SimpleNamespace(
        llm=SimpleNamespace(
            structured_output=True,
            analysis_scratchpad=True,
            reviewer=SimpleNamespace(enabled=True),
            provider="openai",
            guardrails=[],
        ),
        validation=SimpleNamespace(retry_limit=1),
        task=SimpleNamespace(name="phase6_review_task"),
        evaluation=SimpleNamespace(fit_type="group"),
    )
    run_context = RunContext.from_cfg(cfg, project_root=tmp_path)
    artifact_store = ArtifactStore(run_context, inspection_output_enabled=True)
    generator = CandidateGenerator(artifact_store)

    generation_response = json.dumps(
        {
            "models": [
                {
                    "name": "model_a",
                    "rationale": "Initial rationale.",
                    "parameters": [],
                    "code": "def cognitive_model1(x, model_parameters):\n    return 0.0",
                    "analysis": "analysis",
                }
            ]
        }
    )
    review_response = json.dumps(
        {
            "reviews": [
                {
                    "model_name": "model_a",
                    "overall_assessment": "needs_changes",
                    "issues": [
                        {"severity": "medium", "description": "Return a float explicitly."}
                    ],
                }
            ]
        }
    )
    fix_response = json.dumps(
        {
            "models": [
                {
                    "name": "model_a",
                    "rationale": "Updated rationale.",
                    "parameters": [],
                    "code": "def cognitive_model1(x, model_parameters):\n    return float(0.0)",
                    "analysis": "analysis",
                }
            ]
        }
    )

    with patch("gecco.structured_output.validate_single_model") as validate_single_model:
        validate_single_model.return_value = SimpleNamespace(is_valid=True, errors=[], spec=None)
        result = generator.generate_non_cmg_iteration(
            iteration=0,
            run_idx=1,
            feedback="build one model",
            n_models=1,
            cfg=cfg,
            tag="",
            prompt_builder=SimpleNamespace(
                build_input_prompt=MagicMock(return_value="build one model")
            ),
            generate_text=MagicMock(
                side_effect=[generation_response, review_response, fix_response]
            ),
            model=object(),
            tokenizer=object(),
        )

    assert result.code_text == generation_response
    assert "return float(0.0)" in result.parsed_models[0]["code"]
    review_file = run_context.results_dir / "reviews" / "iter0.json"
    assert review_file.exists()
    assert json.loads(review_file.read_text(encoding="utf-8"))["reviews"][0]["model_name"] == "model_a"
    run_context.close()


def test_candidate_generator_review_files_use_explicit_iteration_and_tag(tmp_path: Path):
    """Review file names should depend only on iteration and tag."""

    cfg = SimpleNamespace(
        llm=SimpleNamespace(
            structured_output=True,
            analysis_scratchpad=True,
            reviewer=SimpleNamespace(enabled=True),
            provider="openai",
            guardrails=[],
        ),
        validation=SimpleNamespace(retry_limit=1),
        task=SimpleNamespace(name="phase6_review_task"),
        evaluation=SimpleNamespace(fit_type="group"),
    )
    run_context = RunContext.from_cfg(cfg, project_root=tmp_path)
    artifact_store = ArtifactStore(run_context, inspection_output_enabled=True)
    generator = CandidateGenerator(artifact_store)

    reviews_dir = run_context.results_dir / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    (reviews_dir / "iter1.json").write_text("{}", encoding="utf-8")

    generation_response = json.dumps(
        {
            "models": [
                {
                    "name": "model_a",
                    "rationale": "Initial rationale.",
                    "parameters": [],
                    "code": "def cognitive_model1(x, model_parameters):\n    return 0.0",
                    "analysis": "analysis",
                }
            ]
        }
    )
    review_response = json.dumps(
        {
            "reviews": [
                {
                    "model_name": "model_a",
                    "overall_assessment": "needs_changes",
                    "issues": [],
                }
            ]
        }
    )

    with patch("gecco.structured_output.validate_single_model") as validate_single_model:
        validate_single_model.return_value = SimpleNamespace(is_valid=True, errors=[], spec=None)
        generator.generate_non_cmg_iteration(
            iteration=7,
            run_idx=1,
            feedback="build one model",
            n_models=1,
            cfg=cfg,
            tag="_abc",
            prompt_builder=SimpleNamespace(
                build_input_prompt=MagicMock(return_value="build one model")
            ),
            generate_text=MagicMock(side_effect=[generation_response, review_response]),
            model=object(),
            tokenizer=object(),
        )

    review_file = run_context.results_dir / "reviews" / "iter7_abc.json"
    assert review_file.exists()
    assert not (run_context.results_dir / "reviews" / "iter2_abc.json").exists()
    run_context.close()
