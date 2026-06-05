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
    assert model_file.with_suffix(".json").exists()
    run_context.close()
