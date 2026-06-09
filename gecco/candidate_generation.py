"""Candidate generation service extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from gecco.artifacts import ArtifactStore
from gecco.load_llms.provider_registry import get_provider_spec
from gecco.utils import TimestampedConsole


console = TimestampedConsole()


def _mapping_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a key from either a mapping or an attribute container."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass(slots=True)
class CandidateGenerationResult:
    """Structured result from a generation pass."""

    code_text: str
    parsed_models: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    model_file: Path


class CandidateGenerator:
    """Generate and persist candidate models for one iteration."""

    def __init__(self, artifact_store: ArtifactStore):
        self.artifact_store = artifact_store

    def generate_iteration(
        self,
        *,
        iteration: int,
        run_idx: int,
        feedback: str,
        cmg_cfg: Any,
        tag: str,
        client_id: Any,
        naive_enabled: bool,
        prompt_builder: Any,
        generate_text: Callable[..., str],
        model: Any,
        tokenizer: Any,
        cfg: Any,
        shared_registry: Any,
        participant: str | None = None,
        set_activity: Callable[[str], None] | None = None,
    ) -> CandidateGenerationResult:
        """Run one candidate-generation iteration."""

        n_models = cmg_cfg.n_models
        if set_activity is not None:
            set_activity(f"generating centralized candidates (iter {iteration})")

        clients = _mapping_get(cfg, "clients")
        client_config = _mapping_get(clients, client_id) if client_id else None

        try:
            if naive_enabled:
                code_text, parsed_models = self.generate_models_naive(
                    feedback_text=feedback,
                    n_models=n_models,
                    cfg=cfg,
                    prompt_builder=prompt_builder,
                    generate_text=generate_text,
                    model=model,
                    tokenizer=tokenizer,
                    iteration=iteration,
                    tag=tag,
                    client_config=client_config,
                )
            else:
                prompt = prompt_builder.build_input_prompt(
                    feedback_text=feedback, n_models=n_models
                )
                code_text, parsed_models = self.generate_models(
                    prompt=prompt,
                    n_models=n_models,
                    cfg=cfg,
                    generate_text=generate_text,
                    model=model,
                    tokenizer=tokenizer,
                    iteration=iteration,
                    tag=tag,
                )

            model_file = self.artifact_store.write_candidate_artifacts(
                iteration=iteration,
                run_idx=run_idx,
                tag=tag,
                code_text=code_text,
                parsed_models=parsed_models,
                participant=participant,
            )

            candidates: list[dict[str, Any]] = []
            for index, model in enumerate(parsed_models):
                func_name = f"cognitive_model{index + 1}"
                candidates.append(
                    {
                        "index": index,
                        "func_name": func_name,
                        "name": model.get("name", func_name),
                        "code": model.get("code", ""),
                        "rationale": model.get("rationale", ""),
                        "analysis": model.get("analysis", ""),
                        "parameters": model.get("parameters", []),
                        "validation_failed": model.get("validation_failed", False),
                        "validation_errors": model.get("validation_errors", []),
                    }
                )

            if len(candidates) != n_models:
                raise ValueError(
                    f"CMG generator produced {len(candidates)} candidates, expected {n_models}"
                )

            shared_registry.set_candidate_models(iteration, candidates, client_id)
            shared_registry.set_generator_status(
                iteration=iteration,
                client_id=client_id,
                status="complete",
                n_candidates=len(candidates),
            )
            return CandidateGenerationResult(
                code_text=code_text,
                parsed_models=parsed_models,
                candidates=candidates,
                model_file=model_file,
            )
        except Exception as exc:
            if shared_registry is not None:
                shared_registry.set_generator_status(
                    iteration=iteration,
                    client_id=client_id,
                    status="failed",
                    n_candidates=0,
                    error=str(exc),
                )
            raise

    def generate_non_cmg_iteration(
        self,
        *,
        iteration: int,
        run_idx: int,
        feedback: str,
        n_models: int,
        cfg: Any,
        tag: str,
        prompt_builder: Any,
        generate_text: Callable[..., str],
        model: Any,
        tokenizer: Any,
        participant: str | None = None,
        force_include_feedback: bool = False,
        client_id: Any | None = None,
    ) -> CandidateGenerationResult:
        """Generate and persist one non-CMG candidate batch."""

        clients = _mapping_get(cfg, "clients")
        client_config = _mapping_get(clients, client_id) if client_id else None

        naive_cfg = _mapping_get(client_config, "naive_ideation")

        if naive_cfg and _mapping_get(naive_cfg, "enabled", False):
            code_text, parsed_models = self.generate_models_naive(
                feedback_text=feedback,
                n_models=n_models,
                cfg=cfg,
                prompt_builder=prompt_builder,
                generate_text=generate_text,
                model=model,
                tokenizer=tokenizer,
                iteration=iteration,
                tag=tag,
                client_config=client_config,
                force_include_feedback=force_include_feedback,
            )
        else:
            prompt = prompt_builder.build_input_prompt(
                feedback_text=feedback,
                n_models=n_models,
                force_include_feedback=force_include_feedback,
            )
            code_text, parsed_models = self.generate_models(
                prompt=prompt,
                n_models=n_models,
                cfg=cfg,
                generate_text=generate_text,
                model=model,
                tokenizer=tokenizer,
                iteration=iteration,
                tag=tag,
            )

        model_file = self.artifact_store.write_candidate_artifacts(
            iteration=iteration,
            run_idx=run_idx,
            tag=tag,
            code_text=code_text,
            parsed_models=parsed_models,
            participant=participant,
        )

        candidates: list[dict[str, Any]] = []
        for index, model_dict in enumerate(parsed_models):
            func_name = f"cognitive_model{index + 1}"
            candidates.append(
                {
                    "index": index,
                    "func_name": func_name,
                    "name": model_dict.get("name", func_name),
                    "code": model_dict.get("code", ""),
                    "rationale": model_dict.get("rationale", ""),
                    "analysis": model_dict.get("analysis", ""),
                    "parameters": model_dict.get("parameters", []),
                    "validation_failed": model_dict.get("validation_failed", False),
                    "validation_errors": model_dict.get("validation_errors", []),
                }
            )

        if len(candidates) != n_models:
            raise ValueError(
                f"Generator produced {len(candidates)} candidates, expected {n_models}"
            )

        return CandidateGenerationResult(
            code_text=code_text,
            parsed_models=parsed_models,
            candidates=candidates,
            model_file=model_file,
        )

    def generate_models(
        self,
        *,
        prompt: str,
        n_models: int,
        cfg: Any,
        generate_text: Callable[..., str],
        model: Any,
        tokenizer: Any,
        iteration: int | None = None,
        tag: str = "",
    ) -> tuple[str, list[dict[str, Any]]]:
        """Generate structured candidate models from an explicit prompt.

        Args:
            prompt: Prompt text to send to the generation backend.
            n_models: Number of candidate models to request.
            cfg: Runtime configuration.
            generate_text: Low-level text-generation callable.
            model: Back-end model object.
            tokenizer: Tokeniser for the back-end model.
            iteration: Optional iteration index for review persistence.
            tag: Optional file tag used for review persistence.

        Returns:
            A ``(raw_text, models)`` pair.
        """

        from gecco.structured_output import (
            build_correction_prompt,
            build_fix_prompt,
            build_review_prompt,
            get_model_schema,
            get_review_schema,
            get_schema_instructions,
            parse_model_response,
            parse_review_response,
            validate_single_model,
        )

        structured = getattr(cfg.llm, "structured_output", True)
        validation_cfg = getattr(cfg, "validation", None)
        max_retries = (
            getattr(validation_cfg, "retry_limit", 3)
            if validation_cfg is not None
            else 3
        )

        include_analysis = getattr(cfg.llm, "analysis_scratchpad", True)
        model_schema = get_model_schema(n_models, include_analysis=include_analysis)

        raw_text = generate_text(
            model,
            tokenizer,
            prompt,
            response_schema=model_schema if structured else None,
        )
        models, _ = parse_model_response(raw_text, n_models, structured_output=structured)

        if not models:
            console.print("[yellow]No models extracted from LLM response[/]")
            return raw_text, []

        for model_dict in models:
            if model_dict.get("analysis"):
                console.print(
                    f"  [dim]{model_dict['name']} analysis:[/] "
                    f"{model_dict['analysis'][:200]}"
                    f"{'...' if len(model_dict['analysis']) > 200 else ''}"
                )

        validated_models: list[dict[str, Any]] = []
        for index, model_dict in enumerate(models):
            validated_model = model_dict
            model_name = model_dict.get("name", f"cognitive_model{index + 1}")
            validation_result = None

            for retry_attempt in range(max_retries):
                validation_result = validate_single_model(validated_model)

                if validation_result.is_valid:
                    console.print(
                        f"  [dim]Model {index + 1} ({model_name}) passed validation[/]"
                    )
                    break

                error_trace = "\n".join(f"  {err}" for err in validation_result.errors)
                console.print(
                    f"  [yellow]Model {index + 1} ({model_name}) failed validation "
                    f"(attempt {retry_attempt + 1}/{max_retries}):[/]\n{error_trace}"
                )

                schema_instructions = get_schema_instructions(1, include_analysis=False)
                correction_prompt = build_correction_prompt(
                    model=validated_model,
                    model_index=index + 1,
                    validation_errors=validation_result.errors,
                    schema_instructions=schema_instructions,
                )

                correction_schema = get_model_schema(1, include_analysis=False)
                correction_text = generate_text(
                    model,
                    tokenizer,
                    correction_prompt,
                    response_schema=correction_schema if structured else None,
                )
                corrected, _ = parse_model_response(
                    correction_text, 1, structured_output=structured
                )

                if corrected:
                    validated_model = corrected[0]
                else:
                    console.print("  [yellow]Failed to parse correction attempt[/]")
                    break

            if validation_result is not None and validation_result.is_valid:
                validated_models.append(validated_model)
            else:
                console.print(
                    f"  [bold red]Model {index + 1} ({model_name}) failed validation "
                    f"after {max_retries} retries — skipping[/]"
                )
                validated_models.append(
                    {
                        "name": model_name,
                        "rationale": model_dict.get("rationale", ""),
                        "code": model_dict.get("code", ""),
                        "analysis": model_dict.get("analysis", ""),
                        "validation_failed": True,
                        "validation_errors": validation_result.errors if validation_result else [],
                    }
                )

        models = validated_models
        if not models:
            console.print("[yellow]All models failed validation — no models to process[/]")
            return raw_text, []

        reviewer_config = getattr(cfg.llm, "reviewer", None)
        if reviewer_config and getattr(reviewer_config, "enabled", False) and models:
            guardrails = getattr(cfg.llm, "guardrails", [])
            persona = getattr(reviewer_config, "persona", None)
            focus_areas = getattr(reviewer_config, "focus_areas", None)

            console.print("[dim]Running code review...[/]")
            review_prompt = build_review_prompt(
                models,
                guardrails=guardrails,
                persona=persona,
                focus_areas=focus_areas,
            )
            review_schema = get_review_schema()
            review_text = generate_text(
                model,
                tokenizer,
                review_prompt,
                response_schema=review_schema if structured else None,
            )
            review = parse_review_response(review_text)

            if iteration is not None:
                self.artifact_store.write_review(review, iteration=iteration, tag=tag)

            total_issues = sum(len(r.get("issues", [])) for r in review.get("reviews", []))

            if total_issues > 0:
                console.print(
                    f"[dim]Review found {total_issues} issue(s) across "
                    f"{sum(1 for r in review.get('reviews', []) if r.get('issues'))} model(s)[/]"
                )

                fix_prompt = build_fix_prompt(models, review, guardrails=guardrails)
                if fix_prompt:
                    console.print("[dim]Requesting fixes...[/]")
                    fix_text = generate_text(
                        model,
                        tokenizer,
                        fix_prompt,
                        response_schema=model_schema if structured else None,
                    )
                    fixed_models, _ = parse_model_response(
                        fix_text, n_models, structured_output=structured
                    )

                    if fixed_models and len(fixed_models) == len(models):
                        for original, fixed in zip(models, fixed_models):
                            original["code"] = fixed["code"]
                            if fixed.get("rationale"):
                                original["rationale"] = fixed["rationale"]
                        console.print(f"[dim]Applied fixes to {len(models)} model(s)[/]")
                    else:
                        console.print("[yellow]Fix parsing failed — using original models[/]")
            else:
                non_passing = sum(
                    1
                    for r in review.get("reviews", [])
                    if r.get("overall_assessment", "passes") != "passes"
                )
                if non_passing == 0:
                    console.print("[dim]Review passed — no issues found[/]")
                else:
                    console.print(
                        f"[dim]Review found {non_passing} model(s) with issues (no fix attempted)[/]"
                    )

        return raw_text, models

    def generate_models_naive(
        self,
        *,
        feedback_text: str,
        n_models: int,
        cfg: Any,
        prompt_builder: Any,
        generate_text: Callable[..., str],
        model: Any,
        tokenizer: Any,
        iteration: int | None = None,
        tag: str = "",
        force_include_feedback: bool = False,
        client_config: Any | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Run the two-phase naive ideation generation path.

        Args:
            feedback_text: Current feedback to condition generation.
            n_models: Number of models to generate.
            cfg: Runtime configuration.
            prompt_builder: Prompt builder with the standard and naive prompts.
            generate_text: Low-level text-generation callable.
            model: Back-end model object.
            tokenizer: Tokeniser for the back-end model.
            iteration: Optional iteration index for review persistence.
            tag: Optional file tag used for review persistence.
            force_include_feedback: Whether the prompt must include feedback.

        Returns:
            A ``(raw_text, models)`` pair.
        """

        naive_cfg = _mapping_get(client_config, "naive_ideation") if client_config else None

        if not naive_cfg or not _mapping_get(naive_cfg, "enabled", False):
            prompt = prompt_builder.build_input_prompt(
                feedback_text=feedback_text,
                n_models=n_models,
                force_include_feedback=force_include_feedback,
            )
            return self.generate_models(
                prompt=prompt,
                n_models=n_models,
                cfg=cfg,
                generate_text=generate_text,
                model=model,
                tokenizer=tokenizer,
                iteration=iteration,
                tag=tag,
            )

        persona = _mapping_get(naive_cfg, "persona")
        translation_preamble = _mapping_get(naive_cfg, "translation_preamble")

        if not get_provider_spec(cfg.llm.provider).supports_system_prompt:
            console.print(
                "[yellow]Warning: HuggingFace backend does not support system prompts. "
                "Phase 1 persona will have no effect.[/]"
            )

        console.print("  [dim]Phase 1: Naive psychological ideation...[/]")
        naive_prompt = prompt_builder.build_naive_prompt(feedback_text)

        naive_idea = generate_text(
            model,
            tokenizer,
            naive_prompt,
            response_schema=None,
            system_prompt=persona,
        )

        if not naive_idea:
            console.print(
                "  [yellow]Phase 1 ideation failed to return an idea — using empty idea.[/]"
            )
            naive_idea = ""
        else:
            console.print(f"  [dim]Naive hypothesis:[/] {naive_idea[:200]}...")

        console.print("  [dim]Phase 2: Computational translation...[/]")
        prompt = prompt_builder.build_input_prompt(
            feedback_text=feedback_text,
            naive_idea=naive_idea,
            translation_preamble=translation_preamble,
            n_models=n_models,
            force_include_feedback=force_include_feedback,
        )

        return self.generate_models(
            prompt=prompt,
            n_models=n_models,
            cfg=cfg,
            generate_text=generate_text,
            model=model,
            tokenizer=tokenizer,
            iteration=iteration,
            tag=tag,
        )
