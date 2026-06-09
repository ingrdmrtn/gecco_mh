"""Validated runtime configuration schema for GeCCo."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from gecco.load_llms.provider_registry import get_provider_spec
from gecco.prepare_data.data2text import narrative


JudgeCapability = Literal[
    "attempted_models_overview",
    "performance_summary",
    "best_model_code",
    "recommendations",
    "mechanistic_coherence",
    "tools",
    "citations",
    "persona_synthesis",
    "coverage",
    "random_feedback",
]


class GeCCoBaseModel(BaseModel):
    """Base model that preserves unknown config keys unless overridden."""

    model_config = ConfigDict(extra="allow")


class TaskConfig(GeCCoBaseModel):
    """Task-level runtime configuration."""

    name: str
    description: str
    goal: str
    instructions: str = ""
    extra: str = ""


class DataConfig(GeCCoBaseModel):
    """Input data configuration."""

    path: str
    id_column: str
    input_columns: list[str]
    data2text_function: str = "narrative"
    narrative_template: str | None = None


class LLMConfig(GeCCoBaseModel):
    """LLM runtime configuration."""

    provider: str
    base_model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    guardrails: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_provider_key(self) -> "LLMConfig":
        """Ensure the provider matches one exact registered registry key."""
        get_provider_spec(self.provider)
        return self


class EvaluationConfig(GeCCoBaseModel):
    """Model evaluation configuration."""

    metric: str = "BIC"
    fit_type: str = "group"
    fitting_method: str = "scipy_minimize"
    best_model_path: str | None = None
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    split_seed: int = 42
    n_test_models: int = 10
    n_starts: int = 10

    @model_validator(mode="after")
    def check_ratios_sum_to_one(self) -> "EvaluationConfig":
        """Ensure train/validation/test ratios form a full partition."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio ({self.train_ratio}) + val_ratio ({self.val_ratio}) "
                f"+ test_ratio ({self.test_ratio}) must sum to 1.0, got {total}"
            )
        return self


class LoopConfig(GeCCoBaseModel):
    """Top-level search loop configuration."""

    max_iterations: int
    max_independent_runs: int = 1
    n_clients: int | None = None


class CentralizedModelGenerationConfig(GeCCoBaseModel):
    """Configuration for centralized model generation mode."""

    enabled: bool = False
    generator_client: str = ""
    n_models: int = 0
    run_final_evaluation: bool = True


class ValidationConfig(GeCCoBaseModel):
    """Validation and retry controls."""

    retry_limit: int = 3
    max_syntax_retries: int = 2


class BarrierConfig(GeCCoBaseModel):
    """Barrier timing settings for distributed orchestration."""

    orchestrator_wait_seconds: float = 1800
    client_wait_seconds: float = 1800
    retry_wait_seconds: float = 300


class JudgeStuckSearchConfig(GeCCoBaseModel):
    """Thresholds for judge stuck-search detection."""

    tolerance: float = 10.0
    window: int = 2


class JudgeDiagnosticStoreConfig(GeCCoBaseModel):
    """Diagnostic store settings for judge analysis."""

    enabled: bool = False


class JudgePPCConfig(GeCCoBaseModel):
    """Posterior predictive check settings for the judge."""

    enabled: bool = False
    n_sims: int = 100


class JudgeBlockResidualsConfig(GeCCoBaseModel):
    """Residual-analysis settings for the judge."""

    enabled: bool = False


class JudgeProfileConfig(GeCCoBaseModel):
    """Persona profile overrides used during synthesis."""


class JudgeConfig(GeCCoBaseModel):
    """Judge runtime configuration."""

    barrier: BarrierConfig = Field(default_factory=BarrierConfig)
    max_tool_calls: int | None = None
    verbose: bool = False
    stuck_search: JudgeStuckSearchConfig = Field(default_factory=JudgeStuckSearchConfig)
    capabilities: list[JudgeCapability] = Field(default_factory=list)
    diagnostic_store: JudgeDiagnosticStoreConfig = Field(
        default_factory=JudgeDiagnosticStoreConfig
    )
    ppc: JudgePPCConfig = Field(default_factory=JudgePPCConfig)
    block_residuals: JudgeBlockResidualsConfig = Field(
        default_factory=JudgeBlockResidualsConfig
    )
    persona_profiles: dict[str, JudgeProfileConfig] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def reject_retired_mode_field(cls, data: Any) -> Any:
        """Reject retired judge runtime fields with a clear message."""
        if isinstance(data, dict) and "mode" in data:
            raise ValueError(
                "judge.mode has been retired; remove the field. "
                "The judge now always uses the orchestrated pipeline."
            )
        if isinstance(data, dict) and "orchestrated" in data:
            raise ValueError(
                "judge.orchestrated has been retired; remove the field. "
                "Orchestrator launch is now inferred from the validated judge configuration."
            )
        return data

    @model_validator(mode="after")
    def validate_capabilities(self) -> "JudgeConfig":
        """Reject invalid or duplicated capability combinations."""
        if len(self.capabilities) != len(set(self.capabilities)):
            raise ValueError("judge.capabilities must not contain duplicates")
        if "random_feedback" in self.capabilities and len(self.capabilities) != 1:
            raise ValueError(
                "judge.capabilities must use random_feedback as an exclusive mode"
            )
        return self


class SentryConfig(GeCCoBaseModel):
    """Sentry instrumentation settings."""

    environment: str = "development"
    traces_sample_rate: float = 0.1
    profiles_sample_rate: float = 0.0
    release: str | None = None


class GeCCoConfig(GeCCoBaseModel):
    """Full validated GeCCo runtime configuration."""

    task: TaskConfig
    data: DataConfig
    llm: LLMConfig
    evaluation: EvaluationConfig
    loop: LoopConfig | None = None
    judge: JudgeConfig | None = None
    sentry: SentryConfig | None = None
    validation: ValidationConfig | None = None
    centralized_model_generation: CentralizedModelGenerationConfig | None = None
    clients: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_judge_dependencies(self) -> "GeCCoConfig":
        """Validate cross-section judge capability requirements."""
        if self.judge is None:
            return self

        capabilities = set(self.judge.capabilities)
        if "persona_synthesis" in capabilities and not self._has_persona_configuration():
            raise ValueError(
                "judge.capabilities includes persona_synthesis but no multiple "
                "configured personas or explicit judge.persona_profiles were provided"
            )
        return self

    def _has_persona_configuration(self) -> bool:
        """Return whether the config defines enough persona synthesis context."""
        if self.judge is not None and len(self.judge.persona_profiles) >= 2:
            return True

        persona_count = 0
        for client_cfg in self.clients.values():
            llm_cfg = _get_mapping_value(client_cfg, "llm")
            if llm_cfg is None:
                continue
            if _get_mapping_value(llm_cfg, "persona") or _get_mapping_value(
                llm_cfg, "feedback_guidance"
            ):
                persona_count += 1
        return persona_count >= 2


def _get_mapping_value(obj: Any, key: str) -> Any:
    """Read a key from a dict-like or attribute-bearing config object."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def get_judge_capabilities(cfg_or_judge: Any) -> list[str]:
    """Return the configured judge capabilities as a stable list."""
    judge_cfg = getattr(cfg_or_judge, "judge", cfg_or_judge)
    capabilities = getattr(judge_cfg, "capabilities", []) if judge_cfg else []
    return list(capabilities or [])


def judge_has_capability(cfg_or_judge: Any, capability: str) -> bool:
    """Return whether the named judge capability is enabled."""
    return capability in set(get_judge_capabilities(cfg_or_judge))


def load_data_from_config(cfg: GeCCoConfig | dict[str, Any]) -> Any:
    """Load and narrativise data described by the runtime config."""
    data_cfg = cfg["data"] if isinstance(cfg, dict) else cfg.data
    df = pd.read_csv(data_cfg["path"] if isinstance(data_cfg, dict) else data_cfg.path)
    template = (
        data_cfg.get("narrative_template")
        if isinstance(data_cfg, dict)
        else data_cfg.narrative_template
    )
    id_column = (
        data_cfg.get("id_column", "participant")
        if isinstance(data_cfg, dict)
        else data_cfg.id_column
    )
    return narrative(df, template=template, id_col=id_column)


def load_config(path: str | Path) -> GeCCoConfig:
    """Load and validate a YAML config via the Pydantic schema."""
    with Path(path).open("r", encoding="utf-8") as file_obj:
        cfg_dict = yaml.safe_load(file_obj) or {}

    try:
        cfg = GeCCoConfig.model_validate(cfg_dict)
        return cfg
    except ValidationError as exc:
        raise exc
