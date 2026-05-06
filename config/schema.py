from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Optional
import yaml
import pandas as pd
from gecco.prepare_data.data2text import narrative
from types import SimpleNamespace


class TaskConfig(BaseModel):
    name: str
    description: str
    goal: str
    instructions: str
    extra: Optional[str] = ""


class DataConfig(BaseModel):
    path: str
    id_column: str
    input_columns: List[str]
    data2text_function: str = "narrative"
    narrative_template: Optional[str] = None


class LLMConfig(BaseModel):
    base_model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    guardrails: List[str]
    show_best_model_code: bool = False  # R1: Gate best-model code append per-client


class EvaluationConfig(BaseModel):
    metric: str = "BIC"
    fitting_method: str = "scipy_minimize"
    best_model_path: Optional[str] = None
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    split_seed: int = 42
    n_test_models: int = 10

    @model_validator(mode="after")
    def check_ratios_sum_to_one(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio ({self.train_ratio}) + val_ratio ({self.val_ratio}) "
                f"+ test_ratio ({self.test_ratio}) must sum to 1.0, got {total}"
            )
        return self


class LoopConfig(BaseModel):
    max_iterations: int
    max_independent_runs: int = 1
    n_clients: Optional[int] = None  # Number of clients for barrier synchronization


class CentralizedModelGenerationConfig(BaseModel):
    """Configuration for centralized model generation (CMG) mode.

    When enabled, one designated generator proposes all candidate models
    and a pool of numeric evaluator clients fits those candidates in parallel.
    """

    enabled: bool = False
    generator_client: str = ""
    n_models: int = 0


class ValidationConfig(BaseModel):
    retry_limit: int = 3
    max_syntax_retries: int = 2  # NEW: retries for syntax/validation failures


class BarrierConfig(BaseModel):
    orchestrator_wait_seconds: float = 1800  # Orchestrator waits for clients
    client_wait_seconds: float = 1800  # Clients wait for orchestrator
    retry_wait_seconds: float = 300  # NEW: extra time for client retries


class JudgeStuckSearchConfig(BaseModel):
    tolerance: float = (
        10.0  # R3: Δ BIC threshold over window to trigger stuck detection
    )
    window: int = 2  # R3: Number of iterations to look back


class JudgeLesionConfig(BaseModel):
    """Configuration for systematic judge lesion experiments."""

    enabled: bool = False
    lesion_type: str = "complete"
    noise_text: str = "Try different parameters and see if you can improve."


class JudgeConfig(BaseModel):
    mode: str = "manual"  # "manual" or "tool_using"
    orchestrated: bool = False  # Enable centralized judge orchestration
    barrier: Optional[BarrierConfig] = BarrierConfig()
    max_tool_calls: Optional[int] = None
    verbose: Optional[bool] = False
    stuck_search: Optional[JudgeStuckSearchConfig] = (
        JudgeStuckSearchConfig()
    )  # R3: Configurable stuck detection
    lesion: Optional[JudgeLesionConfig] = JudgeLesionConfig()


class SentryConfig(BaseModel):
    environment: str = "development"
    traces_sample_rate: float = 0.1
    profiles_sample_rate: float = 0.0
    release: Optional[str] = None


class GeCCoConfig(BaseModel):
    task: TaskConfig
    data: DataConfig
    llm: LLMConfig
    evaluation: EvaluationConfig
    loop: Optional[LoopConfig] = None
    judge: Optional[JudgeConfig] = None
    sentry: Optional[SentryConfig] = None
    validation: Optional[ValidationConfig] = None
    centralized_model_generation: Optional[CentralizedModelGenerationConfig] = None


def load_data_from_config(cfg):
    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["path"])
    text_output = narrative(
        df,
        template=data_cfg["narrative_template"],
        id_col=data_cfg.get("id_column", "participant"),
    )
    return text_output


def dict_to_namespace(d):
    """Recursively convert nested dicts into SimpleNamespace objects."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d


def load_config(path: str):
    """Load YAML config and allow dot notation access."""
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return dict_to_namespace(cfg_dict)
