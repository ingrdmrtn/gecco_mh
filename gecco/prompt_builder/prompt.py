import numpy as np
from typing import Optional


def build_prompt(
    cfg,
    data_text,
    data,
    feedback_text=None,
    naive_idea: Optional[str] = None,
    translation_preamble: Optional[str] = None,
    n_models: Optional[int] = None,
    force_include_feedback: bool = False,
):
    """
    Construct the structured LLM prompt for cognitive model generation.
    Order:
    1. Task description
    2. Example participant data
    3. Modeling goal and instructions
    4. Guardrails
    5. Template model
    """
    task, llm, evaluation, metadata = (
        cfg.task,
        cfg.llm,
        cfg.evaluation,
        getattr(cfg, "metadata", None),
    )
    guardrails = getattr(llm, "guardrails", [])
    include_feedback = getattr(llm, "include_feedback", False)
    fit_type = getattr(evaluation, "fit_type", "group")
    metadata = getattr(metadata, "flag", False)
    abstract_base_model = getattr(llm, "abstract_base_model", None)
    diversity_requirement = getattr(llm, "diversity_requirement", None)

    # Format goal section dynamically
    model_count = n_models if n_models is not None else llm.models_per_iteration
    names = [f"`cognitive_model{i + 1}`" for i in range(model_count)]
    goal_text = task.goal.format(
        models_per_iteration=model_count,
        model_names=", ".join(names),
    )

    if fit_type == "individual":
        individual_variability_feature = cfg.individual_difference.individual_feature
        if individual_variability_feature == "None":
            individual_variability_section = ""
        else:
            individual_variability_feature = np.array(
                data[individual_variability_feature]
            )[0]
            individual_variability_section = (
                cfg.individual_difference.description.format(
                    individual_feature=individual_variability_feature
                )
            )
    else:
        individual_variability_section = ""

    # Individual differences evaluation context (if configured)
    id_eval_section = ""
    if hasattr(cfg, "individual_differences_eval"):
        id_cfg = cfg.individual_differences_eval
        predictor_names = ", ".join(id_cfg.predictors)
        id_eval_section = (
            "### Individual Differences Evaluation\n"
            "In addition to fitting the behavioural data well (minimising BIC), "
            "your models will also be evaluated on how well their fitted parameters "
            "explain individual differences in questionnaire measures.\n"
            f"The questionnaire measures are: {predictor_names}.\n"
            "For each model parameter, we run a regression predicting "
            "the parameter values from these questionnaire scores across participants. "
            "Even one model parameter that strongly predicts a symptom measure is valuable. "
            "You don't need high average R² across all parameters — focus on designing "
            "parameters that could have clear, interpretable links to specific symptom "
            "dimensions. One strong relationship matters more than many weak ones.\n"
            "Design your model parameters to capture psychologically meaningful "
            "individual variation that could relate to these measures."
        )

    feedback_enabled = include_feedback or force_include_feedback
    feedback_section = (
        f"\n\n### Feedback from previous iterations\n{feedback_text.strip()}"
        if (feedback_text and feedback_enabled)
        else ""
    )

    # Structured output instructions (appended to prompt when enabled)
    structured_output_section = ""
    if getattr(llm, "structured_output", True):
        from gecco.structured_output import get_schema_instructions

        include_analysis = getattr(llm, "analysis_scratchpad", True)
        structured_output_section = get_schema_instructions(
            model_count,
            include_analysis=include_analysis,
        )

    metadata_section = (
        f"### Metadata\n{cfg.metadata.description.strip()}" if metadata else ""
    )

    introduce_data = (
        """Here is the participant data: """
        if fit_type == "individual"
        else """Here is the data from several participants: """
    )

    # Naive ideation preamble
    naive_preamble_section = ""
    if naive_idea:
        if not translation_preamble:
            # Fallback default — used only when no translation_preamble is set in the config.
            # If editing this, also update naive_ideation.translation_preamble in the YAML
            # config(s) to keep them in sync.
            translation_preamble = (
                "A psychologist — without any knowledge of computational modeling or "
                "reinforcement learning — observed participant behavior and proposed "
                "the following psychological explanation:\n\n"
                "---\n{naive_idea}\n---\n\n"
                "Your task is to implement this idea faithfully as a cognitive model. "
                "Important:\n"
                "- Treat the above as the authoritative specification. Do NOT "
                "reinterpret it through existing RL or computational frameworks "
                "unless they genuinely capture the idea.\n"
                "- If the idea does not map cleanly onto standard Q-learning, Bayesian "
                "inference, or any other known formalism, invent a new mathematical "
                "form that directly expresses the psychological mechanism described.\n"
                "- The goal is a direct computational implementation of this specific "
                "theory, not a standard model that vaguely resembles it."
            )
        naive_preamble_section = (
            f"### Psychological Hypothesis\n"
            f"{translation_preamble.replace('{naive_idea}', naive_idea)}\n\n"
        )

    if cfg.llm.provider in ["openai", "claude", "gemini", "kcl"]:
        # --- prompt layout for closed models ---

        # {metadata_section}

        prompt = f"""
### Task Description
{task.name}
{task.description.strip()}

{"### Individual Variability" if individual_variability_section else ""}
{individual_variability_section}


### Participant Data
{introduce_data}
{data_text.strip()}

{naive_preamble_section}
{"### Base Class (DO NOT MODIFY) " + abstract_base_model.strip() if abstract_base_model else ""}

### Template Model
{llm.template_model.strip()}

### Guardrails
{chr(10).join(guardrails)}

{f"### Diversity Requirement\n{chr(10).join(diversity_requirement)}" if diversity_requirement else ""}

{"" if not id_eval_section else id_eval_section}

### Your Task
{goal_text.strip()}

{structured_output_section}

{feedback_section}
""".strip()

    # --- prompt layout for open models ---
    else:
        prompt = f"""

{task.description.strip()}

{introduce_data}
{data_text.strip()}

{naive_preamble_section}### Your Task
{goal_text.strip()}

### Implementation Guidelines
{chr(10).join(guardrails)}

### Initial Model Template
{llm.template_model.strip()}

{"" if not id_eval_section else id_eval_section}

{structured_output_section}

{feedback_section}

Your function:
        """.strip()

    return prompt


class PromptBuilderWrapper:
    def __init__(self, cfg, data_text, data):
        self.cfg = cfg
        self._data_text = data_text
        self.data = data

    def build_naive_prompt(self, feedback_text: Optional[str] = "") -> str:
        """
        Construct a minimal prompt for naive psychological ideation.
        Includes task description and plain-language feedback, but NO
        code or modeling formalisms.
        """
        task = self.cfg.task
        feedback_text = feedback_text or ""

        # Take first 5 trials from data_text as a short narrative excerpt
        data_excerpt = "\n".join(self._data_text.strip().split("\n")[:5])

        prompt = f"""
### Task Description
{task.description.strip()}

### Example Participant Behavior
{data_excerpt}

### Observations and Feedback
{feedback_text.strip()}

### Your Task
Based on the task description and the observations above, propose a clear 
psychological hypothesis about how people behave in this task. 
Describe the hypothesis in plain language only, focusing on processes 
like habits, attention, emotion, memory, motivation, fatigue, or 
frustration. 

DO NOT propose equations, code, or computational formalisms. 
Just describe your theory of human behavior.
""".strip()
        return prompt

    def build_input_prompt(
        self,
        feedback_text: str = "",
        naive_idea: Optional[str] = None,
        translation_preamble: Optional[str] = None,
        n_models: Optional[int] = None,
        force_include_feedback: bool = False,
    ):
        return build_prompt(
            self.cfg,
            self._data_text,
            self.data,
            feedback_text=feedback_text,
            naive_idea=naive_idea,
            translation_preamble=translation_preamble,
            n_models=n_models,
            force_include_feedback=force_include_feedback,
        )
