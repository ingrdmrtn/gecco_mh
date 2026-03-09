import numpy as np
def build_prompt(cfg, data_text, data, feedback_text=None):
    """
    Construct the structured LLM prompt for cognitive model generation.
    Order:
    1. Task description
    2. Example participant data
    3. Modeling goal and instructions
    4. Guardrails
    5. Template model
    """
    task, llm, evaluation, metadata = cfg.task, cfg.llm, cfg.evaluation, getattr(cfg, "metadata", None)
    guardrails = getattr(llm, "guardrails", [])
    include_feedback = getattr(llm, "include_feedback", False)
    fit_type = getattr(evaluation, "fit_type", "group")
    metadata = getattr(metadata, "flag", False)
    abstract_base_model = getattr(llm, "abstract_base_model", None)
    diversity_requirement = getattr(llm, "diversity_requirement", None)

    # Format goal section dynamically
    names = [f"`cognitive_model{i+1}`" for i in range(llm.models_per_iteration)]
    goal_text = task.goal.format(
        models_per_iteration=llm.models_per_iteration,
        model_names=", ".join(names),
    )

    if fit_type == "individual":
        individual_variability_feature = cfg.individual_difference.individual_feature
        if individual_variability_feature == "None":
            individual_variability_section = ""
        else:

            individual_variability_feature = np.array(data[individual_variability_feature])[0]
            individual_variability_section = cfg.individual_difference.description.format(individual_feature = individual_variability_feature)
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
            "Specifically, for each model parameter, we will run a regression predicting "
            "the parameter values from these questionnaire scores across participants. "
            "Models whose parameters show meaningful relationships with these measures "
            "(higher R²) are preferred, alongside good BIC.\n"
            "Design your model parameters to capture psychologically meaningful "
            "individual variation that could relate to these measures."
        )

    feedback_section = (
        f"\n\n### Feedback\n{feedback_text.strip()}"
        if (feedback_text and include_feedback)
        else ""
    )

    metadata_section = (
        f"### Metadata\n{cfg.metadata.description.strip()}" 
        if metadata 
        else "")

    introduce_data =  """Here is the participant data: """ if  fit_type == "individual" else """Here is the data from several participants: """ 

    if cfg.llm.provider in ["openai", "claude", "gemini"]:
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


{"### Base Class (DO NOT MODIFY) " + abstract_base_model.strip() if abstract_base_model else ""}

### Template Model
{llm.template_model.strip()}

### Guardrails
{chr(10).join(guardrails)}

{f"### Diversity Requirement\n{chr(10).join(diversity_requirement)}" if diversity_requirement else ""}

{"" if not id_eval_section else id_eval_section}

### Your Task
{goal_text.strip()}

{feedback_section}
""".strip()

    # --- prompt layout for open models ---
    else:

        prompt = f"""

{task.description.strip()}

{introduce_data}
{data_text.strip()}

{goal_text.strip()}

### Implementation Guidelines
{chr(10).join(guardrails)}

### Initial Model Template
{llm.template_model.strip()}

{"" if not id_eval_section else id_eval_section}

Your function:

{feedback_section}
        """.strip()

    return prompt




class PromptBuilderWrapper:
    def __init__(self, cfg, data_text, data):
        self.cfg = cfg
        self._data_text = data_text
        self.data = data
    def build_input_prompt(self, feedback_text: str = ""):
        return build_prompt(self.cfg, self._data_text, self.data, feedback_text=feedback_text)