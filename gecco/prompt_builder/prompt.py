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
    task, llm, evaluation, metadata = cfg.task, cfg.llm, cfg.evaluation, cfg.metadata
    guardrails = getattr(llm, "guardrails", [])
    include_feedback = getattr(llm, "include_feedback", False)
    fit_type = getattr(evaluation, "fit_type")
    metadata = getattr(metadata, "flag", False)

    # Format goal section dynamically
    names = [f"`cognitive_model{i+1}`" for i in range(llm.models_per_iteration)]
    goal_text = task.goal.format(
        models_per_iteration=llm.models_per_iteration,
        model_names=", ".join(names),
    )

    if fit_type == "individual":
        individual_variability_feature = cfg.metadata.individual_feature
        if individual_variability_feature == "None":
            individual_variability_section = ""
        else:

            individual_variability_feature = data[individual_variability_feature][0]
            individual_variability_section = cfg.metadata.description.format(individual_feature = individual_variability_feature)
    else:
        individual_variability_section = ""

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

# individual_variability_section
{individual_variability_section}


### Participant Data
{introduce_data}
{data_text.strip()}

### Your Task
{goal_text.strip()}

### Guardrails
{chr(10).join(guardrails)}

### Template Model (for reference only — do not reuse its logic)
{llm.template_model.strip()}

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