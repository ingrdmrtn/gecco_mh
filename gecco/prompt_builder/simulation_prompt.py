

def simulation_prompt(best_model,cfg):

    simulation_prompt = f"""
    
Below  is model fitting code:

{cfg.llm.abstract_base_model if getattr(cfg.llm, "abstract_base_model", None) else ""}

{best_model}

Your task is to convert it into model simulation code. Below is an example of what it should look like:

{cfg.llm.simulation_template}

The function should be called simulate_model.
It should take the following inputs: {cfg.data.simulation_columns}, parameters.
It should return the following: {cfg.data.simulation_return}.
Make sure your simulation code follows the logic of the best fitting model. 

    """

    return simulation_prompt