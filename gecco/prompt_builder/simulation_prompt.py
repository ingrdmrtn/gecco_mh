

def simulation_prompt(best_model, simulation_template):

    simulation_prompt = f"""
    
Below  is model fitting code:

{best_model}

Your task is to convert it into model simulation code. Below is an example of what it should look like:

{simulation_template}

The function should be called simulate_model.
It should take the following inputs: n_trials, parameters, drift1, drift2, drift3, drift4):
It should return the following: stage1_choice, state2, stage2_choice, reward
Make sure your simulation code follows the logic of the best fitting model. 

    """

    return simulation_prompt