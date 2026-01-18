from gecco.load_llms.model_loader import load_llm
from gecco.utils import *
from gecco.offline_evaluation.evaluation_functions import aic as _aic, bic as _bic
from scipy.optimize import minimize
from gecco.offline_evaluation.utils import build_model_spec_from_llm_output, extract_parameter_names
from gecco.prepare_data.io import load_data, split_by_participant
from config.schema import load_config
from pathlib import Path
from gecco.run_gecco import GeCCoModelSearch
import re
import json
import glob
import numpy as np
import pandas as pd
import ipdb
import os
import sys

from regex import match
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))


# from data.ocd.preprocess_data import hybrid_model


def save_llm_json(raw_text):

    # 1. Use Regex to find the content inside json ... or just { ... } # This pattern looks for content between curly braces, handling the code blocks
    try:
        match = re.search(r"{.*}", raw_text, re.DOTALL)
        if match:
            json_str = match.group(0)

            # 2. Parse the string into a Python dict to ensure it's valid
            data = json.loads(json_str)

            # # 3. Save to file
            # with open(filename, 'w') as f:
            #     json.dump(data, f, indent=4)
            print(f"Success! valid JSON extracted and saved.")
        else:
            print("Error: No JSON object found in the text.")

    except json.JSONDecodeError as e:
        print(
            f"Error: The text was found but is not valid JSON.\nDetails: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return data


rng = np.random.default_rng()

# project_root = Path(__file__).resolve().parents[1]
project_root = Path(__file__).resolve().parents[2]
cfg = load_config(project_root / "config" /
                  "two_step_psychiatry_individual_function_gemini-3-pro_ocd_maxsetting.yaml")
compare_config = load_config(project_root / "analysis" /
                             "two_step_task" / "compare_gecco_baseline.yaml")
data_cfg = cfg.data
df = load_data(data_cfg.path)
participants = df.participant.unique()

metric_name = cfg.evaluation.metric.upper()
metric_map = {"AIC": _aic, "BIC": _bic}
metric_func = metric_map.get(metric_name, _bic)

best_models = f'{project_root}/results/{cfg.task.name}_{cfg.evaluation.fit_type}/models/'
best_simulated_models = f'{project_root}/results/{cfg.task.name}_{cfg.evaluation.fit_type}/simulation/'
simulation_columns = cfg.data.simulation_columns
param_dir = project_root / \
    f"results/{cfg.task.name}_{cfg.evaluation.fit_type}/parameters/"

best_params_list = {'participant': [], 'parameters': [],
                    'param_names': [], 'num_params': [],
                    'shared_params': [], 'unique_params': [],
                    'shared_mechanisms': [], 'unique_mechanisms': [],
                    'shared': [], 'model_type': []}

for p in participants[14:]:

    print(p)
    df_participant = df[df.participant == p].reset_index()
    # load best model
    model_path = f'{best_models}best_model_0_participant{p}.txt'
    with open(model_path, 'r') as f:
        best_model = f.read()
    psychiatry = False
    vals = {'base_code': compare_config.compare.hybrid_model, 'model_code': best_model,
            'psychiatry': "- OCD: obsesstive compulsive disorder score (0-1) modulates behavior" if psychiatry else ""}
    prompt = compare_config.compare.comparison_prompt.format(**vals)

    model, tokenizer = load_llm(cfg.llm.provider, cfg.llm.base_model)
    search = GeCCoModelSearch(model, tokenizer, cfg, df_participant, None)
    output = search.generate(model, tokenizer, prompt)
    print(output)
    model_outputs = save_llm_json(output)

    try:
        # print(param_dir)
        best_parameters = pd.read_csv(
            f'{param_dir}/best_params_run0_participant{p}.csv')
    except:  # noqa: E722
        print(f'No parameters for participant {p}')
        continue

    # parameter_names = extract_parameter_names(participant_simulation_model)

    # if oci in  parameter_names:
    parameters = [best_parameters[n][0] for n in best_parameters.columns]
    best_params_list['param_names'].append(
        best_parameters.columns.values)
    best_params_list['participant'].append(p)
    best_params_list['parameters'].append(parameters)
    best_params_list['num_params'].append(len(parameters))
    best_params_list['shared_params'].append(
        model_outputs['shared_parameters'])
    best_params_list['unique_params'].append(
        model_outputs['unique_parameters'])
    best_params_list['shared_mechanisms'].append(
        model_outputs['shared_mechanisms'])
    best_params_list['unique_mechanisms'].append(
        model_outputs['unique_mechanisms'])
    best_params_list['shared'].append(model_outputs['shared'])
    best_params_list['model_type'].append(model_outputs['model_type'])

pooled_best_params = pd.DataFrame(best_params_list)
pooled_best_params.to_csv(
    f'{project_root}/analysis/two_step_task/pooled_params_{cfg.task.name}_{cfg.evaluation.fit_type}_oci.csv')
