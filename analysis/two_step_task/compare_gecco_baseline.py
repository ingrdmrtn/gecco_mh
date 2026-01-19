import os
import sys
from regex import match
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
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
run = 0
cfg = load_config(project_root / "config" /
                  "two_step_psychiatry_individual_function_gemini-3-pro_ocd_maxsetting.yaml")
compare_config = load_config(project_root / "analysis" /
                             "two_step_task" / "compare_gecco_baseline_twostep_individual_ocd.yaml")
data_cfg = cfg.data
df = load_data(data_cfg.path)
participants = df.participant.unique()

metric_name = cfg.evaluation.metric.upper()
metric_map = {"AIC": _aic, "BIC": _bic}
metric_func = metric_map.get(metric_name, _bic)

best_models = f'{project_root}/results/{cfg.task.name}_{cfg.evaluation.fit_type}/models/'
bics = f'{project_root}/results/{cfg.task.name}_{cfg.evaluation.fit_type}/bics/'
best_simulated_models = f'{project_root}/results/{cfg.task.name}_{cfg.evaluation.fit_type}/simulation/'
simulation_columns = cfg.data.simulation_columns
param_dir = project_root / \
    f"results/{cfg.task.name}_{cfg.evaluation.fit_type}/parameters/"

best_params_list = {'participant': [], 'fitted_parameters': [],
                    'parsed_params': [], 'num_params': [], 'extraction_mismatch': [],
                    'shared_params': [], 'unique_params': [],
                    'shared_mechanisms': [], 'unique_mechanisms': [],
                    'shared': [], 'model_type': [], 'baseline_params': [],
                    'baseline_bic': [], 'best_model_bic': []}
baseline_params = ['learning_rate', 'learning_rate_2', 'beta', 'beta_2', 'w', 'lambd', 'perseveration']
if os.path.exists(f'{project_root}/results/{cfg.task.name}_{cfg.evaluation.fit_type}/gecco_baseline_comparison.csv'):
        existing_df = pd.read_csv(
            f'{project_root}/results/{cfg.task.name}_{cfg.evaluation.fit_type}/gecco_baseline_comparison.csv')
        processed_participants = existing_df['participant'].values
else:
    processed_participants = []
for p in participants[14:]:

    if p in processed_participants:
        print(f'Participant {p} already processed, skipping.')
        continue

    print(f'Processing participant {p}')
    df_participant = df[df.participant == p].reset_index()
    # load best model
    model_path = f'{best_models}best_model_{run}_participant{p}.txt'
    with open(model_path, 'r') as f:
        best_model = f.read()
    vals = {'base_code': compare_config.compare.base_model, 'model_code': best_model,
            'psychiatry': "- OCD: obsesstive compulsive disorder score (0-1) modulates behavior" if compare_config.compare.psychiatry else ""}
    prompt = compare_config.compare.comparison_prompt.format(**vals)
    model, tokenizer = load_llm(cfg.llm.provider, cfg.llm.base_model)
    search = GeCCoModelSearch(model, tokenizer, cfg, df_participant, None)
    retry_count = 0
    while True:
        output = search.generate(model, tokenizer, prompt)
        print(output)
        model_outputs = save_llm_json(output)
        # if baseline_params and participant_model_parms match with regex'ed output, break
        llm_extracted_model_parameter_names = model_outputs["participant_model_parameter_names"]
        llm_extracted_base_model_parameter_names = model_outputs["base_model_parameter_names"]
        # retry if baseline_params do not match
        if set(baseline_params) == set(llm_extracted_base_model_parameter_names):
            print('Baseline model parameters match expected parameters.')
            best_params_list['extraction_mismatch'].append(False)
            break
        if retry_count >= 3:
            print(
                f'Max retries reached for participant {p}. Proceeding with current LLM output despite mismatch in baseline parameters.')
            best_params_list['extraction_mismatch'].append(True)
            break
        else:
            prompt = prompt + "\n\nThe previous output did not match the expected baseline model parameters. Please ensure that the baseline model parameters are correctly parsed. Try again."
            print(
                f'Retrying LLM output for participant {p} due to mismatch in baseline parameters.') 
            retry_count += 1
    try:
        # print(param_dir)
        best_model_parameters = pd.read_csv(
            f'{param_dir}/best_params_run{run}_participant{p}.csv')
        fitted_parameters = [best_model_parameters[n][0] for n in best_model_parameters.columns]
    except:  # noqa: E722
        print(f'No parameters for participant {p}')
        continue
    
    # load bics 
    bic_path = f'{bics}best_bic_{run}_participant{p}.json'
    # load json from bic_path
    bic = json.load(open(bic_path, 'r'))
    # get baseline model bic
    best_model_bic = bic['bic']
    # baseline bic from df_participant
    baseline_bic = df_participant['baseline_bic'].iloc[0]

    # if oci in  parameter_names:
    best_params_list['participant'].append(p)
    best_params_list['parsed_params'].append(
        best_model_parameters.columns.values)
    best_params_list['fitted_parameters'].append(fitted_parameters)
    best_params_list['num_params'].append(len(fitted_parameters))
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
    best_params_list['baseline_params'].append(baseline_params)
    best_params_list['baseline_bic'].append(baseline_bic)
    best_params_list['best_model_bic'].append(best_model_bic)
    best_params_df = pd.DataFrame(best_params_list)
    
    # check if dataframe ecists append to existing else create new
    if os.path.exists(f'{project_root}/results/{cfg.task.name}_{cfg.evaluation.fit_type}/gecco_baseline_comparison.csv'):
        combined_df = pd.concat([existing_df, best_params_df], ignore_index=True)
        combined_df.to_csv(f'{project_root}/results/{cfg.task.name}_{cfg.evaluation.fit_type}/gecco_baseline_comparison.csv', index=False)
        print(f'Appended comparison results to {project_root}/results/{cfg.task.name}_{cfg.evaluation.fit_type}/gecco_baseline_comparison.csv')
    else:
        best_params_df.to_csv(f'{project_root}/results/{cfg.task.name}_{cfg.evaluation.fit_type}/gecco_baseline_comparison.csv', index=False)
        print(f'Saved comparison results to {project_root}/results/{cfg.task.name}_{cfg.evaluation.fit_type}/gecco_baseline_comparison.csv')