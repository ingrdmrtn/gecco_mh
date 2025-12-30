import os, sys, re, glob, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gecco.offline_evaluation.fit_generated_models import run_fit
from config.schema import load_config
from gecco.prepare_data.io import load_data, split_by_participant, parse_split
from gecco.offline_evaluation.utils import build_model_spec_from_llm_output, extract_parameter_names
from gecco.utils import *
import matplotlib.pyplot as plt
from utils import compute_stay_probabilities
project_root = Path(__file__).resolve().parents[1]


model = 'group_metadata_stai'
split = 'test'
cfg = load_config(project_root / "config" / "two_step_psychiatry_group_metadata_stai.yaml")


data_cfg = cfg.data
metadata = cfg.metadata.flag
max_independent_runs  = cfg.loop.max_independent_runs

df = load_data(data_cfg.path)
splits = split_by_participant(df, data_cfg.id_column, data_cfg.splits)
df_test = splits[split]
participants = parse_split(getattr(cfg.data.splits, split, None), df.participant.unique().tolist())

best_model_path = f'{project_root}/results/{cfg.task.name}{"_" + cfg.evaluation.fit_type if cfg.evaluation.fit_type=="individual" else ""}/models/'
best_simulated_model_path = f'{project_root}/results/{cfg.task.name}{"_" + cfg.evaluation.fit_type if cfg.evaluation.fit_type=="individual" else ""}/simulation/'
simulation_columns  = cfg.data.simulation_columns
df_hybrid_bics = [np.array(df[df.participant == p].baseline_bic)[0] for p in participants]

for run in range(max_independent_runs):
    print(f"Evaluating run {run}...")

    files = glob.glob(f'{best_model_path}best_model_{run}.txt')

    if not files:
        print(f"No best model found for run {run}. Skipping.")
        continue

    best_model_file = files[0]

    func_code = open(best_model_file, 'r').read()
    func_name = re.search(r'def (\w+)\(', func_code).group(1)

    fit_res = run_fit(df_test, func_code, cfg=cfg, expected_func_name=func_name)

    print(f"Best GECCO model for run {run}: {np.mean(fit_res['eval_metrics'])} $\\pm$ {np.std(fit_res['eval_metrics'])/np.sqrt(len(fit_res['eval_metrics']))}")
    print(f"Hybrid model for run {run}: {np.mean(df_hybrid_bics)} $\\pm$ {np.std(df_hybrid_bics)/np.sqrt(len(df_hybrid_bics))}")

    p_stay = {'prob_stay_common_rewarded':[],
            'prob_stay_rare_rewarded':[],
            'prob_stay_common_not_rewarded':[],
            'prob_stay_rare_not_rewarded':[]}

    for idx, p in enumerate(participants):


        df_participant = df[df.participant==p].reset_index()

        best_parameters = fit_res['parameter_values'][idx]
        reward_p_s0_0, reward_p_s0_1, reward_p_s1_0, reward_p_s1_1 = (np.array(df_participant.reward_p_s0_0),
                                                                    np.array(df_participant.reward_p_s0_1),
                                                                    np.array(df_participant.reward_p_s1_0),
                                                                    np.array(df_participant.reward_p_s1_1))
        stai = df_participant['stai'][0]
        n_trials = df_participant.shape[0]
        participant_simulation_model = open(f'{best_simulated_model_path}simulation_model.txt', 'r')
        participant_simulation_model = participant_simulation_model.read()
        participant_simulation_model = extract_full_function(participant_simulation_model,'simulate_model')

        exec(participant_simulation_model, globals())
        model_func = globals()['simulate_model']

        parameter_names  = extract_parameter_names(participant_simulation_model)
        # make sure stai is passed to the model function depending on how stai_score is taken in by the function
        simulation_pars = [best_parameters[idx] for idx, name in enumerate(fit_res["param_names"])]
        stage1_choice, state2, stage2_choice, reward = model_func(
            n_trials,
            simulation_pars,
            *[globals()[name] for name in simulation_columns],
            stai_score = stai, ## was hand coded
        )
        


        (prob_stay_common_rewarded,
        prob_stay_rare_rewarded,
        prob_stay_common_not_rewarded,
        prob_stay_rare_not_rewarded) = compute_stay_probabilities(stage1_choice, state2, reward)

        p_stay['prob_stay_common_rewarded'].append(prob_stay_common_rewarded)
        p_stay['prob_stay_rare_rewarded'].append(prob_stay_rare_rewarded)
        p_stay['prob_stay_common_not_rewarded'].append(prob_stay_common_not_rewarded)
        p_stay['prob_stay_rare_not_rewarded'].append(prob_stay_rare_not_rewarded)



    ppcs = pd.DataFrame(p_stay)
    ppcs.to_csv(f'ppcs_model{model}_split{split}_run{run}.csv')



    figure, axis  = plt.subplots()
    axis.bar(np.arange(4),[np.mean(np.mean(p_stay['prob_stay_common_rewarded'])),
                        np.mean(np.mean(p_stay['prob_stay_rare_rewarded'])),
                        np.mean(np.mean(p_stay['prob_stay_common_not_rewarded'])),
                        np.mean(np.mean(p_stay['prob_stay_rare_not_rewarded']))])

    axis.set_xticks(np.arange(4))
    axis.set_xticklabels(['common/r','rare/r','common/nr','rare/nr'])
    figure.savefig(f'ppcs_model{model}_split{split}_run{run}.png')

print('stop')

