import os, sys, re, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path
from config.schema import load_config
from gecco.prepare_data.io import load_data, split_by_participant
from gecco.offline_evaluation.utils import build_model_spec_from_llm_output, extract_parameter_names
from scipy.optimize import minimize
from gecco.offline_evaluation.evaluation_functions import aic as _aic, bic as _bic
from gecco.utils import *
import matplotlib.pyplot as plt


project_root = Path(__file__).resolve().parents[1]
cfg = load_config(project_root / "config" / "two_step_psychiatry_baseline_individual.yaml")
data_cfg = cfg.data
df = load_data(data_cfg.path)
participants = df.participant.unique()

metric_name = cfg.evaluation.metric.upper()
metric_map = {"AIC": _aic, "BIC": _bic}
metric_func = metric_map.get(metric_name, _bic)

best_models = f'{project_root}/results/{cfg.task.name + '_' + cfg.evaluation.fit_type}/models/'
best_simulated_models = f'{project_root}/results/{cfg.task.name + '_' + cfg.evaluation.fit_type}/simulation/'
simulation_columns  = cfg.data.simulation_columns

p_stay = {'prob_stay_common_rewarded':[],
          'prob_stay_rare_rewarded':[],
          'prob_stay_common_not_rewarded':[],
          'prob_stay_rare_not_rewarded':[]}
param_dir = project_root / "results/two_step_task_individual/parameters/"




def compute_stay_probabilities(choice_1, state, reward):
    choice_1 = np.asarray(choice_1)
    state = np.asarray(state)
    reward = np.asarray(reward)

    # stay from trial t -> t+1
    stay = (choice_1[1:] == choice_1[:-1])   # length T-1

    # common / rare transition on previous trial (t-1)
    common_transition = (
        ((choice_1 == 0) & (state == 0)) |
        ((choice_1 == 1) & (state == 1))
    )[:-1]                                   # length T-1

    rare_transition = ~common_transition     # complementary

    # reward on previous trial (t-1)
    rewarded = (reward[:-1] == 1)
    not_rewarded = ~rewarded                 # since 0/1

    def stay_prob(condition):
        return np.mean(stay[condition]) if np.any(condition) else np.nan

    prob_stay_common_rewarded     = stay_prob(common_transition & rewarded)
    prob_stay_common_not_rewarded = stay_prob(common_transition & not_rewarded)
    prob_stay_rare_rewarded       = stay_prob(rare_transition & rewarded)
    prob_stay_rare_not_rewarded   = stay_prob(rare_transition & not_rewarded)

    return [
        prob_stay_common_rewarded,
        prob_stay_rare_rewarded,
        prob_stay_common_not_rewarded,
        prob_stay_rare_not_rewarded,
    ]




for p in participants:

    print(p)
    df_participant = df[df.participant==p].reset_index()

    best_parameters = pd.read_csv(f'{param_dir}/best_params_run0_participant{p}.csv')
    reward_p_s0_0, reward_p_s0_1, reward_p_s1_0, reward_p_s1_1 = (np.array(df_participant.reward_p_s0_0),
                                                                  np.array(df_participant.reward_p_s0_1),
                                                                  np.array(df_participant.reward_p_s1_0),
                                                                  np.array(df_participant.reward_p_s1_1))
    stai = df_participant['stai'][0]
    n_trials = df_participant.shape[0]
    participant_simulation_model = open(f'{best_simulated_models}simulation_model_participant{p}.txt', 'r')
    participant_simulation_model = participant_simulation_model.read()
    participant_simulation_model = extract_full_function(participant_simulation_model,'simulate_model')

    exec(participant_simulation_model, globals())
    model_func = globals()['simulate_model']

    parameter_names  = extract_parameter_names(participant_simulation_model)

    # if stai in  parameter_names:
    simulation_pars = [best_parameters[n][0] for n in best_parameters.columns]
    #simulation_pars.append(stai)
    stage1_choice, state2, stage2_choice, reward = model_func(
        n_trials,
        simulation_pars,
        *[globals()[name] for name in simulation_columns]
    )
    # else:


    (prob_stay_common_rewarded,
    prob_stay_rare_rewarded,
    prob_stay_common_not_rewarded,
    prob_stay_rare_not_rewarded) = compute_stay_probabilities(stage1_choice, state2, reward)

    p_stay['prob_stay_common_rewarded'].append(prob_stay_common_rewarded)
    p_stay['prob_stay_rare_rewarded'].append(prob_stay_rare_rewarded)
    p_stay['prob_stay_common_not_rewarded'].append(prob_stay_common_not_rewarded)
    p_stay['prob_stay_rare_not_rewarded'].append(prob_stay_rare_not_rewarded)



ppcs = pd.DataFrame(p_stay)
ppcs.to_csv('ppcs_individual.csv')



figure, axis  = plt.subplots()
axis.bar(np.arange(4),[np.mean(np.mean(p_stay['prob_stay_common_rewarded'])),
                       np.mean(np.mean(p_stay['prob_stay_rare_rewarded'])),
                       np.mean(np.mean(p_stay['prob_stay_common_not_rewarded'])),
                       np.mean(np.mean(p_stay['prob_stay_rare_not_rewarded']))])

axis.set_xticks(np.arange(4))
axis.set_xticklabels(['common/r','rare/r','common/nr','rare/nr'])
figure.savefig('ppcs_individual.png')

print('stop')





