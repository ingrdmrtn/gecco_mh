
import os, sys, re, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pathlib import Path
from config.schema import load_config
from gecco.prepare_data.io import load_data, split_by_participant, parse_split
from gecco.offline_evaluation.utils import build_model_spec_from_llm_output, extract_parameter_names
from scipy.optimize import minimize
from gecco.offline_evaluation.evaluation_functions import aic as _aic, bic as _bic
from gecco.utils import *
import matplotlib.pyplot as plt


# project_root = Path(__file__).resolve().parents[1]
project_root = Path('/home/aj9225/gecco-1')
cfg = load_config(project_root / "config" / "two_step_psychiatry_individual_stai_class_gemini-2.5-pro.yaml")
data_cfg = cfg.data
df = load_data(data_cfg.path)
participants = df.participant.unique()


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


p_stay = {'prob_stay_common_rewarded':[],
          'prob_stay_rare_rewarded':[],
          'prob_stay_common_not_rewarded':[],
          'prob_stay_rare_not_rewarded':[]}

for idx, p in enumerate(participants[14:]):

    print(p)
    df_participant = df[df.participant==p].reset_index()

    
    reward_p_s0_0, reward_p_s0_1, reward_p_s1_0, reward_p_s1_1 = (np.array(df_participant.reward_p_s0_0),
                                                                np.array(df_participant.reward_p_s0_1),
                                                                np.array(df_participant.reward_p_s1_0),
                                                                np.array(df_participant.reward_p_s1_1))
    stai = df_participant['stai'][0]
    n_trials = df_participant.shape[0]
    stage1_choice = np.array(df_participant['choice_1'])
    state2 = np.array(df_participant['state'])
    reward = np.array(df_participant['reward'])

    (prob_stay_common_rewarded,
    prob_stay_rare_rewarded,
    prob_stay_common_not_rewarded,
    prob_stay_rare_not_rewarded) = compute_stay_probabilities(stage1_choice, state2, reward)

    p_stay['prob_stay_common_rewarded'].append(prob_stay_common_rewarded)
    p_stay['prob_stay_rare_rewarded'].append(prob_stay_rare_rewarded)
    p_stay['prob_stay_common_not_rewarded'].append(prob_stay_common_not_rewarded)
    p_stay['prob_stay_rare_not_rewarded'].append(prob_stay_rare_not_rewarded)



ppcs = pd.DataFrame(p_stay)
ppcs.to_csv(f'{project_root}/analysis/two_step_task/ppcs_humans.csv')



figure, axis  = plt.subplots()
axis.bar(np.arange(4),[np.mean(np.mean(p_stay['prob_stay_common_rewarded'])),
                       np.mean(np.mean(p_stay['prob_stay_rare_rewarded'])),
                       np.mean(np.mean(p_stay['prob_stay_common_not_rewarded'])),
                       np.mean(np.mean(p_stay['prob_stay_rare_not_rewarded']))])
axis.set_title('Humans - Two Step Task')
axis.set_ylabel('Stay Probability')
axis.set_xticks(np.arange(4))
axis.set_xticklabels(['common/r','rare/r','common/nr','rare/nr'])
figure.savefig(f'{project_root}/analysis/two_step_task/ppcs_humans.png')

print('stop')
