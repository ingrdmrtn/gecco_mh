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


project_root = Path(__file__).resolve().parents[2]
cfg = load_config(project_root / "config" / "rlwm.yaml")
data_cfg = cfg.data
df = load_data(data_cfg.path)
# participants = df.participant.unique()

metric_name = cfg.evaluation.metric.upper()
metric_map = {"AIC": _aic, "BIC": _bic}
metric_func = metric_map.get(metric_name, _bic)

best_models = f'{project_root}/results/{cfg.task.name}/models/'
best_simulated_models = f'{project_root}/results/{cfg.task.name}/simulation/'
simulation_columns  = cfg.data.simulation_columns

p_correct_ns3_young = []
p_correct_ns3_old = []

p_correct_ns6_young = []
p_correct_ns6_old = []


p_correct_ns3_young_real = []
p_correct_ns3_old_real = []

p_correct_ns6_young_real = []
p_correct_ns6_old_real = []


param_dir = project_root / "results/rlwm_individual/parameters/"

# do the participants that weren't used for evaluation or fitting
young_participants = list(df[df.age < 45].participant.unique()[:15])
old_participants = list(df[df.age > 45].participant.unique()[:15])
all_participants = young_participants + old_participants


def learning_curves(p_df):

    participant_ns_3 = []
    participant_ns_6 = []

    blocks = p_df.blocks.unique()

    ns3_learning_curve = []
    ns6_learning_curve = []

    for b in blocks:

        print(f'participant:{p}; block:{b}')

        ns = np.array(p_df[p_df.blocks == b].set_sizes)[0]
        stimulus, rewards = np.array(p_df[p_df.blocks == b].stimulus), np.array(p_df[p_df.blocks == b].rewards)

        learning_curve = []#np.nan*np.ones((ns,max_repeat))
        iteration = np.stack([np.cumsum(stimulus == stim) for stim in np.unique(stimulus)], axis=1)
        colIteration = np.array(iteration[np.arange(len(stimulus)), stimulus])-1

        for st in range(9):

            learning_curve.append(np.mean(rewards[colIteration == st]))

        if ns == 3:
            ns3_learning_curve.append(learning_curve)
        else:
            ns6_learning_curve.append(learning_curve)

    participant_ns_3.append(np.nanmean(np.array(ns3_learning_curve),axis=0))
    participant_ns_6.append(np.nanmean(np.array(ns6_learning_curve),axis=0))


    ns3_mean = np.mean(participant_ns_3,axis=0)
    ns3_sem = np.std(participant_ns_3,axis=0)/np.sqrt(len(all_participants))

    ns6_mean = np.mean(participant_ns_6,axis=0)
    ns6_sem = np.std(participant_ns_6,axis=0)/np.sqrt(len(all_participants))


    return (ns3_mean, ns3_sem,
            ns6_mean, ns6_sem)




best_model = open(f'{best_models}best_model_0.txt', 'r')
best_model = best_model.read()
best_model = extract_full_function(best_model,best_model[4:20])
exec(best_model, globals())
model_func = globals()[best_model[4:20]]

n_starts = getattr(cfg.evaluation, "n_starts", 10)
spec = build_model_spec_from_llm_output(best_model, expected_func_name=best_model[4:20])
parameter_bounds = list(spec.bounds.values())

eval_metrics = []
parameter_estimates = []

for p in all_participants:
    df_participant = df[df.participant==p].reset_index()
    print(p)


    input_cols = list(data_cfg.input_columns)
    inputs = [df_participant[c].to_numpy() for c in input_cols]

    min_ll = np.inf
    best_parameter_values = []
    for _ in range(n_starts):
        x0 = [np.random.uniform(lo, hi) for lo, hi in parameter_bounds]
        res = minimize(
            lambda x: float(model_func(*inputs, x)),
            x0,
            method="L-BFGS-B",
            bounds=parameter_bounds,
        )
        if res.fun < min_ll:
            min_ll = res.fun
            best_parameter_values = res.x

    eval_metrics.append(metric_func(min_ll, len(parameter_bounds), len(df_participant)))
    parameter_estimates.append(best_parameter_values)



    ns3_mean_real, ns3_sem_real, ns6_mean_real, ns6_sem_real = learning_curves(df_participant)
    if p >= 36:
        p_correct_ns3_old_real.append(ns3_mean_real)
        p_correct_ns6_old_real.append(ns6_mean_real)

    else:
        p_correct_ns3_young_real.append(ns3_mean_real)
        p_correct_ns6_young_real.append(ns6_mean_real)


    # stimulus, blocks, set_sizes, correct_answer, age, parameters
    stimulus, blocks , set_sizes, correct_answer, age = (np.array(df_participant.stimulus),
                                                          np.array(df_participant.blocks),
                                                          np.array(df_participant.set_sizes),
                                                          np.array(df_participant.correct_answer),
                                                          np.array(df_participant.age))

    participant_simulation_model = open(f'{best_simulated_models}simulation_model.txt', 'r')
    participant_simulation_model = participant_simulation_model.read()
    participant_simulation_model = extract_full_function(participant_simulation_model,'simulate_model')

    exec(participant_simulation_model, globals())
    model_func_sim = globals()['simulate_model']

    parameter_names  = extract_parameter_names(participant_simulation_model)

    # if stai in  parameter_names:
    simulation_pars = best_parameter_values
    #simulation_pars.append(stai)
    simulated_actions, simulated_rewards  = model_func_sim(stimulus, blocks, set_sizes, correct_answer, simulation_pars)
    # else:


    ppc_df = pd.DataFrame({'actions':simulated_actions,
                  'rewards':simulated_rewards,
                  'stimulus':stimulus,
                  'correct_answer':simulated_rewards,
                  'blocks': blocks,
                  'set_sizes':set_sizes})
    ns3_mean, ns3_sem, ns6_mean, ns6_sem = learning_curves(ppc_df)


    if p >= 36:
        p_correct_ns3_old.append(ns3_mean)
        p_correct_ns6_old.append(ns6_mean)

    else:
        p_correct_ns3_young.append(ns3_mean)
        p_correct_ns6_young.append(ns6_mean)





# ppcs = pd.DataFrame(p_stay)
# ppcs.to_csv('ppcs_individual.csv')



figure, axis  = plt.subplots(1,2, figsize = (12,5))
axis[0].plot(np.arange(9), np.mean(p_correct_ns3_young,axis =0), '--',color='b')
axis[0].plot(np.arange(9), np.mean(p_correct_ns6_young,axis =0), '--',color='r')
axis[0].plot(np.arange(9), np.mean(p_correct_ns3_young_real,axis =0),'b')
axis[0].plot(np.arange(9), np.mean(p_correct_ns6_young_real,axis =0),'r')



axis[1].plot(np.arange(9), np.mean(p_correct_ns3_old,axis =0), '--', color = 'b')
axis[1].plot(np.arange(9), np.mean(p_correct_ns6_old,axis =0), '--', color = 'r')
axis[1].plot(np.arange(9), np.mean(p_correct_ns3_old_real,axis =0),'b')
axis[1].plot(np.arange(9), np.mean(p_correct_ns6_old_real,axis =0),'r')

axis[0].set_xticks(np.arange(9))
axis[0].set_xlabel('stimulus iterations')
axis[0].set_ylabel('p(correct)')
axis[0].set_title('young')
axis[0].set_ylim([0,1])

axis[1].set_xticks(np.arange(9))
axis[1].set_xlabel('stimulus iterations')
axis[1].set_ylabel('p(correct)')
axis[1].set_title('old')
axis[1].set_ylim([0,1])


figure.savefig('ppcs_group.png')

print('stop')



n3_old = pd.DataFrame(p_correct_ns3_old)
n3_old.to_csv('ns3_old_group_age.csv')
n6_old = pd.DataFrame(p_correct_ns6_old)
n6_old.to_csv('ns6_old_group_age.csv')
n3_young = pd.DataFrame(p_correct_ns3_young)
n3_young.to_csv('ns3_young_group_age.csv')
n6_young = pd.DataFrame(p_correct_ns6_young)
n6_young.to_csv('ns6_young_group_age.csv')



bics = pd.DataFrame({'bic':eval_metrics})
bics.to_csv('group_bics.csv')