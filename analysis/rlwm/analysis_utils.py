import numpy as np
import pandas as pd


def learning_curves(participants,real_rlwm_data,llm_rlwm_data,literature_model_data):
    participant_real_ns3 = []
    participant_llm_ns3 = []
    participant_lit_ns3 = []


    participant_real_ns6 = []
    participant_llm_ns6= []
    participant_lit_ns6 = []

    max_repeat = 10

    for p in participants:
        real_rlwm_participant_data = real_rlwm_data[real_rlwm_data.participant == p].reset_index()
        llm_rlwm_participant_data = llm_rlwm_data[llm_rlwm_data.participant == p].reset_index()
        literature_model_rlwm_participant_data = literature_model_data[literature_model_data.participant == p].reset_index()

        blocks = real_rlwm_participant_data.blocks.unique()

        ns3_learning_curve_real = []
        ns3_learning_curve_llm = []
        ns3_learning_curve_lit = []

        ns6_learning_curve_real = []
        ns6_learning_curve_llm = []
        ns6_learning_curve_lit = []

        for b in blocks:

            print(f'participant:{p}; block:{b}')

            ns = np.array(real_rlwm_participant_data[real_rlwm_participant_data.blocks == b].set_sizes)[0]
            real_stimulus, real_rewards = np.array(real_rlwm_participant_data[real_rlwm_participant_data.blocks == b].stimulus), np.array(real_rlwm_participant_data[real_rlwm_participant_data.blocks == b].rewards)
            llm_stimulus, llm_rewards = np.array(llm_rlwm_participant_data[llm_rlwm_participant_data.blocks == b].stimulus), np.array(llm_rlwm_participant_data[llm_rlwm_participant_data.blocks == b].rewards)
            lit_model_stimulus, lit_model_rewards = np.array(literature_model_rlwm_participant_data[literature_model_rlwm_participant_data.blocks == b].stimulus), np.array(literature_model_rlwm_participant_data[literature_model_rlwm_participant_data.blocks == b].rewards)

            real_learning_curve = []#np.nan*np.ones((ns,max_repeat))
            llm_learning_curve = []#np.nan*np.ones((ns,max_repeat))
            lit_model_learning_curve = []#np.nan*np.ones((ns,max_repeat))




            iteration = np.stack([np.cumsum(real_stimulus == stim) for stim in np.unique(real_stimulus)], axis=1)
            colIteration = np.array(iteration[np.arange(len(real_stimulus)), real_stimulus])-1

            for st in range(9):
                # if b == 0 and p == 1:
                #     print('stop')
                # num_stimulus_occurences = np.sum(real_stimulus == st

                # np.mean(lit_model_rewards[colIteration == st])
                # np.mean(real_rewards[colIteration == st])


                real_learning_curve.append(np.mean(real_rewards[colIteration == st]))#[st,:num_stimulus_occurences] = np.cumsum(real_rewards[real_stimulus == st])/np.sum(real_stimulus == st)#    .append(np.cumsum(real_rewards[real_stimulus == st]) / np.sum(real_stimulus == st))
                llm_learning_curve.append(np.mean(llm_rewards[colIteration == st]))#[st, :num_stimulus_occurences] = np.cumsum(llm_rewards[llm_stimulus == st])/np.sum(llm_stimulus == st)#    .append(np.cumsum(real_rewards[real_stimulus == st]) / np.sum(real_stimulus == st))
                lit_model_learning_curve.append(np.mean(lit_model_rewards[colIteration == st]))#[st, :num_stimulus_occurences] = np.cumsum(lit_model_rewards[lit_model_stimulus == st])/np.sum(lit_model_stimulus == st)#    .append(np.cumsum(real_rewards[real_stimulus == st]) / np.sum(real_stimulus == st))

            if ns == 3:
                ns3_learning_curve_real.append(real_learning_curve)
                ns3_learning_curve_llm.append(llm_learning_curve)
                ns3_learning_curve_lit.append(lit_model_learning_curve)

            else:
                ns6_learning_curve_real.append(real_learning_curve)
                ns6_learning_curve_llm.append(llm_learning_curve)
                ns6_learning_curve_lit.append(lit_model_learning_curve)

        participant_real_ns3.append(np.nanmean(np.array(ns3_learning_curve_real),axis=0))
        participant_llm_ns3.append(np.nanmean(np.array(ns3_learning_curve_llm),axis=0))
        participant_lit_ns3.append(np.nanmean(np.array(ns3_learning_curve_lit),axis=0))

        participant_real_ns6.append(np.nanmean(np.array(ns6_learning_curve_real),axis=0))
        participant_llm_ns6.append(np.nanmean(np.array(ns6_learning_curve_llm),axis=0))
        participant_lit_ns6.append(np.nanmean(np.array(ns6_learning_curve_lit),axis=0))





    real_ns3_mean = np.mean(participant_real_ns3,axis=0)
    llm_ns3_mean =np.mean(participant_llm_ns3,axis=0)
    lit_ns3_mean = np.mean(participant_lit_ns3,axis=0)

    real_ns3_sem = np.std(participant_real_ns3,axis=0)/np.sqrt(len(participants))
    llm_ns3_sem =np.std(participant_llm_ns3,axis=0)/np.sqrt(len(participants))
    lit_ns3_sem = np.std(participant_lit_ns3,axis=0)/np.sqrt(len(participants))



    real_ns6_mean = np.mean(participant_real_ns6,axis=0)
    llm_ns6_mean =np.mean(participant_llm_ns6,axis=0)
    lit_ns6_mean = np.mean(participant_lit_ns6,axis=0)
    real_ns6_sem = np.std(participant_real_ns6,axis=0)/np.sqrt(len(participants))
    llm_ns6_sem =np.std(participant_llm_ns6,axis=0)/np.sqrt(len(participants))
    lit_ns6_sem = np.std(participant_lit_ns6,axis=0)/np.sqrt(len(participants))



    return (real_ns3_mean, llm_ns3_mean, lit_ns3_mean,
            real_ns3_sem, llm_ns3_sem, lit_ns3_sem,
            real_ns6_mean, llm_ns6_mean, lit_ns6_mean,
            real_ns6_sem, llm_ns6_sem, lit_ns6_sem)
