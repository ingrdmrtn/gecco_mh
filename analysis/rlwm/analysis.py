import os, sys, re, glob, numpy as np, pandas as pd, matplotlib.pyplot as plt

group_bics = pd.read_csv('group_bics.csv')
individual_age_bics = pd.read_csv('individual_age_bics.csv')
baseline_bics = pd.read_csv('baseline_bics.csv')


figure, axis  = plt.subplots(figsize=(5,5))
axis.bar(np.arange(3),[np.mean(baseline_bics['bic']), np.mean(group_bics['bic']), np.mean(individual_age_bics['bic'])])
axis.errorbar(np.arange(3),
         [np.mean(baseline_bics['bic']), np.mean(group_bics['bic']), np.mean(individual_age_bics['bic'])],
         [np.std(baseline_bics['bic'])/np.sqrt(len(baseline_bics['bic'])), np.std(group_bics['bic'])/np.sqrt(len(group_bics['bic'])), np.std(individual_age_bics['bic'])/np.sqrt(len(group_bics['bic']))],
    fmt = 'o',color = 'k')
axis.set_xticks(np.arange(3))
axis.set_xticklabels(['baseline','group','individual'])
axis.set_ylabel('bic')
axis.set_ylim([270,370])

figure.savefig('bics_comparison.png')
