import os, sys, re, glob, numpy as np, pandas as pd, matplotlib.pyplot as plt

group_bics = pd.read_csv('group_bics.csv')[:10]
individual_age_bics = pd.read_csv('individual_age_bics.csv')[:10]
baseline_bics = pd.read_csv('baseline_bics.csv')[:10]
individual_bics = pd.read_csv('individual_bics.csv')[:10]





figure, axis  = plt.subplots(figsize=(5,5))
axis.bar(np.arange(4),[np.mean(baseline_bics['bic']), np.mean(group_bics['bic']), np.mean(individual_bics['bic']), np.mean(individual_age_bics['bic'])])
axis.errorbar(np.arange(4),
         [np.mean(baseline_bics['bic']), np.mean(group_bics['bic']), np.mean(individual_bics['bic']), np.mean(individual_age_bics['bic'])],
         [np.std(baseline_bics['bic'])/np.sqrt(len(baseline_bics['bic'])),
          np.std(group_bics['bic'])/np.sqrt(len(group_bics['bic'])),
          np.std(individual_bics['bic'])/np.sqrt(len(individual_bics['bic'])),
          np.std(individual_age_bics['bic'])/np.sqrt(len(individual_age_bics['bic']))],
    fmt = 'o',color = 'k')
axis.set_xticks(np.arange(4))
axis.set_xticklabels(['baseline','group\ngecco','individual' , 'individual\n+ age'])
axis.set_ylabel('bic')
axis.set_ylim([200,300])

figure.savefig('bics_comparison.png')
