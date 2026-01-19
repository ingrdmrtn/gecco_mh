import os, sys, re, glob, numpy as np, pandas as pd, matplotlib.pyplot as plt

group_bics = pd.read_csv('group_bics.csv')
individual_age_bics = pd.read_csv('individual_age_bics.csv')
baseline_bics = pd.read_csv('baseline_bics.csv')
individual_bics = pd.read_csv('individual_bics.csv')

best_idx = []
for i in range(len(individual_bics)):

    best_idx.append(np.argmin([baseline_bics['bic'][i],
               group_bics['bic'][i],
               individual_bics['bic'][i],
               individual_age_bics['bic'][i]]))

counts = []
for i in np.arange(4):
    counts.append(np.sum(best_idx == i))



figure, axis  = plt.subplots(1,2,figsize=(15,5))
axis[0].bar(np.arange(4),
         [np.mean(baseline_bics['bic']),
          np.mean(group_bics['bic']),
          np.mean(individual_bics['bic']),
          np.mean(individual_age_bics['bic'])])
axis[0].errorbar(np.arange(4),
         [np.mean(baseline_bics['bic']), np.mean(group_bics['bic']), np.mean(individual_bics['bic']), np.mean(individual_age_bics['bic'])],
         [np.std(baseline_bics['bic'])/np.sqrt(len(baseline_bics['bic'])),
          np.std(group_bics['bic'])/np.sqrt(len(group_bics['bic'])),
          np.std(individual_bics['bic'])/np.sqrt(len(individual_bics['bic'])),
          np.std(individual_age_bics['bic'])/np.sqrt(len(individual_age_bics['bic']))],
    fmt = 'o',color = 'k')
axis[0].set_xticks(np.arange(4))
axis[0].set_xticklabels(['baseline','group\ngecco','individual\ngecco' ,'individual\ngecco + age'])
axis[0].set_ylabel('bic')
axis[0].set_ylim([220, 320])



axis[1].bar(np.arange(4),counts)
axis[1].set_ylabel('# participants\nbest fit by model')
axis[1].set_xticks(np.arange(4))
axis[1].set_xticklabels(['baseline','group\ngecco','individual\ngecco' ,'individual\ngecco + age'])

figure.savefig('bics_comparison.png')
