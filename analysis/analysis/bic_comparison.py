import os, sys, re, glob,numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
ids = np.array(pd.read_csv(f"{project_root}/ids.csv").drop(columns="Unnamed: 0")["ids"])


group_bics = pd.read_csv('group_bics.csv')
individual_age_bics = pd.read_csv('individual_age_bics.csv')
baseline_bics = pd.read_csv('baseline_bics.csv')
individual_bics = pd.read_csv('individual_bics.csv')




best_idx = []
best_id_age_young = []
best_id_age_old = []

for i in range(len(individual_bics)):

    best_model_idx = np.argmin([baseline_bics['bic'][i],
               group_bics['bic'][i],
               individual_bics['bic'][i],
               individual_age_bics['bic'][i]])


    best_idx.append(best_model_idx)


    if best_model_idx == 3:
        if ids[i] <= 36 :
            best_id_age_young.append(ids[i])
        else:
            best_id_age_old.append(ids[i])



len(best_id_age_young)/sum(ids<36)
len(best_id_age_old)/sum(ids>=36)


best_age_id_young = pd.DataFrame({'ids':best_id_age_young})
best_age_id_young.to_csv(f"{project_root}/best_age_id_young.csv")

best_age_id_old = pd.DataFrame({'ids':best_id_age_old})
best_age_id_old.to_csv(f"{project_root}/best_age_id_old.csv")



counts = []
for i in np.arange(4):
    counts.append(np.sum(best_idx == i))

best_age_model_bics = ids[np.array(best_idx)==3]
best_age_model_bics = pd.DataFrame({'ids':best_age_model_bics})
best_age_model_bics.to_csv(f"{project_root}/best_age_model_bics.csv")


# Global polish (same as before, safe to reuse)
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.2,
})

labels = [
    'RL-WM',
    'Group\nGeCCo',
    'Individual\nGeCCo',
    'Individual\nGeCCo + meta data'
]

x = np.arange(4)
bar_width = 0.65

means = [
    np.mean(baseline_bics['bic']),
    np.mean(group_bics['bic']),
    np.mean(individual_bics['bic']),
    np.mean(individual_age_bics['bic'])
]

sems = [
    np.std(baseline_bics['bic']) / np.sqrt(len(baseline_bics['bic'])),
    np.std(group_bics['bic']) / np.sqrt(len(group_bics['bic'])),
    np.std(individual_bics['bic']) / np.sqrt(len(individual_bics['bic'])),
    np.std(individual_age_bics['bic']) / np.sqrt(len(individual_age_bics['bic']))
]


best_model_color = "#44baeb"
rlwm_model_color = "#368082"
other_colors = "#808080"


figure, axis = plt.subplots(1, 2, figsize=(15, 5))

# ---------- BIC PANEL ----------
axis[0].bar(x, means, width=bar_width, color = [rlwm_model_color,other_colors,best_model_color,other_colors])

axis[0].errorbar(
    x, means, sems,
    fmt='o',
    color='k',
    capsize=4,
    capthick=1.2,
    elinewidth=1.2
)

axis[0].set_xticks(x)
axis[0].set_xticklabels(labels)
axis[0].set_ylabel('BIC')
axis[0].set_ylim(220, 320)
axis[0].grid(axis='y', alpha=0.15)

# ---------- COUNTS PANEL ----------
axis[1].bar(x, counts, width=bar_width, color = [rlwm_model_color,other_colors,best_model_color,other_colors])
axis[1].set_ylabel('# participants\nbest fit by model')
axis[1].set_xticks(x)
axis[1].set_xticklabels(labels)
axis[1].grid(axis='y', alpha=0.15)

# ---------- FINAL TOUCHES ----------
for ax in axis:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

figure.tight_layout()
figure.savefig('bics_comparison.pdf')
