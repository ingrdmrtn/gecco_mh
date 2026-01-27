import numpy as np, pandas as pd, matplotlib.pyplot as plt


age = 1

if age == 1:

    ns3_young = pd.read_csv('ns3_young_group_age.csv').drop(columns="Unnamed: 0")
    ns6_young = pd.read_csv('ns6_young_group_age.csv').drop(columns="Unnamed: 0")

    ns3_old = pd.read_csv('ns3_old_group_age.csv').drop(columns="Unnamed: 0")
    ns6_old = pd.read_csv('ns6_old_group_age.csv').drop(columns="Unnamed: 0")
    save_file = 'individual_age_ppcs.pdf'

else:
    ns3_young = pd.read_csv('ns3_young_individual.csv').drop(columns="Unnamed: 0")
    ns6_young = pd.read_csv('ns6_young_individual.csv').drop(columns="Unnamed: 0")

    ns3_old = pd.read_csv('ns3_old_individual.csv').drop(columns="Unnamed: 0")
    ns6_old = pd.read_csv('ns6_old_individual.csv').drop(columns="Unnamed: 0")
    save_file = 'individual_ppcs.pdf'


ns3_young_real = pd.read_csv('ns3_young_real.csv').drop(columns="Unnamed: 0")
ns6_young_real = pd.read_csv('ns6_young_real.csv').drop(columns="Unnamed: 0")
ns3_old_real = pd.read_csv('ns3_old_real.csv').drop(columns="Unnamed: 0")
ns6_old_real = pd.read_csv('ns6_old_real.csv').drop(columns="Unnamed: 0")

ns3_young_rlwm = pd.read_csv('ns3_young_rlwm.csv').drop(columns="Unnamed: 0")
ns6_young_rlwm = pd.read_csv('ns6_young_rlwm.csv').drop(columns="Unnamed: 0")
ns3_old_rlwm = pd.read_csv('ns3_old_rlwm.csv').drop(columns="Unnamed: 0")
ns6_old_rlwm = pd.read_csv('ns6_old_rlwm.csv').drop(columns="Unnamed: 0")





color_rlwm = "#368082"
color_gecco = "#44baeb"


# Global style tweaks (no color changes)
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.2,
})

figure, axis = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

x = np.arange(9)
err_kw = dict(capsize=3, capthick=1, elinewidth=1.2, alpha=0.9)
marker_kw = dict(marker='o', markersize=5)

# ---------- YOUNG ----------
axis[0].plot(x, np.mean(ns3_young, axis=0),
             color=color_gecco, linewidth=2.5, **marker_kw)
axis[0].errorbar(x,
                 np.mean(ns3_young, axis=0),
                 np.std(ns3_young, axis=0) / np.sqrt(ns3_young.shape[0]),
                 color=color_gecco, fmt='none', **err_kw)

axis[0].plot(x, np.mean(ns6_young, axis=0),
             '--', color=color_gecco, linewidth=2.5)
axis[0].errorbar(x,
                 np.mean(ns6_young, axis=0),
                 np.std(ns6_young, axis=0) / np.sqrt(ns6_young.shape[0]),
                 color=color_gecco, fmt='none', **err_kw)

axis[0].plot(x, np.mean(ns3_young_real, axis=0),
             color='k', linewidth=2.5, **marker_kw)
axis[0].errorbar(x,
                 np.mean(ns3_young_real, axis=0),
                 np.std(ns3_young_real, axis=0) / np.sqrt(ns3_young_real.shape[0]),
                 color='k', fmt='none', **err_kw)

axis[0].plot(x, np.mean(ns6_young_real, axis=0),
             '--', color='k', linewidth=2.5)
axis[0].errorbar(x,
                 np.mean(ns6_young_real, axis=0),
                 np.std(ns6_young_real, axis=0) / np.sqrt(ns6_young_real.shape[0]),
                 color='k', fmt='none', **err_kw)

axis[0].plot(x, np.mean(ns3_young_rlwm, axis=0),
             color=color_rlwm, linewidth=2.5, **marker_kw)
axis[0].errorbar(x,
                 np.mean(ns3_young_rlwm, axis=0),
                 np.std(ns3_young_rlwm, axis=0) / np.sqrt(ns3_young_rlwm.shape[0]),
                 color=color_rlwm, fmt='none', **err_kw)

axis[0].plot(x, np.mean(ns6_young_rlwm, axis=0),
             '--', color=color_rlwm, linewidth=2.5)
axis[0].errorbar(x,
                 np.mean(ns6_young_rlwm, axis=0),
                 np.std(ns6_young_rlwm, axis=0) / np.sqrt(ns6_young_rlwm.shape[0]),
                 color=color_rlwm, fmt='none', **err_kw)

axis[0].set_title("Young")
axis[0].set_xlabel("Stimulus iterations")
axis[0].set_ylabel("p(correct)")
axis[0].set_ylim(0, 1)
axis[0].set_xticks(x)

# ---------- OLD ----------
axis[1].plot(x, np.mean(ns3_old, axis=0),
             color=color_gecco, linewidth=2.5, **marker_kw)
axis[1].errorbar(x,
                 np.mean(ns3_old, axis=0),
                 np.std(ns3_old, axis=0) / np.sqrt(ns3_old.shape[0]),
                 color=color_gecco, fmt='none', **err_kw)

axis[1].plot(x, np.mean(ns6_old, axis=0),
             '--', color=color_gecco, linewidth=2.5)
axis[1].errorbar(x,
                 np.mean(ns6_old, axis=0),
                 np.std(ns6_old, axis=0) / np.sqrt(ns6_old.shape[0]),
                 color=color_gecco, fmt='none', **err_kw)

axis[1].plot(x, np.mean(ns3_old_real, axis=0),
             color='k', linewidth=2.5, **marker_kw)
axis[1].errorbar(x,
                 np.mean(ns3_old_real, axis=0),
                 np.std(ns3_old_real, axis=0) / np.sqrt(ns3_old_real.shape[0]),
                 color='k', fmt='none', **err_kw)

axis[1].plot(x, np.mean(ns6_old_real, axis=0),
             '--', color='k', linewidth=2.5)
axis[1].errorbar(x,
                 np.mean(ns6_old_real, axis=0),
                 np.std(ns6_old_real, axis=0) / np.sqrt(ns6_old_real.shape[0]),
                 color='k', fmt='none', **err_kw)

axis[1].plot(x, np.mean(ns3_old_rlwm, axis=0),
             color=color_rlwm, linewidth=2.5, **marker_kw)
axis[1].errorbar(x,
                 np.mean(ns3_old_rlwm, axis=0),
                 np.std(ns3_old_rlwm, axis=0) / np.sqrt(ns3_old_rlwm.shape[0]),
                 color=color_rlwm, fmt='none', **err_kw)

axis[1].plot(x, np.mean(ns6_old_rlwm, axis=0),
             '--', color=color_rlwm, linewidth=2.5)
axis[1].errorbar(x,
                 np.mean(ns6_old_rlwm, axis=0),
                 np.std(ns6_old_rlwm, axis=0) / np.sqrt(ns6_old_rlwm.shape[0]),
                 color=color_rlwm, fmt='none', **err_kw)

axis[1].set_title("Old")
axis[1].set_xlabel("Stimulus iterations")
axis[1].set_ylim(0, 1)
axis[1].set_xticks(x)

# ---------- FINAL TOUCHES ----------
for ax in axis:
    # ax.grid(axis='y', alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

figure.tight_layout()
figure.savefig(save_file, dpi=300)
print("✨ plot slayed ✨")


figure.savefig(f'{save_file}')
print('stop')
