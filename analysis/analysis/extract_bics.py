import os, sys, re, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
df = pd.read_csv(f"{project_root}/rlwm.csv")
participants = df.participant.unique()

age = 1
if age == 1:
    results_dir = f'{project_root}/results/rlwm_individual_age/bics/'
    save_file = "individual_age_bics.csv"
else:
    results_dir = f'{project_root}/results/rlwm_individual/bics/'
    save_file = "individual_bics.csv"


all_participants = np.array(pd.read_csv(f'{project_root}/ids.csv')['ids'])

df_baseline_bics = [np.array(df[df.participant == p].baseline_bic)[0] for p in all_participants]


gecco_bics = []
baseline_bics = []
for p_idx, p in enumerate(all_participants):
    print(p)

    files = glob.glob(f'{results_dir}iter*_run0_participant{p}.json')

    if not files:
        gecco_bics.append([])
        continue

    max_iter = max(
        int(re.search(r'iter(\d+)_', os.path.basename(f)).group(1))
        for f in files
    )

    min_bic = [
        (
            np.min(df['metric_value'].values)
            if not df.empty and 'metric_value' in df
            else 1000
        )
        for i in range(max_iter + 1)
        for df in (
            [pd.read_json(f'{results_dir}iter{i}_run0_participant{p}.json')]
            if os.path.exists(f'{results_dir}iter{i}_run0_participant{p}.json')
            else [pd.DataFrame()]
        )
    ]



    gecco_bics.append(np.min(min_bic))
    baseline_bics.append(df_baseline_bics[p_idx])


gecco_individual_bics = pd.DataFrame({'bic':gecco_bics})
gecco_individual_bics.to_csv(f'{save_file}')

baseline_bics_df = pd.DataFrame({'bic':baseline_bics})
baseline_bics_df.to_csv('baseline_bics.csv')

