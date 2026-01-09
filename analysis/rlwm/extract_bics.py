import os, sys, re, glob, numpy as np, pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path
from config.schema import load_config
from gecco.prepare_data.io import load_data, split_by_participant


project_root = Path(__file__).resolve().parents[2]
cfg = load_config(project_root / "config" / "rlwm_individual.yaml")
data_cfg = cfg.data
df = load_data(data_cfg.path)
participants = df.participant.unique()
df_baseline_bics = [np.array(df[df.participant == p].baseline_bic)[0] for p in participants]
max_iterations = cfg.loop.max_iterations

results_dir = f'{project_root}/results/{cfg.task.name + '_' + cfg.evaluation.fit_type}/bics/'


young_participants = list(df[df.age < 45].participant.unique()[:15])
old_participants = list(df[df.age > 45].participant.unique()[:15])
all_participants = young_participants + old_participants
all_participants = all_participants[:10]

gecco_bics = []
baseline_bics = []
for p in all_participants:
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


    # gecco_bics.append({"participant":p,
    # "min_bic":np.min(min_bic)})
    # hybrid_bics.append({"participant":p,
    # "min_bic":df_hybrid_bics[p]})
    #

    gecco_bics.append(np.min(min_bic))
    baseline_bics.append(df_baseline_bics[p])



gecco_individual_bics = pd.DataFrame({'bic':gecco_bics})
gecco_individual_bics.to_csv('individual_bics.csv')

baseline_bics_df = pd.DataFrame({'bic':baseline_bics})
baseline_bics_df.to_csv('baseline_bics.csv')

print('stop')
