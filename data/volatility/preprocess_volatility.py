"""Prepare volatility task data from the reference paper repo for GeCCo.

The source repository stores one CSV per participant-task.  This script
combines those files into a single trial-level table with the columns expected
by the GeCCo config for this task.
"""

from __future__ import annotations

import argparse
import runpy
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "reference_repos" / "volatility_paper_elife-master" / "data"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "volatility"


def _participant_and_task(path: Path, exp: int) -> tuple[str, str]:
    name = path.name
    if exp == 1:
        match = re.search(r"behavioral_trial_table_(.+?)_(pain|rew)_modelready", name)
        if not match:
            raise ValueError(f"Could not parse exp1 filename: {name}")
        participant = match.group(1)
        task = "pain" if match.group(2) == "pain" else "reward"
    else:
        match = re.search(r"behavioral_tablehit_batch_(gain|loss)_mturk(.+?)_modelready", name)
        if not match:
            raise ValueError(f"Could not parse exp2 filename: {name}")
        participant = f"mturk{match.group(2)}"
        task = "pain" if match.group(1) == "loss" else "reward"
    return participant, task


def _load_one(path: Path, exp: int) -> pd.DataFrame:
    participant, task = _participant_and_task(path, exp)
    df = pd.read_csv(path)
    out = pd.DataFrame(index=df.index)
    out["participant"] = participant
    out["participant_task"] = f"{participant}_{task}"
    out["task"] = task
    out["task_order"] = 0 if task == "reward" else 1
    out["trial_in_task"] = range(len(df))
    out["trial"] = out["task_order"] * 10000 + out["trial_in_task"]
    out["choice"] = df["choice"].astype(float)
    out["green_outcome"] = df["green_outcome"].astype(float)
    out["green_mag"] = df["green_mag"].astype(float)
    out["blue_mag"] = df["blue_mag"].astype(float)
    out["rt"] = df["rt"].astype(float)
    stable = df["block"].map({"stable": 1.0, "volatile": 0.0}).astype(float)
    out["stable"] = stable
    out["volatile"] = 1.0 - stable
    out["run"] = df["run"].astype(float) if "run" in df.columns else 0.0
    out["reward_task"] = 1.0 if task == "reward" else 0.0
    out["pain_task"] = 1.0 - out["reward_task"]
    out["green_good_outcome"] = (
        out["green_outcome"] if task == "reward" else 1.0 - out["green_outcome"]
    )
    out["reset"] = 0.0
    out.loc[out["trial_in_task"] == 0, "reset"] = 1.0
    return out


def _load_paper_exclusions(source_root: Path) -> dict[str, set[str]]:
    exclusion_path = source_root.parent / "data_processing_code" / "exclude.py"
    if not exclusion_path.exists():
        raise FileNotFoundError(exclusion_path)
    namespace = runpy.run_path(exclusion_path)
    return {
        "pain": set(namespace["EXCLUDE_PAIN"]),
        "reward": set(namespace["EXCLUDE_REW"]),
    }


def build_dataset(
    exp: int,
    source_root: Path = DEFAULT_SOURCE,
    apply_exclusions: bool = True,
) -> pd.DataFrame:
    raw_dir = source_root / f"data_raw_exp{exp}"
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)
    files = sorted(raw_dir.glob("*modelready.csv"))
    if not files:
        raise FileNotFoundError(f"No modelready CSVs found in {raw_dir}")
    exclusions = (
        _load_paper_exclusions(source_root)
        if apply_exclusions
        else {"pain": set(), "reward": set()}
    )
    frames = []
    for path in files:
        participant, task = _participant_and_task(path, exp)
        if participant in exclusions[task]:
            continue
        frames.append(_load_one(path, exp))
    if not frames:
        raise ValueError(f"No data files remained after exclusions for exp{exp}")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["choice"]).copy()
    df["choice"] = df["choice"].astype(int)
    # Recompute reset: first remaining trial of each participant_task gets reset=1,
    # in case the original first trial was dropped due to NaN choice.
    df["reset"] = 0.0
    first_idx = df.groupby("participant_task").head(1).index
    df.loc[first_idx, "reset"] = 1.0
    df = df.sort_values(["participant", "task_order", "trial_in_task"]).reset_index(
        drop=True
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, choices=[1, 2], default=1)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-apply-exclusions",
        action="store_true",
        help="Include participant-task files excluded by the paper's preprocessing.",
    )
    args = parser.parse_args()

    df = build_dataset(
        args.exp,
        args.source_root,
        apply_exclusions=not args.no_apply_exclusions,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"volatility_exp{args.exp}_modelready.csv"
    df.to_csv(out_path, index=False)
    n_participants = df["participant"].nunique()
    n_participant_tasks = df["participant_task"].nunique()
    task_counts = df.groupby("task")["participant"].nunique().to_dict()
    print(
        f"Wrote {out_path} with {len(df)} rows, "
        f"{n_participants} participants, {n_participant_tasks} participant-tasks "
        f"({task_counts})."
    )


if __name__ == "__main__":
    main()
