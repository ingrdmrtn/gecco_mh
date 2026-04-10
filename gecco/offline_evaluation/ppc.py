"""
Posterior Predictive Checks (PPC) for GeCCo cognitive models.

For every successfully fitted model, compute a set of summary statistics
on both the observed participant data and on data simulated forward from
the fitted parameters.  The resulting records are stored in the diagnostic
store's ``ppc`` table.

Design goals
------------
* **Reuse existing simulation machinery** — we use
  :class:`gecco.parameter_recovery.TaskSimulator` to generate synthetic
  trials.  No parallel simulator is implemented here.
* **Config-driven** — ``n_sims`` and the set of computed statistics are
  controlled via ``cfg.judge.ppc``.
* **Graceful degradation** — if a simulator is not configured, PPC is
  skipped without error.

Output format (list of dicts)
------------------------------
Each record has keys:
    participant_id   — string participant identifier
    statistic_name   — e.g. "choice_proportion_0", "win_stay_rate"
    condition        — optional condition label (str) or None
    observed         — the statistic on real data
    simulated_mean   — mean of the statistic across n_sims simulations
    simulated_q025   — 2.5th percentile across simulations
    simulated_q975   — 97.5th percentile across simulations
    n_sims           — number of forward simulations performed
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


# ======================================================================
# Statistics extractors
# ======================================================================

def _choice_proportions(choices: np.ndarray) -> dict[str, float]:
    """Proportion of each choice value (0, 1, ...) in the array."""
    n = len(choices)
    if n == 0:
        return {}
    unique, counts = np.unique(choices.astype(int), return_counts=True)
    return {f"choice_proportion_{int(u)}": float(c) / n
            for u, c in zip(unique, counts)}


def _win_stay_lose_shift(choices: np.ndarray,
                         rewards: np.ndarray) -> dict[str, float]:
    """Win-stay and lose-shift rates.

    win_stay  = P(choice_t == choice_{t-1} | reward_{t-1} == 1)
    lose_shift = P(choice_t != choice_{t-1} | reward_{t-1} == 0)
    """
    if len(choices) < 2 or len(rewards) < 2:
        return {}
    same = (choices[1:] == choices[:-1]).astype(float)
    prev_reward = rewards[:-1].astype(int)
    win_mask = prev_reward == 1
    lose_mask = prev_reward == 0
    stats = {}
    if win_mask.sum() > 0:
        stats["win_stay_rate"] = float(same[win_mask].mean())
    if lose_mask.sum() > 0:
        stats["lose_shift_rate"] = float(1.0 - same[lose_mask].mean())
    return stats


def _lag1_transition(choices: np.ndarray) -> dict[str, float]:
    """Lag-1 stay probability."""
    if len(choices) < 2:
        return {}
    return {"lag1_stay_prob": float((choices[1:] == choices[:-1]).mean())}


def _per_condition_choice_prop(choices: np.ndarray,
                                condition_col: np.ndarray,
                                col_name: str) -> dict[str, float]:
    """Choice proportions broken down by condition column values."""
    stats = {}
    for cval in np.unique(condition_col):
        mask = condition_col == cval
        sub = choices[mask]
        if len(sub) == 0:
            continue
        prop = float((sub == 1).mean()) if len(np.unique(sub.astype(int))) <= 2 else float(sub.mean())
        stats[f"choice_prop_cond_{col_name}_{cval}"] = prop
    return stats


def _two_step_stay_probabilities(
    choice_1: np.ndarray,
    state: np.ndarray,
    reward: np.ndarray,
) -> dict[str, float]:
    """Two-step stay probabilities and canonical contrasts on the probability scale."""
    if len(choice_1) < 2 or len(state) < 2 or len(reward) < 2:
        return {}

    stay = choice_1[1:] == choice_1[:-1]
    common = (choice_1[:-1] == state[:-1])
    rare = ~common
    rewarded = reward[:-1] == 1
    unrewarded = ~rewarded

    def _mean(mask: np.ndarray) -> float:
        return float(np.mean(stay[mask])) if np.any(mask) else np.nan

    stay_common_rewarded = _mean(common & rewarded)
    stay_common_unrewarded = _mean(common & unrewarded)
    stay_rare_rewarded = _mean(rare & rewarded)
    stay_rare_unrewarded = _mean(rare & unrewarded)
    rewarded_mean = _mean(rewarded)
    unrewarded_mean = _mean(unrewarded)

    return {
        "stay_common_rewarded": stay_common_rewarded,
        "stay_common_unrewarded": stay_common_unrewarded,
        "stay_rare_rewarded": stay_rare_rewarded,
        "stay_rare_unrewarded": stay_rare_unrewarded,
        "reward_effect": rewarded_mean - unrewarded_mean,
        "transition_x_reward": (
            (stay_common_rewarded - stay_common_unrewarded)
            - (stay_rare_rewarded - stay_rare_unrewarded)
        ),
    }


def _compute_statistics(data_dict: dict[str, np.ndarray],
                         input_columns: list[str]) -> dict[str, float]:
    """Compute all applicable statistics from a data dict.

    Parameters
    ----------
    data_dict:
        Maps column name → numpy array of length n_trials.
    input_columns:
        Ordered list of column names for the model.

    Returns
    -------
    dict mapping statistic_name → float value.
    """
    stats: dict[str, float] = {}
    if not input_columns:
        return stats

    # Primary choice column is always the first input column
    primary_choice_col = input_columns[0]
    choices = data_dict.get(primary_choice_col)
    if choices is None:
        return stats

    # Basic choice proportions
    stats.update(_choice_proportions(choices))

    # Lag-1 stay
    stats.update(_lag1_transition(choices))

    # Win-stay / lose-shift (if reward column is present)
    reward_col = next(
        (c for c in input_columns if "reward" in c.lower() or c.lower() == "r"),
        None
    )
    if reward_col and reward_col in data_dict:
        stats.update(_win_stay_lose_shift(choices, data_dict[reward_col]))

    # Per-condition choice proportions (for remaining input columns)
    for col in input_columns[1:]:
        if col == reward_col:
            continue
        col_data = data_dict.get(col)
        if col_data is None:
            continue
        n_unique = len(np.unique(col_data.astype(int)))
        if 2 <= n_unique <= 10:  # Only for low-cardinality columns
            stats.update(_per_condition_choice_prop(choices, col_data.astype(int), col))

    two_step_choice_col = next(
        (col for col in ("choice_1", "action_1") if col in data_dict),
        None,
    )
    if two_step_choice_col and "state" in data_dict and reward_col and reward_col in data_dict:
        stats.update(
            _two_step_stay_probabilities(
                data_dict[two_step_choice_col],
                data_dict["state"],
                data_dict[reward_col],
            )
        )

    return stats


def _get_participants(df) -> tuple[str | None, np.ndarray]:
    """Return the inferred participant ID column and ordered participant IDs."""
    id_col = _find_id_column(df)
    participants = df[id_col].unique() if id_col else df.index.unique()
    return id_col, participants


def _extract_participant_arrays(
    df,
    participant,
    input_columns: list[str],
    id_col: str | None,
) -> tuple[str, dict[str, np.ndarray], list[np.ndarray]] | None:
    """Extract per-participant arrays in both dict and ordered-list form."""
    if id_col:
        df_p = df[df[id_col] == participant].reset_index(drop=True)
    else:
        df_p = df.iloc[[participant]].reset_index(drop=True)

    if len(df_p) == 0:
        return None

    data_dict: dict[str, np.ndarray] = {}
    arrays: list[np.ndarray] = []
    for col in input_columns:
        if col not in df_p.columns:
            return None
        arr = df_p[col].to_numpy()
        data_dict[col] = arr
        arrays.append(arr)

    return str(participant), data_dict, arrays


# ======================================================================
# Forward simulation from fitted parameters
# ======================================================================

def _simulate_forward(
    simulator,
    model_func,
    fitted_params: np.ndarray,
    n_trials: int,
    rng: np.random.Generator,
    input_columns: list[str],
) -> dict[str, np.ndarray]:
    """Run one forward simulation and return a data dict.

    Parameters
    ----------
    simulator:
        A :class:`gecco.parameter_recovery.TaskSimulator` instance.
    model_func:
        Compiled @njit model function.
    fitted_params:
        1-D array of parameter values.
    n_trials:
        Number of trials to simulate.
    rng:
        Random number generator.
    input_columns:
        Ordered list of input column names.

    Returns
    -------
    dict mapping column name → numpy array.
    """
    sim_cols = simulator.simulate_subject(model_func, fitted_params, n_trials, rng=rng)
    sim_col_names = simulator.get_input_columns()
    return {name: arr for name, arr in zip(sim_col_names, sim_cols)}


# ======================================================================
# Main PPC entry point
# ======================================================================

def compute_ppc(
    spec,
    df,
    fitted_params_list: list[list[float]],
    simulator,
    n_sims: int = 100,
    input_columns: list[str] | None = None,
) -> dict:
    """Compute posterior predictive checks for one model.

    Parameters
    ----------
    spec:
        :class:`gecco.offline_evaluation.utils.ModelSpec` with ``func``,
        ``param_names``, and ``bounds``.
    df:
        The behavioural data DataFrame.  Must have an ``id_column``-named
        column for participant identifiers.
    fitted_params_list:
        Per-participant fitted parameters, indexed to match the unique
        participant order in ``df``.
    simulator:
        A :class:`gecco.parameter_recovery.TaskSimulator` instance.
    n_sims:
        Number of forward simulations per participant.
    input_columns:
        Ordered list of input column names.  If None, inferred from the
        simulator.

    Returns
    -------
    dict with key ``"records"`` → list of PPC record dicts.
    """
    if input_columns is None:
        input_columns = simulator.get_input_columns()

    records: list[dict[str, Any]] = []
    id_col, participants = _get_participants(df)

    for p_idx, participant in enumerate(participants):
        if p_idx >= len(fitted_params_list):
            break

        fitted = np.array(fitted_params_list[p_idx], dtype=float)
        if len(fitted) == 0:
            continue

        participant_arrays = _extract_participant_arrays(
            df=df,
            participant=participant,
            input_columns=input_columns,
            id_col=id_col,
        )
        if participant_arrays is None:
            continue

        participant_id, obs_data, arrays = participant_arrays
        n_trials_p = len(arrays[0])

        # --- Compute observed statistics ---
        obs_stats = _compute_statistics(obs_data, input_columns)

        # --- Run n_sims forward simulations ---
        sim_stats_list: list[dict[str, float]] = []
        for sim_i in range(n_sims):
            rng = np.random.default_rng([int(participant) if str(participant).isdigit() else hash(str(participant)) % (2**31), sim_i])
            try:
                sim_data = _simulate_forward(
                    simulator, spec.func, fitted, n_trials_p, rng, input_columns
                )
                sim_stats = _compute_statistics(sim_data, input_columns)
                sim_stats_list.append(sim_stats)
            except Exception as e:
                warnings.warn(f"PPC simulation failed for participant {participant}, sim {sim_i}: {e}")
                continue

        if not sim_stats_list:
            continue

        # --- Aggregate simulated statistics ---
        all_stat_names = set(obs_stats.keys())
        for s in sim_stats_list:
            all_stat_names.update(s.keys())

        for stat_name in all_stat_names:
            obs_val = obs_stats.get(stat_name)
            sim_vals = np.array([s.get(stat_name, np.nan) for s in sim_stats_list])
            valid_sims = sim_vals[np.isfinite(sim_vals)]

            if len(valid_sims) == 0:
                continue

            records.append({
                "participant_id": participant_id,
                "statistic_name": stat_name,
                "condition": None,
                "observed": float(obs_val) if obs_val is not None else None,
                "simulated_mean": float(np.mean(valid_sims)),
                "simulated_q025": float(np.quantile(valid_sims, 0.025)),
                "simulated_q975": float(np.quantile(valid_sims, 0.975)),
                "n_sims": len(valid_sims),
            })

    return {"records": records}


def compute_block_residuals(
    spec,
    df,
    fitted_params_list: list[list[float]],
    n_blocks: int = 10,
    input_columns: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Compute observed-data blockwise NLL residuals for each participant."""
    if input_columns is None:
        id_col = _find_id_column(df)
        input_columns = [col for col in df.columns if col != id_col]

    id_col, participants = _get_participants(df)
    records: list[dict[str, Any]] = []

    for p_idx, participant in enumerate(participants):
        if p_idx >= len(fitted_params_list):
            break

        fitted = np.asarray(fitted_params_list[p_idx], dtype=float)
        if fitted.size == 0:
            continue

        participant_arrays = _extract_participant_arrays(
            df=df,
            participant=participant,
            input_columns=input_columns,
            id_col=id_col,
        )
        if participant_arrays is None:
            continue

        participant_id, _, arrays = participant_arrays
        n_trials = len(arrays[0])
        if n_trials == 0:
            continue

        boundaries = np.linspace(0, n_trials, num=n_blocks + 1, dtype=int)
        prev_nll = 0.0
        for block_idx, (block_start, block_end) in enumerate(
            zip(boundaries[:-1], boundaries[1:])
        ):
            if block_end <= block_start:
                continue

            truncated = [arr[:block_end] for arr in arrays]
            cumulative_nll = float(spec.func(*truncated, fitted))
            block_n_trials = block_end - block_start
            mean_nll = (cumulative_nll - prev_nll) / block_n_trials
            prev_nll = cumulative_nll

            records.append(
                {
                    "participant_id": participant_id,
                    "block_idx": block_idx,
                    "block_start": int(block_start),
                    "block_end": int(block_end),
                    "mean_nll_per_trial": float(mean_nll),
                    "n_trials": int(block_n_trials),
                }
            )

    return {"records": records}


def _find_id_column(df) -> str | None:
    """Heuristically find the participant ID column."""
    for candidate in ("participant", "subject", "id", "subject_id",
                       "participant_id", "subj"):
        if candidate in df.columns:
            return candidate
    return None
