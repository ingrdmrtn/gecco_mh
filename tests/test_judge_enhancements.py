import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gecco.diagnostic_store.store import DiagnosticStore
from gecco.diagnostic_store.tools import (
    get_block_residuals,
    get_participant_best_models,
)
from gecco.offline_evaluation.ppc import (
    _two_step_stay_probabilities,
    compute_block_residuals,
    compute_ppc,
)


class DummyTwoStepSimulator:
    def __init__(self, arrays):
        self.arrays = arrays

    def simulate_subject(self, model_func, fitted_params, n_trials, rng=None):
        return [arr[:n_trials].copy() for arr in self.arrays]

    def get_input_columns(self):
        return ["choice_1", "state", "reward"]


def test_two_step_stay_probabilities_and_ppc_integration():
    choice_1 = np.array([0, 0, 1, 1, 0])
    state = np.array([0, 0, 0, 0, 1])
    reward = np.array([1, 0, 1, 0, 0])

    stats = _two_step_stay_probabilities(choice_1, state, reward)

    assert stats == {
        "stay_common_rewarded": 1.0,
        "stay_common_unrewarded": 0.0,
        "stay_rare_rewarded": 1.0,
        "stay_rare_unrewarded": 0.0,
        "reward_effect": 1.0,
        "transition_x_reward": 0.0,
    }

    df = pd.DataFrame(
        {
            "participant": ["p1"] * len(choice_1),
            "choice_1": choice_1,
            "state": state,
            "reward": reward,
        }
    )
    simulator = DummyTwoStepSimulator([choice_1, state, reward])
    spec = SimpleNamespace(func=lambda *args: 0.0)

    ppc = compute_ppc(
        spec=spec,
        df=df,
        fitted_params_list=[[0.1]],
        simulator=simulator,
        n_sims=3,
        input_columns=["choice_1", "state", "reward"],
    )

    records = {record["statistic_name"]: record for record in ppc["records"]}
    assert "stay_common_rewarded" in records
    assert "reward_effect" in records
    assert records["stay_common_rewarded"]["observed"] == pytest.approx(1.0)
    assert records["stay_common_rewarded"]["simulated_mean"] == pytest.approx(1.0)
    assert records["reward_effect"]["observed"] == pytest.approx(1.0)


def test_compute_block_residuals_uses_incremental_nll_by_block():
    def cumulative_model(choice, model_parameters):
        return float(np.sum(choice) + model_parameters[0] * len(choice))

    spec = SimpleNamespace(func=cumulative_model)
    df = pd.DataFrame(
        {
            "participant": ["p1"] * 4,
            "choice": [0, 1, 1, 1],
        }
    )

    residuals = compute_block_residuals(
        spec=spec,
        df=df,
        fitted_params_list=[[0.5]],
        n_blocks=2,
        input_columns=["choice"],
    )

    assert residuals["records"] == [
        {
            "participant_id": "p1",
            "block_idx": 0,
            "block_start": 0,
            "block_end": 2,
            "mean_nll_per_trial": 1.0,
            "n_trials": 2,
        },
        {
            "participant_id": "p1",
            "block_idx": 1,
            "block_start": 2,
            "block_end": 4,
            "mean_nll_per_trial": 1.5,
            "n_trials": 2,
        },
    ]


def test_store_tools_expose_block_residuals_and_participant_best_models(tmp_path):
    store = DiagnosticStore(tmp_path / "diagnostics.duckdb")
    try:
        store.write_iteration(
            iteration=0,
            run_idx=0,
            iteration_results=[
                {
                    "function_name": "model_a",
                    "metric_name": "BIC",
                    "metric_value": 15.0,
                    "param_names": ["alpha"],
                    "code": "def model_a():\n    return 0",
                    "eval_metrics": [10.0, 20.0],
                    "participant_n_trials": [100, 100],
                    "parameter_values": [[0.1], [0.2]],
                    "block_residuals": {
                        "records": [
                            {
                                "participant_id": "0",
                                "block_idx": 0,
                                "block_start": 0,
                                "block_end": 10,
                                "mean_nll_per_trial": 1.0,
                                "n_trials": 10,
                            },
                            {
                                "participant_id": "1",
                                "block_idx": 0,
                                "block_start": 0,
                                "block_end": 10,
                                "mean_nll_per_trial": 1.5,
                                "n_trials": 10,
                            },
                            {
                                "participant_id": "0",
                                "block_idx": 1,
                                "block_start": 10,
                                "block_end": 20,
                                "mean_nll_per_trial": 2.0,
                                "n_trials": 10,
                            },
                            {
                                "participant_id": "1",
                                "block_idx": 1,
                                "block_start": 10,
                                "block_end": 20,
                                "mean_nll_per_trial": 2.5,
                                "n_trials": 10,
                            },
                        ]
                    },
                },
                {
                    "function_name": "model_b",
                    "metric_name": "BIC",
                    "metric_value": 16.0,
                    "param_names": ["alpha"],
                    "code": "def model_b():\n    return 0",
                    "eval_metrics": [12.0, 18.0],
                    "participant_n_trials": [100, 100],
                    "parameter_values": [[0.3], [0.4]],
                },
            ],
        )

        model_a_id = store.fetchone(
            "SELECT model_id FROM models WHERE name = ?",
            ["model_a"],
        )["model_id"]

        block_summary = get_block_residuals(store, model_a_id)
        assert block_summary["n_participants"] == 2
        assert len(block_summary["blocks"]) == 2
        assert block_summary["blocks"][0]["mean_nll_per_trial_mean"] == pytest.approx(1.25)
        assert block_summary["blocks"][0]["mean_nll_per_trial_std"] == pytest.approx(math.sqrt(0.125))
        assert block_summary["blocks"][1]["mean_nll_per_trial_mean"] == pytest.approx(2.25)

        best_models = get_participant_best_models(store)
        assert best_models["participants"] == [
            {
                "participant_idx": 0,
                "best_model_id": model_a_id,
                "best_model_name": "model_a",
                "best_bic": 10.0,
                "iteration": 0,
                "run_idx": 0,
            },
            {
                "participant_idx": 1,
                "best_model_id": store.fetchone(
                    "SELECT model_id FROM models WHERE name = ?",
                    ["model_b"],
                )["model_id"],
                "best_model_name": "model_b",
                "best_bic": 18.0,
                "iteration": 0,
                "run_idx": 0,
            },
        ]
        assert best_models["summary"]["n_participants"] == 2
        assert best_models["summary"]["n_unique_models"] == 2
        assert best_models["summary"]["modal_model"] == "model_a"
        assert best_models["summary"]["heterogeneity_index"] == pytest.approx(0.5)
    finally:
        store.close()