"""Contract tests for DiagnosticStore write operations, especially
write_top_model_test with invalid entries and trial counts."""

from __future__ import annotations

from pathlib import Path

import pytest

from gecco.diagnostic_store.store import DiagnosticStore


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def store(tmp_path: Path) -> DiagnosticStore:
    """A fresh DiagnosticStore with proper schema."""
    db_path = tmp_path / "test.duckdb"
    s = DiagnosticStore(db_path)
    return s


# --------------------------------------------------------------------------- #
# write_top_model_test — invalid entries (Finding E.4)
# --------------------------------------------------------------------------- #


class TestWriteTopModelTestInvalid:
    """write_top_model_test must persist invalid audit rows and ensure
    valid predicates (status='ok') exclude them from best-model queries."""

    def test_writes_static_invalid_entry(self, store: DiagnosticStore):
        """A statically invalid entry is persisted with non-ok status
        and validation error details."""
        entry = {
            "model_name": "invalid_model",
            "val_nll": None,
            "test_mean_BIC": None,
            "test_mean_NLL": None,
            "test_individual_BIC": [],
            "test_individual_NLL": [],
            "test_individual_NLL_trials": [],
            "test_individual_differences": None,
            "code": "def invalid_model():\n    return 0.0",
            "param_names": [],
            "status": "constant_likelihood",
            "error_type": "InvalidLikelihoodError",
            "error_message": "Model returns constant 0.0",
            "error_details": {"reason": "constant_likelihood", "snippet": "return 0.0"},
        }
        store.write_top_model_test(entry)

        # Verify model row persisted with non-ok status
        model = store.fetchone(
            "SELECT name, status, metric_value, metric_name, code FROM models"
        )
        assert model is not None
        assert model["name"] == "invalid_model"
        assert model["status"] == "constant_likelihood"
        assert model["metric_value"] is None  # cleared for invalid entries
        assert model["code"] == entry["code"]

        # Verify validation error persisted
        verr = store.fetchone(
            "SELECT error_type, error_message, error_details FROM validation_errors"
        )
        assert verr is not None
        assert verr["error_type"] == "InvalidLikelihoodError"
        assert "constant_likelihood" in (verr["error_details"] or "")

    def test_writes_post_fit_invalid_entry(self, store: DiagnosticStore):
        """A post-fit invalid entry persists with cleared success metrics
        but preserved audit details."""
        entry = {
            "model_name": "zero_nll_model",
            "val_nll": None,
            "test_mean_BIC": None,
            "test_mean_NLL": None,
            "test_individual_BIC": [],
            "test_individual_NLL": [],
            "test_individual_NLL_trials": [100],
            "test_individual_differences": None,
            "code": "def zero_nll_model():\n    return 0.0",
            "param_names": ["alpha"],
            "status": "degenerate_nll",
            "error_type": "InvalidLikelihoodError",
            "error_message": "All zero NLLs",
            "error_details": {
                "reason": "degenerate_nll",
                "per_participant_nll": [0.0],
            },
        }
        store.write_top_model_test(entry)

        model = store.fetchone("SELECT name, status, metric_value FROM models")
        assert model is not None
        assert model["status"] == "degenerate_nll"
        assert model["metric_value"] is None

        # Verify validation_errors row
        verr = store.fetchone("SELECT error_type FROM validation_errors")
        assert verr is not None
        assert verr["error_type"] == "InvalidLikelihoodError"

    def test_invalid_entries_excluded_from_ok_queries(self, store: DiagnosticStore):
        """Invalid entries with non-ok status must not appear in
        status='ok' queries."""
        # Write a valid entry
        store.write_top_model_test({
            "model_name": "valid_model",
            "val_nll": 3.0,
            "test_mean_BIC": 100.0,
            "test_mean_NLL": 4.0,
            "test_individual_BIC": [100.0],
            "test_individual_NLL": [4.0],
            "test_individual_NLL_trials": [50],
            "test_individual_differences": None,
            "code": "def valid_model():\n    pass",
            "param_names": ["alpha"],
            "status": "ok",
        })

        # Write an invalid entry
        store.write_top_model_test({
            "model_name": "invalid_model",
            "val_nll": None,
            "test_mean_BIC": None,
            "test_mean_NLL": None,
            "test_individual_BIC": [],
            "test_individual_NLL": [],
            "test_individual_NLL_trials": [],
            "test_individual_differences": None,
            "code": "def invalid_model():\n    return 0.0",
            "param_names": [],
            "status": "choice_leakage",
            "error_type": "InvalidLikelihoodError",
            "error_message": "Choice leakage",
            "error_details": {"reason": "choice_leakage"},
        })

        # valid-only query should return only the valid model
        ok_models = store.fetchall(
            "SELECT name, status FROM models WHERE status='ok'"
        )
        assert len(ok_models) == 1
        assert ok_models[0]["name"] == "valid_model"

        # All rows query should return both
        all_models = store.fetchall("SELECT name, status FROM models ORDER BY name")
        assert len(all_models) == 2


# --------------------------------------------------------------------------- #
# Trial counts persistence (Finding F)
# --------------------------------------------------------------------------- #


class TestWriteTopModelTestTrialCounts:
    """write_top_model_test must persist trial counts in participant rows."""

    def test_persists_trial_counts_for_valid_entry(self, store: DiagnosticStore):
        """Participant rows for a valid entry should include trial counts."""
        store.write_top_model_test({
            "model_name": "test_model",
            "val_nll": 3.0,
            "test_mean_BIC": 100.0,
            "test_mean_NLL": 4.0,
            "test_individual_BIC": [100.0, 110.0],
            "test_individual_NLL": [4.0, 4.5],
            "test_individual_NLL_trials": [50, 60],
            "test_individual_differences": None,
            "code": "def test_model():\n    pass",
            "param_names": ["alpha"],
            "status": "ok",
        })

        participants = store.fetchall(
            "SELECT participant_idx, bic, nll, n_trials "
            "FROM model_participants ORDER BY participant_idx"
        )
        assert len(participants) == 2

        assert participants[0]["participant_idx"] == 0
        assert participants[0]["bic"] == 100.0
        assert participants[0]["nll"] == 4.0
        assert participants[0]["n_trials"] == 50

        assert participants[1]["participant_idx"] == 1
        assert participants[1]["bic"] == 110.0
        assert participants[1]["nll"] == 4.5
        assert participants[1]["n_trials"] == 60

    def test_persists_trial_counts_for_invalid_entry(self, store: DiagnosticStore):
        """Participant rows for an invalid entry should include trial counts
        when available."""
        store.write_top_model_test({
            "model_name": "bad_model",
            "val_nll": None,
            "test_mean_BIC": None,
            "test_mean_NLL": None,
            "test_individual_BIC": [],
            "test_individual_NLL": [],
            "test_individual_NLL_trials": [100, 200],
            "test_individual_differences": None,
            "code": "def bad_model():\n    return 0.0",
            "param_names": [],
            "status": "degenerate_nll",
            "error_type": "InvalidLikelihoodError",
            "error_message": "zero NLL",
            "error_details": {"reason": "degenerate_nll"},
        })

        participants = store.fetchall(
            "SELECT participant_idx, n_trials FROM model_participants ORDER BY participant_idx"
        )
        # Even with empty BIC/NLL lists, participants should be written
        # based on trial counts when available
        assert len(participants) > 0
        # If trial counts list is non-empty, participants are created
        # from max(len(bic), len(nll), len(trials))
        assert len(participants) == 2
        assert participants[0]["n_trials"] == 100
        assert participants[1]["n_trials"] == 200
