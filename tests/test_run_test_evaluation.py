"""Contract tests for run_test_evaluation.fit_one_on_test.

All tests use mocks to avoid expensive model fitting.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from gecco.offline_evaluation.likelihood_validation import ValidationResult


# ----------------------------------------------------------------------- #
# Fixtures
# ----------------------------------------------------------------------- #


@pytest.fixture
def candidate():
    """A minimal candidate dict with required fields."""
    return {
        "display_name": "test_model",
        "function_name": "test_model",
        "code": "def test_model(stimulus, action, reward, params):\n    return 0.0",
        "client_id": "client_0",
        "iteration": 5,
        "candidate_index": 2,
        "selection_metric_name": "metric_value",
        "selection_metric_value": 100.0,
        "val_mean_nll": 3.5,
        "param_names": ["alpha"],
    }


@pytest.fixture
def mock_df():
    """A mock dataframe."""
    return MagicMock()


@pytest.fixture
def mock_cfg():
    """A mock config with minimal attributes."""
    cfg = MagicMock()
    cfg.evaluation.metric = "BIC"
    # No individual_differences_eval to keep things simple
    return cfg


# ----------------------------------------------------------------------- #
# Static validation failure path
# ----------------------------------------------------------------------- #


class TestFitOneOnTestStaticInvalid:
    """fit_one_on_test with statically-invalid model code."""

    def test_returns_error_dict(self, candidate, mock_df, mock_cfg):
        """Static validation failure returns a dict with error_type='InvalidLikelihoodError'."""
        with patch(
            "gecco.offline_evaluation.likelihood_validation.validate_likelihood_static",
            return_value=ValidationResult(
                passed=False,
                error_type="choice_leakage",
                error_message="Choice leakage detected",
                error_details={"func_name": "test_model"},
            ),
        ):
            from gecco.cli.run_test_evaluation import fit_one_on_test

            result = fit_one_on_test(candidate, mock_df, mock_cfg)

        assert result is not None
        assert result["model_name"] == "test_model"
        assert result["status"] == "choice_leakage"
        assert result["error_type"] == "InvalidLikelihoodError"
        assert "Choice leakage detected" in result["error_message"]
        assert result["error_details"]["reason"] == "choice_leakage"

    def test_does_not_expose_success_metrics(self, candidate, mock_df, mock_cfg):
        """Invalid entries must not have test_mean_BIC / test_mean_NLL set."""
        with patch(
            "gecco.offline_evaluation.likelihood_validation.validate_likelihood_static",
            return_value=ValidationResult(
                passed=False,
                error_type="constant_likelihood",
                error_message="Constant zero NLL",
                error_details={},
            ),
        ):
            from gecco.cli.run_test_evaluation import fit_one_on_test

            result = fit_one_on_test(candidate, mock_df, mock_cfg)

        assert result["test_mean_BIC"] is None
        assert result["test_mean_NLL"] is None
        assert result["test_individual_BIC"] == []
        assert result["test_individual_NLL"] == []
        assert result["test_individual_differences"] is None


# ----------------------------------------------------------------------- #
# Fit exception path
# ----------------------------------------------------------------------- #


class TestFitOneOnTestFitError:
    """fit_one_on_test when run_fit raises an exception."""

    def test_returns_fit_error(self, candidate, mock_df, mock_cfg):
        """When run_fit raises, status='fit_error' and error_type='fit_error'."""
        with patch(
            "gecco.offline_evaluation.likelihood_validation.validate_likelihood_static",
            return_value=ValidationResult(passed=True),
        ):
            with patch(
                "gecco.cli.run_test_evaluation.run_fit",
                side_effect=ValueError("Fitting crashed"),
            ):
                from gecco.cli.run_test_evaluation import fit_one_on_test

                result = fit_one_on_test(candidate, mock_df, mock_cfg)

        assert result is not None
        assert result["status"] == "fit_error"
        assert result["error_type"] == "fit_error"
        assert "Fitting crashed" in result["error_message"]
        assert result["test_mean_BIC"] is None
        assert result["test_mean_NLL"] is None


# ----------------------------------------------------------------------- #
# Post-fit validation failure path
# ----------------------------------------------------------------------- #


class TestFitOneOnTestPostFitInvalid:
    """fit_one_on_test when post-fit validation fails (zero/tiny NLL)."""

    def test_rejects_zero_nll(self, candidate, mock_df, mock_cfg):
        """Zero per-participant NLL leads to InvalidLikelihoodError."""
        fit_res = {
            "metric_value": 0.0,
            "mean_nll": 0.0,
            "eval_metrics": [0.0],
            "per_participant_nll": [0.0, 0.0],
            "participant_n_trials": [10, 10],
        }

        with patch(
            "gecco.offline_evaluation.likelihood_validation.validate_likelihood_static",
            return_value=ValidationResult(passed=True),
        ):
            with patch(
                "gecco.cli.run_test_evaluation.run_fit",
                return_value=fit_res,
            ):
                from gecco.cli.run_test_evaluation import fit_one_on_test

                result = fit_one_on_test(candidate, mock_df, mock_cfg)

        assert result is not None
        assert result["status"] == "degenerate_nll"
        assert result["error_type"] == "InvalidLikelihoodError"
        assert result["test_mean_BIC"] is None
        assert result["test_mean_NLL"] is None

    def test_has_reason_in_details(self, candidate, mock_df, mock_cfg):
        """Post-fit failure includes reason/category in error_details."""
        fit_res = {
            "metric_value": 0.0,
            "mean_nll": 0.0,
            "eval_metrics": [0.0],
            "per_participant_nll": [0.0, 0.0],
            "participant_n_trials": [10, 10],
        }

        with patch(
            "gecco.offline_evaluation.likelihood_validation.validate_likelihood_static",
            return_value=ValidationResult(passed=True),
        ):
            with patch(
                "gecco.cli.run_test_evaluation.run_fit",
                return_value=fit_res,
            ):
                from gecco.cli.run_test_evaluation import fit_one_on_test

                result = fit_one_on_test(candidate, mock_df, mock_cfg)

        assert result["error_details"]["reason"] == "degenerate_nll"


# ----------------------------------------------------------------------- #
# Success path
# ----------------------------------------------------------------------- #


class TestFitOneOnTestSuccess:
    """fit_one_on_test with a valid fit."""

    def test_returns_ok_entry(self, candidate, mock_df, mock_cfg):
        """Successful fit returns entry with status='ok' and test metrics."""
        fit_res = {
            "metric_value": 120.0,
            "mean_nll": 3.8,
            "eval_metrics": [120.0],
            "per_participant_nll": [3.8],
            "participant_n_trials": [50],
        }

        with patch(
            "gecco.offline_evaluation.likelihood_validation.validate_likelihood_static",
            return_value=ValidationResult(passed=True),
        ):
            with patch(
                "gecco.offline_evaluation.likelihood_validation.validate_likelihood_post_fit",
                return_value=ValidationResult(passed=True),
            ):
                with patch(
                    "gecco.cli.run_test_evaluation.run_fit",
                    return_value=fit_res,
                ):
                    from gecco.cli.run_test_evaluation import fit_one_on_test

                    result = fit_one_on_test(candidate, mock_df, mock_cfg)

        assert result is not None
        assert result["status"] == "ok"
        assert result.get("error_type") is None
        assert result["test_mean_BIC"] == 120.0
        assert result["test_mean_NLL"] == 3.8
        assert result["test_individual_BIC"] == [120.0]
        assert result["test_individual_NLL"] == [3.8]
        assert result["model_name"] == "test_model"


# ----------------------------------------------------------------------- #
# Missing code
# ----------------------------------------------------------------------- #


class TestFitOneOnTestMissingCode:
    """fit_one_on_test with missing code returns None."""

    def test_returns_none_when_code_empty(self, candidate, mock_df, mock_cfg):
        candidate["code"] = ""
        from gecco.cli.run_test_evaluation import fit_one_on_test

        result = fit_one_on_test(candidate, mock_df, mock_cfg)
        assert result is None
