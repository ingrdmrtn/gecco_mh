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
# Composed-path static validation (exact reported return-0.5 model)
# ----------------------------------------------------------------------- #


class TestFitOneOnTestExactReportedModel:
    """fit_one_on_test with the exact reported constant-return model
    (``return 0.5``), WITHOUT mocking the static validator."""

    RETURN_HALF_CANDIDATE_CODE = (
        "def test_model(stimulus, action, reward, params):\n"
        "    return 0.5\n"
    )

    def test_rejects_without_success_metrics(self, mock_df, mock_cfg):
        """A constant-return model must be rejected and expose no success
        metrics, without the test needing to mock validate_likelihood_static."""
        candidate = {
            "display_name": "test_model",
            "function_name": "test_model",
            "code": self.RETURN_HALF_CANDIDATE_CODE,
            "client_id": "client_0",
            "iteration": 5,
            "candidate_index": 2,
            "selection_metric_name": "metric_value",
            "selection_metric_value": 100.0,
            "val_mean_nll": 3.5,
            "param_names": ["alpha"],
        }

        from gecco.cli.run_test_evaluation import fit_one_on_test

        result = fit_one_on_test(candidate, mock_df, mock_cfg)

        assert result is not None
        assert result["model_name"] == "test_model"
        assert result["status"] == "constant_likelihood"
        assert result["error_type"] == "InvalidLikelihoodError"
        # No success metrics
        assert result["test_mean_BIC"] is None
        assert result["test_mean_NLL"] is None
        assert result["test_individual_BIC"] == []
        assert result["test_individual_NLL"] == []


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


# ----------------------------------------------------------------------- #
# Collect-candidates backfill
# ----------------------------------------------------------------------- #


def _make_mock_registry(iteration_history, candidate_generations=None):
    """Build a mock SharedRegistry whose .read() returns the given data."""
    from unittest.mock import MagicMock

    registry = MagicMock()
    registry.read.return_value = {
        "candidate_generations": candidate_generations or {},
        "iteration_history": iteration_history,
    }
    return registry


def _make_cfg(metric="BIC"):
    """Build a minimal mock config for collect_candidates."""
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.evaluation.metric = metric
    return cfg


def _generation_candidate(index, code="def gen_func(a, b): return 0.0",
                          func_name="gen_func", param_names=None):
    """Build a candidate generation entry dict."""
    return {
        "index": index,
        "code": code,
        "func_name": func_name,
        "function_name": func_name,
        "executable_function_name": func_name,
        "param_names": param_names or ["a", "b"],
    }


class TestCollectCandidatesBackfill:
    """collect_candidates backfills missing result code from generated candidates."""

    def test_backfills_code_from_matching_candidate_index(self):
        """When an iteration result lacks code but has a candidate_index
        matching a generated candidate, code/executable_function_name/param_names
        are backfilled."""
        from gecco.cli.run_test_evaluation import collect_candidates

        generation_candidates = [
            _generation_candidate(
                index=2, code="def my_func(x): return x",
                func_name="my_func", param_names=["alpha"]
            ),
        ]
        iteration_history = [
            {
                "client_id": "client_0",
                "iteration": 1,
                "results": [
                    {
                        "candidate_index": 2,
                        # No 'code' key → code is empty string
                        "metric_value": 100.0,
                        "mean_nll": 3.0,
                    },
                ],
            },
        ]
        registry = _make_mock_registry(
            iteration_history,
            {"1": {"candidates": generation_candidates}},
        )
        cfg = _make_cfg("BIC")

        candidates = collect_candidates(registry, cfg)
        assert len(candidates) == 1
        assert candidates[0]["code"] == "def my_func(x): return x"
        assert candidates[0]["candidate_index"] == 2
        assert candidates[0]["param_names"] == ["alpha"]
        # executable_function_name may be resolved differently; at least
        # it should not be empty given the backfill provides func_name
        assert candidates[0]["executable_function_name"]

    def test_does_not_backfill_code_when_index_missing(self):
        """When an iteration result lacks code AND has no candidate_index,
        no backfill occurs and the candidate has empty code."""
        from gecco.cli.run_test_evaluation import collect_candidates

        # No candidate_index and no code in the result
        iteration_history = [
            {
                "client_id": "client_0",
                "iteration": 1,
                "results": [
                    {
                        "metric_value": 100.0,
                        "mean_nll": 3.0,
                    },
                ],
            },
        ]
        registry = _make_mock_registry(iteration_history)
        cfg = _make_cfg("BIC")

        candidates = collect_candidates(registry, cfg)
        assert len(candidates) == 1
        assert candidates[0]["code"] == ""

    def test_does_not_backfill_from_non_matching_candidate(self):
        """When no generated candidate matches the candidate_index,
        code stays empty."""
        from gecco.cli.run_test_evaluation import collect_candidates

        generation_candidates = [
            _generation_candidate(index=99, code="def unrelated(x): return x"),
        ]
        iteration_history = [
            {
                "client_id": "client_0",
                "iteration": 1,
                "results": [
                    {
                        "candidate_index": 2,  # No generated candidate with index 2
                        "metric_value": 100.0,
                        "mean_nll": 3.0,
                    },
                ],
            },
        ]
        registry = _make_mock_registry(
            iteration_history,
            {"1": {"candidates": generation_candidates}},
        )
        cfg = _make_cfg("BIC")

        candidates = collect_candidates(registry, cfg)
        assert len(candidates) == 1
        # Code stays empty because no generated candidate matched index 2
        assert candidates[0]["code"] == ""


# ----------------------------------------------------------------------- #
# Write-store target selection
# ----------------------------------------------------------------------- #


def _fake_candidate():
    """Return a minimal candidate dict for write-store tests."""
    return {
        "client_id": "client_0",
        "iteration": 1,
        "candidate_index": 0,
        "function_name": "test_model",
        "display_name": "test_model",
        "executable_function_name": "test_model",
        "code": "def test_model(a, b): return 0.0",
        "code_hash": "abc123",
        "selection_metric_name": "metric_value",
        "selection_metric_value": 100.0,
        "param_names": ["a", "b"],
    }


def _fake_test_entry():
    """Return a minimal test result entry as returned by fit_one_on_test."""
    return {
        "model_name": "test_model",
        "display_name": "test_model",
        "executable_function_name": "test_model",
        "client_id": "client_0",
        "iteration": 1,
        "candidate_index": 0,
        "selection_metric_name": "metric_value",
        "selection_metric_value": 100.0,
        "val_nll": 3.5,
        "test_mean_BIC": 120.0,
        "test_mean_NLL": 3.8,
        "test_individual_BIC": [120.0],
        "test_individual_NLL": [3.8],
        "code": "def test_model(a, b): return 0.0",
        "param_names": ["a", "b"],
        "status": "ok",
    }


def test_write_store_prefers_unified_db(tmp_path):
    """When diagnostics_unified.duckdb exists, write_store writes test
    rows to it instead of diagnostics.duckdb."""
    from gecco.diagnostic_store.store import DiagnosticStore
    from gecco.cli.run_test_evaluation import run_test_evaluation
    from unittest.mock import patch
    import duckdb

    # Create a dummy shared_registry.duckdb so the exists() check passes
    registry_db = tmp_path / "shared_registry.duckdb"
    duckdb.connect(str(registry_db)).close()

    # Create diagnostics_unified.duckdb
    unified_db = tmp_path / "diagnostics_unified.duckdb"
    store = DiagnosticStore(unified_db)
    store.close()

    with patch("gecco.cli.run_test_evaluation.load_config") as mock_load_cfg:
        with patch("gecco.cli.run_test_evaluation.SharedRegistry") as mock_reg:
            with patch("gecco.cli.run_test_evaluation.load_splits") as mock_splits:
                with patch("gecco.cli.run_test_evaluation.collect_candidates",
                           return_value=[_fake_candidate()]) as mock_collect:
                    with patch("gecco.cli.run_test_evaluation.fit_one_on_test",
                               return_value=_fake_test_entry()) as mock_fit:
                        cfg = mock_load_cfg.return_value
                        cfg.evaluation.n_test_models = 1
                        cfg.evaluation.metric = "BIC"

                        mock_reg_instance = mock_reg.open_existing.return_value
                        mock_reg_instance.read.return_value = {}

                        run_test_evaluation(
                            config="dummy",
                            results_dir=str(tmp_path),
                            write_store=True,
                        )

    # Prove rows were actually written to the unified DB
    unified_conn = duckdb.connect(str(unified_db), read_only=True)
    try:
        count = unified_conn.execute(
            "SELECT COUNT(*) FROM models WHERE split='test'"
        ).fetchone()[0]
        assert count > 0, f"Expected test rows in unified DB, got {count}"
    finally:
        unified_conn.close()

    # Also verify the fallback diagnostics.duckdb was NOT written to
    sibling_db = tmp_path / "diagnostics.duckdb"
    if sibling_db.exists():
        sibling_conn = duckdb.connect(str(sibling_db), read_only=True)
        try:
            sib_count = sibling_conn.execute(
                "SELECT COUNT(*) FROM models WHERE split='test'"
            ).fetchone()[0]
            assert sib_count == 0, (
                f"Expected 0 test rows in sibling diagnostics.duckdb, got {sib_count}"
            )
        finally:
            sibling_conn.close()


def test_write_store_falls_back_to_diagnostics_db(tmp_path):
    """When only diagnostics.duckdb exists (no unified DB),
    write_store writes test rows to diagnostics.duckdb."""
    from gecco.diagnostic_store.store import DiagnosticStore
    from gecco.cli.run_test_evaluation import run_test_evaluation
    from unittest.mock import patch
    import duckdb

    # Create a dummy shared_registry.duckdb
    registry_db = tmp_path / "shared_registry.duckdb"
    duckdb.connect(str(registry_db)).close()

    # Create only diagnostics.duckdb (no unified DB)
    legacy_db = tmp_path / "diagnostics.duckdb"
    store = DiagnosticStore(legacy_db)
    store.close()

    with patch("gecco.cli.run_test_evaluation.load_config") as mock_load_cfg:
        with patch("gecco.cli.run_test_evaluation.SharedRegistry") as mock_reg:
            with patch("gecco.cli.run_test_evaluation.load_splits") as mock_splits:
                with patch("gecco.cli.run_test_evaluation.collect_candidates",
                           return_value=[_fake_candidate()]) as mock_collect:
                    with patch("gecco.cli.run_test_evaluation.fit_one_on_test",
                               return_value=_fake_test_entry()) as mock_fit:
                        cfg = mock_load_cfg.return_value
                        cfg.evaluation.n_test_models = 1
                        cfg.evaluation.metric = "BIC"

                        mock_reg_instance = mock_reg.open_existing.return_value
                        mock_reg_instance.read.return_value = {}

                        run_test_evaluation(
                            config="dummy",
                            results_dir=str(tmp_path),
                            write_store=True,
                        )

    # Prove rows were actually written to the legacy (fallback) DB
    legacy_conn = duckdb.connect(str(legacy_db), read_only=True)
    try:
        count = legacy_conn.execute(
            "SELECT COUNT(*) FROM models WHERE split='test'"
        ).fetchone()[0]
        assert count > 0, f"Expected test rows in diagnostics.duckdb, got {count}"
    finally:
        legacy_conn.close()
