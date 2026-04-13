import pytest
from gecco.diagnostic_store import DiagnosticStore
from gecco.diagnostic_store.tools import (
    DiagnosticNotAvailableError,
    get_recovery,
    get_ppc,
    get_individual_differences,
    get_block_residuals,
    dispatch_tool,
)


@pytest.fixture
def empty_store(tmp_path):
    """A DiagnosticStore with schema but no diagnostic data."""
    db_path = tmp_path / "test.duckdb"
    # Create the store, which will initialize tables
    store = DiagnosticStore(db_path)
    # Ensure models table exists for _check_model_exists, ignore if it already exists
    try:
        store.execute(
            "CREATE TABLE models (model_id INTEGER PRIMARY KEY, status TEXT, metric_value REAL)"
        )
    except Exception:
        pass
    return store


class TestDiagnosticNotAvailableErrors:
    """All four diagnostic tools should raise DiagnosticNotAvailableError
    when the relevant table is empty, and ValueError for unknown model IDs."""

    def test_get_recovery_no_data(self, empty_store):
        empty_store.execute(
            "INSERT INTO models (model_id, run_idx, iteration, status, name) VALUES (999, 0, 0, 'ok', 'model999')"
        )
        with pytest.raises(
            DiagnosticNotAvailableError, match="parameter_recovery.enabled"
        ):
            get_recovery(empty_store, model_id=999)

    def test_get_ppc_no_data(self, empty_store):
        empty_store.execute(
            "INSERT INTO models (model_id, run_idx, iteration, status, name) VALUES (999, 0, 0, 'ok', 'model999')"
        )
        with pytest.raises(DiagnosticNotAvailableError, match="judge.ppc.enabled"):
            get_ppc(empty_store, model_id=999)

    def test_get_individual_differences_no_data(self, empty_store):
        empty_store.execute(
            "INSERT INTO models (model_id, run_idx, iteration, status, name) VALUES (999, 0, 0, 'ok', 'model999')"
        )
        with pytest.raises(DiagnosticNotAvailableError, match="covariates"):
            get_individual_differences(empty_store, model_id=999)

    def test_get_block_residuals_no_data(self, empty_store):
        empty_store.execute(
            "INSERT INTO models (model_id, run_idx, iteration, status, name) VALUES (999, 0, 0, 'ok', 'model999')"
        )
        with pytest.raises(
            DiagnosticNotAvailableError, match="block_residuals.enabled"
        ):
            get_block_residuals(empty_store, model_id=999)


class TestModelExistenceCheck:
    """Tools should raise ValueError for model IDs that don't exist
    in the models table, distinct from diagnostic-not-available errors."""

    def test_get_recovery_unknown_model(self, empty_store):
        with pytest.raises(ValueError, match="does not exist"):
            get_recovery(empty_store, model_id=999)

    def test_get_ppc_unknown_model(self, empty_store):
        with pytest.raises(ValueError, match="does not exist"):
            get_ppc(empty_store, model_id=999)


class TestDispatchToolIntegration:
    """dispatch_tool should catch DiagnosticNotAvailableError and
    return it as an error dict."""

    def test_dispatch_catches_diagnostic_error(self, empty_store):
        empty_store.execute(
            "INSERT INTO models (model_id, run_idx, iteration, status, name) VALUES (999, 0, 0, 'ok', 'model999')"
        )
        result = dispatch_tool(empty_store, "get_recovery", {"model_id": 999})
        assert "error" in result
        assert "parameter_recovery.enabled" in result["error"]

    def test_dispatch_catches_unknown_model_error(self, empty_store):
        result = dispatch_tool(empty_store, "get_recovery", {"model_id": 999})
        # ValueError for unknown model is caught too
        assert "error" in result
        assert "does not exist" in result["error"]
