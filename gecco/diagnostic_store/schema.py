"""
DuckDB DDL for the GeCCo diagnostic store.

All tables are keyed by (run_idx, iteration, model_name).  A synthetic
integer ``model_id`` acts as the primary key for the ``models`` table and
is referenced by all child tables.

Call :func:`create_schema` on a fresh DuckDB connection to initialise the
database.
"""

SCHEMA_VERSION = 1

CREATE_STATEMENTS = [
    # ------------------------------------------------------------------ #
    # Schema version tracker
    # ------------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version  INTEGER NOT NULL
    )
    """,

    # ------------------------------------------------------------------ #
    # Iterations — one row per (run_idx, iteration)
    # ------------------------------------------------------------------ #
    """
    CREATE SEQUENCE IF NOT EXISTS iterations_id_seq START 1
    """,
    """
    CREATE TABLE IF NOT EXISTS iterations (
        iteration_id       INTEGER DEFAULT nextval('iterations_id_seq') PRIMARY KEY,
        run_idx            INTEGER NOT NULL,
        iteration          INTEGER NOT NULL,
        client_id          VARCHAR,
        tag                VARCHAR,
        timestamp          VARCHAR,
        n_models_proposed  INTEGER DEFAULT 0,
        UNIQUE (run_idx, iteration, tag)
    )
    """,

    # ------------------------------------------------------------------ #
    # Models — one row per evaluated model candidate
    # ------------------------------------------------------------------ #
    """
    CREATE SEQUENCE IF NOT EXISTS models_id_seq START 1
    """,
    """
    CREATE TABLE IF NOT EXISTS models (
        model_id       INTEGER DEFAULT nextval('models_id_seq') PRIMARY KEY,
        iteration_id   INTEGER REFERENCES iterations(iteration_id),
        run_idx        INTEGER NOT NULL,
        iteration      INTEGER NOT NULL,
        name           VARCHAR NOT NULL,
        code           TEXT,
        metric_name    VARCHAR,
        metric_value   DOUBLE,
        param_names    JSON,
        status         VARCHAR DEFAULT 'ok'
    )
    """,

    # ------------------------------------------------------------------ #
    # Per-participant fit data
    # ------------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS model_participants (
        id              INTEGER PRIMARY KEY,
        model_id        INTEGER REFERENCES models(model_id),
        participant_idx INTEGER NOT NULL,
        bic             DOUBLE,
        n_trials        INTEGER,
        params          JSON
    )
    """,
    """
    CREATE SEQUENCE IF NOT EXISTS model_participants_id_seq START 1
    """,
    # Replace the table to add the default for id
    # (DuckDB doesn't support ALTER COLUMN DEFAULT easily, so we use a sequence
    # inline at insert time via populate.py)

    # ------------------------------------------------------------------ #
    # Parameter recovery
    # ------------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS parameter_recovery (
        model_id          INTEGER PRIMARY KEY REFERENCES models(model_id),
        passed            BOOLEAN,
        mean_r            DOUBLE,
        n_successful      INTEGER,
        per_param_r       JSON,
        simulation_error  VARCHAR
    )
    """,

    # ------------------------------------------------------------------ #
    # Individual differences
    # ------------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS individual_differences (
        model_id         INTEGER PRIMARY KEY REFERENCES models(model_id),
        mean_r2          DOUBLE,
        max_r2           DOUBLE,
        best_param       VARCHAR,
        per_param_r2     JSON,
        per_param_detail JSON
    )
    """,

    # ------------------------------------------------------------------ #
    # Posterior predictive checks
    # ------------------------------------------------------------------ #
    """
    CREATE SEQUENCE IF NOT EXISTS ppc_id_seq START 1
    """,
    """
    CREATE TABLE IF NOT EXISTS ppc (
        ppc_id           INTEGER DEFAULT nextval('ppc_id_seq') PRIMARY KEY,
        model_id         INTEGER REFERENCES models(model_id),
        participant_id   VARCHAR,
        statistic_name   VARCHAR NOT NULL,
        condition        VARCHAR,
        observed         DOUBLE,
        simulated_mean   DOUBLE,
        simulated_q025   DOUBLE,
        simulated_q975   DOUBLE,
        n_sims           INTEGER
    )
    """,

    # ------------------------------------------------------------------ #
    # Block residuals
    # ------------------------------------------------------------------ #
    """
    CREATE SEQUENCE IF NOT EXISTS block_res_id_seq START 1
    """,
    """
    CREATE TABLE IF NOT EXISTS block_residuals (
        id                  INTEGER DEFAULT nextval('block_res_id_seq') PRIMARY KEY,
        model_id            INTEGER REFERENCES models(model_id),
        participant_id      VARCHAR,
        block_idx           INTEGER NOT NULL,
        block_start         INTEGER,
        block_end           INTEGER,
        mean_nll_per_trial  DOUBLE,
        n_trials            INTEGER
    )
    """,

    # ------------------------------------------------------------------ #
    # Validation errors
    # ------------------------------------------------------------------ #
    """
    CREATE SEQUENCE IF NOT EXISTS validation_errors_id_seq START 1
    """,
    """
    CREATE TABLE IF NOT EXISTS validation_errors (
        error_id       INTEGER DEFAULT nextval('validation_errors_id_seq') PRIMARY KEY,
        model_id       INTEGER REFERENCES models(model_id),
        error_type     VARCHAR,
        error_message  TEXT,
        error_details  JSON
    )
    """,
]


def create_schema(conn) -> None:
    """Initialise all tables in *conn* (idempotent)."""
    for stmt in CREATE_STATEMENTS:
        conn.execute(stmt.strip())

    # Insert schema version if not already present
    version_row = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
    if version_row == 0:
        conn.execute(
            "INSERT INTO schema_version VALUES (?)", [SCHEMA_VERSION]
        )
