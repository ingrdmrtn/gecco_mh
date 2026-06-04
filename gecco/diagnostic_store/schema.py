"""
DuckDB DDL for the GeCCo diagnostic store.

All tables are keyed by (run_idx, iteration, model_name).  A synthetic
integer ``model_id`` acts as the primary key for the ``models`` table and
is referenced by all child tables.

Call :func:`create_schema` on a fresh DuckDB connection to initialise the
database.
"""

SCHEMA_VERSION = 2

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
        mean_nll       DOUBLE,
        split          VARCHAR DEFAULT 'train',
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
        nll             DOUBLE,
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
        per_param_detail JSON,
        split            VARCHAR DEFAULT 'train'
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
    # ------------------------------------------------------------------ #
    # Runtime coordination state (DuckDB is canonical)
    # ------------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS runtime_global_best (
        singleton     INTEGER PRIMARY KEY DEFAULT 1,
        metric_value  DOUBLE,
        model_code    TEXT,
        param_names   JSON,
        client_id     VARCHAR,
        iteration     INTEGER,
        CHECK (singleton = 1)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runtime_baseline (
        singleton      INTEGER PRIMARY KEY DEFAULT 1,
        function_name  VARCHAR,
        metric_name    VARCHAR,
        metric_value   DOUBLE,
        param_names    JSON,
        eval_metrics   JSON,
        mean_r2        DOUBLE,
        max_r2         DOUBLE,
        best_param     VARCHAR,
        per_param_r2   JSON,
        code           TEXT,
        val_mean_nll   DOUBLE,
        CHECK (singleton = 1)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runtime_client_entries (
        client_id            VARCHAR PRIMARY KEY,
        last_iteration       INTEGER,
        best_metric          DOUBLE,
        status               VARCHAR,
        updated_at           VARCHAR,
        had_runnable_model   BOOLEAN,
        activity             VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runtime_iteration_history (
        client_id    VARCHAR NOT NULL,
        iteration    INTEGER NOT NULL,
        results      JSON,
        created_at   VARCHAR,
        updated_at   VARCHAR,
        PRIMARY KEY (client_id, iteration)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runtime_tried_param_sets (
        param_key   VARCHAR PRIMARY KEY,
        param_set   JSON
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runtime_judge_iterations (
        iteration               INTEGER PRIMARY KEY,
        synthesized_feedback    JSON,
        verdict                 JSON,
        failed                  BOOLEAN DEFAULT FALSE,
        error                   TEXT,
        timestamp               VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runtime_candidate_generations (
        iteration      INTEGER PRIMARY KEY,
        candidates     JSON,
        generated_by   VARCHAR,
        timestamp      VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runtime_generator_status (
        iteration      INTEGER PRIMARY KEY,
        client_id      VARCHAR,
        status         VARCHAR,
        n_candidates   INTEGER,
        error          TEXT,
        updated_at     VARCHAR
    )
    """,
    # ------------------------------------------------------------------ #
    # Runtime views
    # ------------------------------------------------------------------ #
    """
    CREATE OR REPLACE VIEW runtime_status_view AS
    SELECT
        client_id,
        status,
        last_iteration,
        best_metric,
        had_runnable_model,
        activity,
        updated_at
    FROM runtime_client_entries
    ORDER BY client_id
    """,
    """
    CREATE OR REPLACE VIEW runtime_coordination_view AS
    WITH iterations AS (
        SELECT iteration FROM runtime_iteration_history
        UNION
        SELECT iteration FROM runtime_candidate_generations
        UNION
        SELECT iteration FROM runtime_generator_status
        UNION
        SELECT iteration FROM runtime_judge_iterations
    ),
    history_counts AS (
        SELECT
            iteration,
            COUNT(*) AS n_client_results,
            COALESCE(SUM(json_array_length(results)), 0) AS n_models_reported
        FROM runtime_iteration_history
        GROUP BY iteration
    ),
    client_counts AS (
        SELECT
            h.iteration,
            COUNT(*) FILTER (
                WHERE c.status IN ('complete', 'complete_no_success')
            ) AS n_clients_complete,
            COUNT(*) FILTER (
                WHERE COALESCE(c.had_runnable_model, FALSE)
            ) AS n_clients_with_models
        FROM runtime_iteration_history h
        LEFT JOIN runtime_client_entries c ON c.client_id = h.client_id
        GROUP BY h.iteration
    )
    SELECT
        i.iteration,
        COALESCE(h.n_client_results, 0) AS n_client_results,
        COALESCE(h.n_models_reported, 0) AS n_models_reported,
        COALESCE(c.n_clients_complete, 0) AS n_clients_complete,
        COALESCE(c.n_clients_with_models, 0) AS n_clients_with_models,
        g.generated_by,
        COALESCE(json_array_length(g.candidates), 0) AS n_candidates,
        gs.status AS generator_status,
        CASE
            WHEN j.iteration IS NULL THEN FALSE
            ELSE TRUE
        END AS has_judge_feedback,
        COALESCE(j.failed, FALSE) AS judge_failed,
        j.timestamp AS judge_timestamp
    FROM iterations i
    LEFT JOIN history_counts h ON h.iteration = i.iteration
    LEFT JOIN client_counts c ON c.iteration = i.iteration
    LEFT JOIN runtime_candidate_generations g ON g.iteration = i.iteration
    LEFT JOIN runtime_generator_status gs ON gs.iteration = i.iteration
    LEFT JOIN runtime_judge_iterations j ON j.iteration = i.iteration
    ORDER BY i.iteration
    """,
]


def create_schema(conn) -> None:
    """Initialise all tables in *conn* (idempotent)."""
    for stmt in CREATE_STATEMENTS:
        conn.execute(stmt.strip())

    version_row = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
    if version_row == 0:
        conn.execute("INSERT INTO schema_version VALUES (?)", [SCHEMA_VERSION])
    else:
        conn.execute("UPDATE schema_version SET version = ?", [SCHEMA_VERSION])
