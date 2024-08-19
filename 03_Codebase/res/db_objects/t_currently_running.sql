CREATE TABLE t_currently_running (
    experiment_id   INTEGER NOT NULL,
    system          VARCHAR(255) NOT NULL,
    updated_at      TIMESTAMP,
    PRIMARY KEY (experiment_id)
);

COMMENT ON TABLE t_currently_running IS 'Table to store the currently running experiments';

COMMENT ON COLUMN t_currently_running.experiment_id IS 'Unique identifier for the experiment';
COMMENT ON COLUMN t_currently_running.system IS 'System that is currently running the experiment';
COMMENT ON COLUMN t_currently_running.updated_at IS 'Timestamp of the last update';
