CREATE TABLE t_ran_experiments (
    experiment_id       INTEGER NOT NULL,
    bias_id             INTEGER NOT NULL,
    model_id            INTEGER NOT NULL,
    ran_date            TIMESTAMP NOT NULL,
    correct_ran_loops   INTEGER NOT NULL,
    total_ran_loops     INTEGER NOT NULL,
    updated_at          TIMESTAMP,
    PRIMARY KEY (experiment_id, ran_date),
    FOREIGN KEY (bias_id) REFERENCES t_biases(bias_id),
    FOREIGN KEY (model_id) REFERENCES t_models(model_id)
);

COMMENT ON TABLE t_ran_experiments IS 'Table to store ran experiments and their properties';

COMMENT ON COLUMN t_ran_experiments.experiment_id IS 'Unique identifier for the experiment';
COMMENT ON COLUMN t_ran_experiments.bias_id IS 'Identifier for the bias';
COMMENT ON COLUMN t_ran_experiments.model_id IS 'Identifier for the model';
COMMENT ON COLUMN t_ran_experiments.ran_date IS 'Date when the experiment was ran';
COMMENT ON COLUMN t_ran_experiments.correct_ran_loops IS 'Number of loops with correct output ran in the experiment';
COMMENT ON COLUMN t_ran_experiments.total_ran_loops IS 'Total number of loops ran in the experiment';
COMMENT ON COLUMN t_ran_experiments.updated_at IS 'Timestamp of the last update';
