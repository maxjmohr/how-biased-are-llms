CREATE TABLE t_responses (
    experiment_id   INTEGER NOT NULL,
    bias_id         INTEGER NOT NULL,
    model_id        INTEGER NOT NULL,
    response_type   VARCHAR(255) NOT NULL,
    response        VARCHAR(255) NOT NULL,
    reason          VARCHAR(255),
    correct_run     INTEGER NOT NULL,
    updated_at      TIMESTAMP,
    FOREIGN KEY (bias_id) REFERENCES t_biases(bias_id),
    FOREIGN KEY (model_id) REFERENCES t_models(model_id)
);

COMMENT ON TABLE t_responses IS 'Table to store the model responses to the ran experiments';

COMMENT ON COLUMN t_responses.experiment_id IS 'Unique identifier for the experiment';
COMMENT ON COLUMN t_responses.bias_id IS 'Identifier for the bias';
COMMENT ON COLUMN t_responses.model_id IS 'Identifier for the model';
COMMENT ON COLUMN t_responses.response_type IS 'Type of response';
COMMENT ON COLUMN t_responses.response IS 'Response of the model';
COMMENT ON COLUMN t_responses.reason IS 'Reason for the response';
COMMENT ON COLUMN t_responses.correct_run IS 'Whether the model output is correct';
COMMENT ON COLUMN t_responses.updated_at IS 'Timestamp of the last update';
