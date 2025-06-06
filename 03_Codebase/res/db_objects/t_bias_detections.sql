CREATE TABLE t_bias_detections (
    bias                    VARCHAR(255) NOT NULL,
    scenario                VARCHAR(255) NOT NULL,
    model                   VARCHAR(255) NOT NULL,
    temperature             VARCHAR(255) NOT NULL,
    model_id                INTEGER NOT NULL,
    sample_size_1           INTEGER NOT NULL,
    sample_size_2           INTEGER NOT NULL,
    bias_detected           NUMERIC NOT NULL,
    sampling_variance       NUMERIC NOT NULL,
    bias_detected_mod       NUMERIC,
    sampling_variance_mod   NUMERIC,
    updated_at              TIMESTAMP,
    PRIMARY KEY (bias, scenario, model, temperature),
    FOREIGN KEY (model_id) REFERENCES t_models(model_id)
);

COMMENT ON TABLE t_bias_detections IS 'Table to store the binary target variable bias_detected for each bias and scenario for each model';

COMMENT ON COLUMN t_bias_detections.bias IS 'Name of the bias';
COMMENT ON COLUMN t_bias_detections.scenario IS 'Scenario of the experiment';
COMMENT ON COLUMN t_bias_detections.model IS 'Name of the model';
COMMENT ON COLUMN t_bias_detections.temperature IS 'Temperature of the experiment';
COMMENT ON COLUMN t_bias_detections.model_id IS 'Unique identifier for the model';
COMMENT ON COLUMN t_bias_detections.sample_size_1 IS 'Sample size of the first group';
COMMENT ON COLUMN t_bias_detections.sample_size_2 IS 'Sample size of the second group';
COMMENT ON COLUMN t_bias_detections.bias_detected IS 'Target variable indicating whether the bias was detected';
COMMENT ON COLUMN t_bias_detections.sampling_variance IS 'Sampling variance of the effect size';
COMMENT ON COLUMN t_bias_detections.bias_detected_mod IS 'Binary, modified target variable';
COMMENT ON COLUMN t_bias_detections.sampling_variance_mod IS 'Modified sampling variance of the effect size';
COMMENT ON COLUMN t_bias_detections.updated_at IS 'Timestamp of the last update';
