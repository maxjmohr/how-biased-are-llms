CREATE TABLE t_biases (
    bias_id             INTEGER NOT NULL,
    bias                VARCHAR(255) NOT NULL,
    experiment_type     VARCHAR(255) NOT NULL,
    scenario            VARCHAR(255) NOT NULL,
    content             VARCHAR(255) NOT NULL,
    variables           VARCHAR(255),
    response_type       VARCHAR(255) NOT NULL,
    target_response     VARCHAR(255),
    part                INTEGER NOT NULL,
    parts_total         INTEGER NOT NULL,
    updated_at          TIMESTAMP,
    PRIMARY KEY (bias_id)
);

COMMENT ON TABLE t_biases IS 'Table to store biases and their properties';

COMMENT ON COLUMN t_biases.bias_id IS 'Unique identifier for the bias';
COMMENT ON COLUMN t_biases.bias IS 'Name of the bias';
COMMENT ON COLUMN t_biases.experiment_type IS 'Type of experiment (choice vs. free, comparison via different scenarios, ...)';
COMMENT ON COLUMN t_biases.scenario IS 'Scenario of the experiment';
COMMENT ON COLUMN t_biases.content IS 'Description of the bias';
COMMENT ON COLUMN t_biases.variables IS 'Variables used in the experiment';
COMMENT ON COLUMN t_biases.response_type IS 'Type of response (A, B, C, or e.g. price)';
COMMENT ON COLUMN t_biases.target_response IS 'Target response from studies';
COMMENT ON COLUMN t_biases.part IS 'Which part of a multi-part experiment is this';
COMMENT ON COLUMN t_biases.parts_total IS 'Total number of parts in a multi-part experiment';
COMMENT ON COLUMN t_biases.updated_at IS 'Timestamp of the last update';

-- Create biases
INSERT INTO t_biases (bias_id, bias, experiment_type, scenario, content, variables, response_type, target_response, part, parts_total, updated_at) VALUES
    (100, 'category size bias', 'multi-scenario', 'normal', 'Lets test this and whether it is different to that.', '{"this": 12, "that": 22}', 'choice', 'A', 1, 2, NOW()),
    (101, 'category size bias', 'multi-scenario', 'odd numbers', 'Lets test this and whether it is different to that.', '{"this": 841758, "that": 1341}', 'choice', 'B', 2, 2, NOW()),
    (110, 'category size bias', 'one-scenario', 'odd numbers', 'Lets test this and whether it is different to that.', NULL, 'free', '100', 1, 1, NOW())
;