CREATE TABLE t_biases (
    bias_id             INTEGER NOT NULL,
    bias                VARCHAR(255) NOT NULL,
    experiment_type     VARCHAR(255) NOT NULL,
    content             VARCHAR(255) NOT NULL,
    variables           VARCHAR(255),
    target_choice       VARCHAR(1) NOT NULL,
    updated_at          TIMESTAMP,
    PRIMARY KEY (bias_id)
);

COMMENT ON TABLE t_biases IS 'Table to store biases and their properties';

COMMENT ON COLUMN t_biases.bias_id IS 'Unique identifier for the bias';
COMMENT ON COLUMN t_biases.bias IS 'Name of the bias';
COMMENT ON COLUMN t_biases.experiment_type IS 'Type of experiment';
COMMENT ON COLUMN t_biases.content IS 'Description of the bias';
COMMENT ON COLUMN t_biases.variables IS 'Variables used in the experiment';
COMMENT ON COLUMN t_biases.target_choice IS 'Target choice for the bias';
COMMENT ON COLUMN t_biases.updated_at IS 'Timestamp of the last update';

-- Create biases
INSERT INTO t_biases (bias_id, bias, experiment_type, content, variables, target_choice, updated_at) VALUES
    (10, 'category_size_bias', 'standard', 'Lets test this and whether it is different to that.', '{"this": 12, "that": 22}', 'A', NOW()),
    (11, 'category_size_bias', 'odd_numbers', 'Lets test this and whether it is different to that.', '{"this": 841758, "that": 1341}', 'B', NOW()),
    (12, 'category_size_bias', 'test', 'Lets test this and whether it is different to that.', NULL, 'A', NOW())
;