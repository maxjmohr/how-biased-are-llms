CREATE TABLE t_models (
    model_id    INTEGER NOT NULL,
    model       VARCHAR(255) NOT NULL,
    local       VARCHAR(255) NOT NULL,
    temperature VARCHAR(255) NOT NULL,
    system      VARCHAR(255) NOT NULL,
    updated_at  TIMESTAMP,
    PRIMARY KEY (model_id)
);

COMMENT ON TABLE t_models IS 'Table to store models and their properties';

COMMENT ON COLUMN t_models.model_id IS 'Unique identifier for the model';
COMMENT ON COLUMN t_models.model IS 'Name of the model';
COMMENT ON COLUMN t_models.local IS 'Local model';
COMMENT ON COLUMN t_models.temperature IS 'Temperature of the model';
COMMENT ON COLUMN t_models.system IS 'System of the model';

-- Create models
INSERT INTO t_models (model_id, model, local, temperature, system, updated_at) VALUES
    (10, 'gemma2', 'True', '0.7', 'All', NOW()),
    (15, 'gemma2:27b', 'True', '0.7', 'Linux', NOW()),
    (20, 'gpt-4o-mini', 'False', '0.7', 'All', NOW()),
    (25, 'gpt-4o', 'False', '0.7', 'All', NOW()),
    (30, 'llama3', 'True', '0.7', 'All', NOW()),
    (35, 'llama3:70b', 'True', '0.7', 'Linux', NOW()),
    (40, 'phi3', 'True', '0.7', 'All', NOW()),
    (45, 'phi3:medium', 'True', '0.7', 'All', NOW())
;