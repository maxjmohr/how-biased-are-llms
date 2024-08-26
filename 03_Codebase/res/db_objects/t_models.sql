CREATE TABLE t_models (
    model_id            INTEGER NOT NULL,
    model               VARCHAR(255) NOT NULL,
    local               VARCHAR(255) NOT NULL,
    temperature         VARCHAR(255) NOT NULL,
    system              VARCHAR(255) NOT NULL,
    release_date        DATE,
    last_updated_at     DATE,
    download_date       DATE,
    size                DECIMAL(10, 2),
    number_parameters   INTEGER,
    model_architecture  VARCHAR(255),
    ollama_id           VARCHAR(255),
    updated_at          TIMESTAMP,
    PRIMARY KEY (model_id)
);

COMMENT ON TABLE t_models IS 'Table to store models and their properties';

COMMENT ON COLUMN t_models.model_id IS 'Unique identifier for the model';
COMMENT ON COLUMN t_models.model IS 'Name of the model';
COMMENT ON COLUMN t_models.local IS 'Local model';
COMMENT ON COLUMN t_models.temperature IS 'Temperature of the model';
COMMENT ON COLUMN t_models.system IS 'System of the model';
COMMENT ON COLUMN t_models.release_date IS 'Release date of the model';
COMMENT ON COLUMN t_models.last_updated_at IS 'Last updated date of the model';
COMMENT ON COLUMN t_models.download_date IS 'Personal download date of the model';
COMMENT ON COLUMN t_models.size IS 'Size of the model in GB';
COMMENT ON COLUMN t_models.number_parameters IS 'Number of parameters of the model in billions';
COMMENT ON COLUMN t_models.model_architecture IS 'Architecture of the model';
COMMENT ON COLUMN t_models.ollama_id IS 'Identifier of the model in the OLLAMA database';
COMMENT ON COLUMN t_models.updated_at IS 'Timestamp of the last update of the row';
