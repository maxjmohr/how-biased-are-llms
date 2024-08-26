CREATE VIEW v_experiments AS
WITH
    base AS (
        SELECT
            (b.bias_id * 100 + m.model_id) AS experiment_id,
            b.bias_id,
            m.model_id,
            b.bias,
            b.experiment_type,
            b.scenario,
            m.model,
            m.local,
            m.temperature,
            m.system,
            b.content,
            b.variables,
            b.response_type,
            b.target_response,
            b.part,
            b.parts_total
        FROM
            t_biases b
            CROSS JOIN t_models m
    ),

    ran_experiments AS (
        SELECT
            re.experiment_id,
            re.max_updated_at AS ran_date,
            re.correct_ran_loops,
            re.total_ran_loops
        FROM
            v_ran_experiments re
    )

    SELECT
        b.*,
        re.ran_date,
        COALESCE(re.correct_ran_loops, 0) AS correct_ran_loops,
        COALESCE(re.total_ran_loops, 0) AS total_ran_loops,
        NOW() AS updated_at
    FROM
        base b
        LEFT JOIN ran_experiments re ON re.experiment_id = b.experiment_id
;

COMMENT ON VIEW v_experiments IS 'View to store all experiments and their properties';

COMMENT ON COLUMN v_experiments.experiment_id IS 'Unique identifier for the experiment';
COMMENT ON COLUMN v_experiments.bias_id IS 'Identifier for the bias';
COMMENT ON COLUMN v_experiments.model_id IS 'Identifier for the model';
COMMENT ON COLUMN v_experiments.bias IS 'Name of the bias';
COMMENT ON COLUMN v_experiments.experiment_type IS 'Type of experiment';
COMMENT ON COLUMN v_experiments.scenario IS 'Scenario of the experiment';
COMMENT ON COLUMN v_experiments.model IS 'Name of the model';
COMMENT ON COLUMN v_experiments.local IS 'Local or global model';
COMMENT ON COLUMN v_experiments.temperature IS 'Temperature of the model';
COMMENT ON COLUMN v_experiments.system IS 'System of the model';
COMMENT ON COLUMN v_experiments.content IS 'Description of the bias';
COMMENT ON COLUMN v_experiments.variables IS 'Variables used in the experiment';
COMMENT ON COLUMN v_experiments.response_type IS 'Type of response';
COMMENT ON COLUMN v_experiments.target_response IS 'Target response from studies';
COMMENT ON COLUMN v_experiments.part IS 'Which part of a multi-part experiment is this';
COMMENT ON COLUMN v_experiments.parts_total IS 'Total number of parts in a multi-part experiment';
COMMENT ON COLUMN v_experiments.ran_date IS 'Date when the experiment was ran';
COMMENT ON COLUMN v_experiments.correct_ran_loops IS 'Number of loops with correct output ran in the experiment';
COMMENT ON COLUMN v_experiments.total_ran_loops IS 'Total number of loops ran in the experiment';
COMMENT ON COLUMN v_experiments.updated_at IS 'Timestamp of the last update';
