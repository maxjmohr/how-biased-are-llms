CREATE VIEW v_ran_experiments AS
    SELECT
        experiment_id,
        bias_id,
        model_id,
        SUM(correct_run) as correct_ran_loops,
        COUNT(*) AS total_ran_loops,
        MAX(updated_at) AS max_updated_at
    FROM
        t_responses
    GROUP BY
        experiment_id,
        bias_id,
        model_id
;

COMMENT ON VIEW v_ran_experiments IS 'Table to store ran experiments and their properties';

COMMENT ON COLUMN v_ran_experiments.experiment_id IS 'Unique identifier for the experiment';
COMMENT ON COLUMN v_ran_experiments.bias_id IS 'Identifier for the bias';
COMMENT ON COLUMN v_ran_experiments.model_id IS 'Identifier for the model';
COMMENT ON COLUMN v_ran_experiments.correct_ran_loops IS 'Number of loops with correct output ran in the experiment';
COMMENT ON COLUMN v_ran_experiments.total_ran_loops IS 'Total number of loops ran in the experiment';
COMMENT ON COLUMN v_ran_experiments.max_updated_at IS 'Timestamp of the last run';
