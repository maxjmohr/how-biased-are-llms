CREATE VIEW v_responses_grouped AS
    SELECT
        experiment_id,
        bias_id,
        model_id,
        response_type,
        response,
        COUNT(*) AS count,
        MAX(updated_at) AS max_updated_at
    FROM
        t_responses
    GROUP BY
        experiment_id,
        bias_id,
        model_id,
        response_type,
        response
;

COMMENT ON VIEW v_responses_grouped IS 'View to get a quick overview of the distribution of responses for each experiment';

COMMENT ON COLUMN v_responses_grouped.experiment_id IS 'Unique identifier for the experiment';
COMMENT ON COLUMN v_responses_grouped.bias_id IS 'Identifier for the bias';
COMMENT ON COLUMN v_responses_grouped.model_id IS 'Identifier for the model';
COMMENT ON COLUMN v_responses_grouped.response_type IS 'Type of response';
COMMENT ON COLUMN v_responses_grouped.response IS 'Response of the model';
COMMENT ON COLUMN v_responses_grouped.count IS 'Number of responses';
COMMENT ON COLUMN v_responses_grouped.max_updated_at IS 'Timestamp of the last update';
