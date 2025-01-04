"This is a script to fetch the relevant datasets for transparency and verifiability of the thesis"

import os
import pandas as pd
from src.data.db_helpers import Database

if __name__ == "__main__":
    # Connect to database
    db = Database()
    db.connect()

    # Fetch master data (biases, models and experiments)
    sql: str = """
        SELECT
            *
        FROM
            t_biases
        ORDER BY
            bias_id
        ;
        """
    biases: pd.DataFrame = db.fetch_data(sql=sql)

    sql: str = """
        SELECT
            *
        FROM
            t_models
        ORDER BY
            model_id
        ;
        """
    models: pd.DataFrame = db.fetch_data(sql=sql)

    sql: str = """
        SELECT
            *
        FROM
            v_experiments
        ORDER BY
            experiment_id
        ;
        """
    experiments: pd.DataFrame = db.fetch_data(sql=sql)

    # Fetch all responses
    sql: str = """
        SELECT
            experiment_id,
            bias_id,
            model_id,
            response_type,
            response,
            updated_at
        FROM
            t_responses
        ;
        """
    responses: pd.DataFrame = db.fetch_data(sql=sql)

    # Fetch the bias detections
    sql: str = """
        SELECT
            bias,
            scenario,
            model,
            temperature,
            sample_size_1,
            sample_size_2,
            bias_detected,
            bias_detected_mod
        FROM
            t_bias_detections
        ORDER BY
            bias,
            scenario,
            model,
            temperature
        ;
        """
    bias_detections: pd.DataFrame = db.fetch_data(sql=sql)

    # Write all to csv in target directory ../../../04_Datasets
    current_path = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(current_path, "./../../../04_Datasets/")

    biases.to_csv(target_path + "biases.csv", index=False)
    models.to_csv(target_path + "models.csv", index=False)
    experiments.to_csv(target_path + "experiments.csv", index=False)
    responses.to_csv(target_path + "responses.csv", index=False)
    bias_detections.to_csv(target_path + "bias_detections.csv", index=False)

    # Disconnect from the database
    db.disconnect()
