import pandas as pd
from src.analysis.homogeneity import homogeneity_by_HunterSchmidt, prepare_data
from src.data.db_helpers import Database
from typing import List


def compute_detections(
    group_by: List[str] = ["bias", "model"],
) -> pd.DataFrame:
    """
    Fetch data from the database and prepare it for plotting
    """
    # Connect to the database
    db = Database()
    db.connect()

    detected_biases: pd.DataFrame = db.fetch_data(total_object="t_bias_detections")

    # Group by the specified columns and average bias_detected
    relevant_columns: List[str] = group_by + ["bias_detected"]
    detected_biases = (
        detected_biases[relevant_columns].groupby(by=group_by).mean().reset_index()
    )
    # If <0 = 0, if >1 = 1
    detected_biases["bias_detected"] = detected_biases["bias_detected"].apply(
        lambda x: 0 if x < 0 else (1 if x > 1 else x)
    )

    return detected_biases


def compute_homogeneities() -> pd.DataFrame:
    "Compute the homogeneities to compare them later"
    # Define the levels to group by
    levels_list: List[str] = ["bias", "model"]

    # Fetch and prepare data
    biases_variances = prepare_data(levels=levels_list)
    assert isinstance(biases_variances, pd.DataFrame), "Data is not a DataFrame."

    # Get all unique combinations of the actual values of the relevant levels
    # Connect to the database
    db = Database()
    db.connect()
    levels_cols: str = ", ".join(levels_list)
    sql: str = f"""
                SELECT {levels_cols}
                FROM t_bias_detections
                GROUP BY {levels_cols}
                ORDER BY {levels_cols}
                """
    combinations: pd.DataFrame = db.fetch_data(sql=sql)

    # Get subset of each level_cols combinations and calculate homogeneity per subset
    results: pd.DataFrame = pd.DataFrame()
    for _, row in combinations.iterrows():
        # Get subset
        subset: pd.DataFrame = biases_variances
        for col in levels_list:
            subset = subset.loc[subset[col] == row[col]]

        # Calculate homogeneity
        homegeneity_mod: float = homogeneity_by_HunterSchmidt(
            effect_sizes=subset["bias_detected_mod"].to_numpy(),
            sampling_variances=subset["sampling_variance_mod"].to_numpy(),
        )

        # Concate to results
        append_row: pd.DataFrame = (
            pd.Series(
                {
                    **row,
                    "homogeneity_mod": homegeneity_mod,
                }
            )
            .to_frame()
            .T
        )
        results = pd.concat([results, append_row], axis=0)

    return results


def join_detections_homogeneities(
    detected_biases: pd.DataFrame, homogeneities: pd.DataFrame
) -> pd.DataFrame:
    "Join the detected biases and homogeneities"
    join: pd.DataFrame = detected_biases.merge(homogeneities, on=["bias", "model"])

    # Add column detected_bias + homogeneity (for later sorting)
    join["combined"] = join["bias_detected"] + join["homogeneity_mod"]

    return join


def filter(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the data where 'bias_detected' > 0.5 and 'homogeneity_mod' > 0.5
    """
    return data[
        ((data["bias_detected"] > 0.3) & (data["homogeneity_mod"] > 0.3))
        | ((data["bias_detected"] < 0.3) & (data["homogeneity_mod"] < 0.3))
    ].sort_values(
        by=["combined", "bias", "model"],
        ascending=[False, True, True],
    )[["bias", "model", "bias_detected", "homogeneity_mod"]]


if __name__ == "__main__":
    # Compute the detected biases and homogeneities
    detected_biases: pd.DataFrame = compute_detections()
    homogeneities: pd.DataFrame = compute_homogeneities()

    # Join the detected biases and homogeneities
    joined_data: pd.DataFrame = join_detections_homogeneities(
        detected_biases=detected_biases, homogeneities=homogeneities
    )

    # Filter the data
    filtered_data: pd.DataFrame = filter(data=joined_data)

    print(filtered_data)
