import numpy as np
import pandas as pd
from src.data.db_helpers import Database
from src.experiments.bias_detection import BiasDetector
from typing import Dict, List, Tuple


def prepare_data(
    levels: List[str] | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | pd.DataFrame | pd.Series:
    """
    Prepare the data for homogeneity calculation
    levels are the levels for which the homogeneity should be computed
    """
    # Connect to the database
    db = Database()
    db.connect()

    # Fetch the effect sizes and their variances
    data: pd.DataFrame = db.fetch_data(total_object="t_bias_detections")

    # If no levels are passed, return total effect sizes and their variances
    if levels is None:
        # Return all effect sizes and their average variances
        return (
            data["bias_detected"].to_numpy(),
            data["sampling_variance"].to_numpy(),
            data["bias_detected_mod"].to_numpy(),
            data["sampling_variance_mod"].to_numpy(),
        )

    else:
        # Group level effect sizes to calculate population effect sizes
        levels_str: str = ", ".join(levels)
        sql: str = f"""
                    SELECT {levels_str}, AVG(bias_detected) AS pop_bias_detected, AVG(bias_detected_mod) AS pop_bias_detected_mod
                    FROM t_bias_detections
                    GROUP BY {levels_str}
                    """
        averages: pd.DataFrame = db.fetch_data(sql=sql)

        # Join the data with the original data
        merged_df: pd.DataFrame = pd.merge(data, averages, on=levels, how="inner")

        # Calculate the sampling variances
        bd = BiasDetector()
        merged_df["sampling_variance"] = merged_df.apply(
            lambda x: bd.sampling_variance(
                sample_size_g1=x["sample_size_1"],
                sample_size_g2=x["sample_size_2"],
                population_effect_size=x["pop_bias_detected"],
            ),
            axis=1,
        )
        merged_df["sampling_variance_mod"] = merged_df.apply(
            lambda x: bd.sampling_variance(
                sample_size_g1=x["sample_size_1"],
                sample_size_g2=x["sample_size_2"],
                population_effect_size=x["pop_bias_detected_mod"],
            ),
            axis=1,
        )
        assert (
            not merged_df.isnull().values.any()
        ), "There are missing values in the data."

        # Return all effect sizes and their average variances
        return_cols: List[str] = levels + [
            "bias_detected",
            "sampling_variance",
            "bias_detected_mod",
            "sampling_variance_mod",
        ]
        return merged_df[return_cols]


def homogeneity_by_HunterSchmidt(
    effect_sizes: np.ndarray, sampling_variances: np.ndarray
) -> float:
    """
    Calculate homogeneity by comparing observed variance to variance due to sampling error
    Hunter and Schmidt (1990) suggest that if the ratio >= 75%, the effect is homogeneous
    """
    # Compute weighted average of individual variances
    # vars_e = len(experiments) / (sum (1 / var_i))
    vars_e = len(sampling_variances) / np.sum(1 / sampling_variances)

    # Observed variance
    # var_d = sum (w_i * (effect_i - mean_effect)^2) / sum(w_i) where w_i = 1 / var_i
    vars_d = np.sum(
        (1 / sampling_variances) * (effect_sizes - np.mean(effect_sizes)) ** 2
    ) / np.sum(1 / sampling_variances)

    # Homogeneity ratio
    return float(vars_e / vars_d) if vars_d != 0 else 1.0


if __name__ == "__main__":
    import inquirer

    # Ask the user for filter options
    print(
        "If you want to compute the homogeneity on a more-detailed level, confirm the desired levels. Otherwise, the homogeneity will be computed across all experiments."
    )
    questions: List = [
        inquirer.Confirm(
            name="bias", message="Compute homogeneity per bias?", default=False
        ),
        inquirer.Confirm(
            name="scenario", message="Compute homogeneity per scenario?", default=False
        ),
        inquirer.Confirm(
            name="model", message="Compute homogeneity per model?", default=False
        ),
        inquirer.Confirm(
            name="temperature",
            message="Compute homogeneity per temperature?",
            default=False,
        ),
    ]
    levels: Dict[str, bool] | None = inquirer.prompt(questions)
    # print(levels)
    # {'bias': False, 'scenario': False, 'model': False, 'temperature': False}
    if levels is None:
        print("No level dictionary provided. Exiting.")
        exit()
    else:
        # Put the relevant levels in a list
        levels_list: List[str] = [key for key, value in levels.items() if value]

    print(
        "Computing homogeneity by Hunter and Schmidt (1990) across all experiments.\n"
    )

    # If all levels are False, return the total homogeneity
    if not any(levels.values()):
        # Fetch and prepare data
        bias_detected, sampling_variances, bias_detected_mod, sampling_variances_mod = (
            prepare_data(levels=None)
        )

        # Calculate homogeneity
        homogeneity: float = homogeneity_by_HunterSchmidt(
            effect_sizes=bias_detected, sampling_variances=sampling_variances
        )
        homegeneity_mod: float = homogeneity_by_HunterSchmidt(
            effect_sizes=bias_detected_mod, sampling_variances=sampling_variances_mod
        )

        print(f"Homogeneity: {homogeneity}")
        print(f"Homogeneity with modified effect size: {homegeneity_mod}")

    # Calculate homogeneity for each level
    else:
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
        for _, row in combinations.iterrows():
            # Get subset
            subset: pd.DataFrame = biases_variances
            for col in levels_list:
                subset = subset.loc[subset[col] == row[col]]

            # Calculate homogeneity
            homogeneity: float = homogeneity_by_HunterSchmidt(
                effect_sizes=subset["bias_detected"].to_numpy(),
                sampling_variances=subset["sampling_variance"].to_numpy(),
            )
            homegeneity_mod: float = homogeneity_by_HunterSchmidt(
                effect_sizes=subset["bias_detected_mod"].to_numpy(),
                sampling_variances=subset["sampling_variance_mod"].to_numpy(),
            )

            # Print
            print(f"\033[1mCombination: {', '.join(row.tolist())}\033[0m")
            print(f"Homogeneity: {homogeneity}")
            print(f"Homogeneity with modified effect size: {homegeneity_mod}")
