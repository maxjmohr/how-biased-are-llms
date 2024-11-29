import numpy as np
import pandas as pd
from src.data.db_helpers import Database
from typing import Tuple


def prepare_data(modified: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare the data for homogeneity calculation
    """
    # Connect to the database
    db = Database()
    db.connect()

    # Fetch the effect sizes and their variances
    data: pd.DataFrame = db.fetch_data(total_object="t_bias_detections")

    # Extract effect sizes and their variances
    effect_sizes = data[f"bias_detected{'_mod' if modified else ''}"].to_numpy()
    sampling_variances = data[
        f"sampling_variance{'_mod' if modified else ''}"
    ].to_numpy()

    # Return the data
    return effect_sizes, sampling_variances


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
    return float(vars_e / vars_d)


if __name__ == "__main__":
    # Fetch and prepare data
    bias_detected, sampling_variances = prepare_data(modified=False)
    bias_detected_mod, sampling_variances_mod = prepare_data(modified=True)

    # Calculate homogeneity
    homogeneity: float = homogeneity_by_HunterSchmidt(
        effect_sizes=bias_detected, sampling_variances=sampling_variances
    )
    homegeneity_mod: float = homogeneity_by_HunterSchmidt(
        effect_sizes=bias_detected_mod, sampling_variances=sampling_variances_mod
    )

    print(f"Homogeneity by Hunter and Schmidt (1990): {homogeneity}")
    print(
        f"Homogeneity by Hunter and Schmidt (1990) with modified effect size: {homegeneity_mod}"
    )
