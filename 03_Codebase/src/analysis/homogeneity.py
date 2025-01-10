import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.data.db_helpers import Database
from src.experiments.bias_detection import BiasDetector
from typing import Dict, List, Literal, Tuple


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
        assert not merged_df.isnull().values.any(), (
            "There are missing values in the data."
        )

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
    homogeneity_ratio = float(vars_e / vars_d) if vars_d != 0 else 1.0
    return homogeneity_ratio if homogeneity_ratio <= 1.0 else 1.0


def get_homogeneity_one_level(level: Literal["bias", "model"] = "bias") -> pd.DataFrame:
    """
    Calculate homogeneity for one level (bias or model) to display at the far right and bottom of the heatmap
    """
    # Fetch and prepare data
    biases_variances = prepare_data(levels=[level])
    assert isinstance(biases_variances, pd.DataFrame), "Data is not a DataFrame."

    # Get all unique combinations of the actual values of the relevant levels
    # Connect to the database
    db = Database()
    db.connect()

    # Get unique values of the level
    sql: str = f"""
                SELECT DISTINCT {level}
                FROM t_bias_detections
                """
    unique_values: pd.DataFrame = db.fetch_data(sql=sql)

    # Get subset of each level_cols combinations and calculate homogeneity per subset
    results: pd.DataFrame = pd.DataFrame()
    for _, row in unique_values.iterrows():
        # Get subset
        subset: pd.DataFrame = biases_variances
        subset = subset.loc[subset[level] == row[level]]

        # Calculate homogeneity
        homogeneity: float = homogeneity_by_HunterSchmidt(
            effect_sizes=subset["bias_detected"].to_numpy(),
            sampling_variances=subset["sampling_variance"].to_numpy(),
        )
        homegeneity_mod: float = homogeneity_by_HunterSchmidt(
            effect_sizes=subset["bias_detected_mod"].to_numpy(),
            sampling_variances=subset["sampling_variance_mod"].to_numpy(),
        )

        # Concate to results
        append_row: pd.DataFrame = (
            pd.Series(
                {
                    **row,
                    "homogeneity": homogeneity,
                    "homogeneity_mod": homegeneity_mod,
                }
            )
            .to_frame()
            .T
        )
        results = pd.concat([results, append_row], axis=0)

    return results


def plot_homogeneity_heatmap(
    x_axis_names: List[str],
    y_axis_names: List[str],
    values: np.ndarray,
    bias_homogeneities: np.ndarray | None = None,
    model_homogeneities: np.ndarray | None = None,
    save: bool = False,
) -> None:
    """
    Plot a heatmap of homogeneity values
    """
    # If the dtype of the values is not float, convert it to float
    if values.dtype != float:
        values = np.array(values, dtype=float)

    # Round
    values = np.round(values, 2)

    # Configure matplotlib to use LaTeX fonts
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams.update({"font.size": 11})

    # Create a color map
    cmap = plt.get_cmap("coolwarm")

    fig, ax = plt.subplots()
    cax = ax.imshow(values, cmap=cmap)

    # Set ticks
    ax.set_xticks(np.arange(len(x_axis_names)), labels=x_axis_names)
    ax.set_yticks(np.arange(len(y_axis_names)), labels=y_axis_names)

    # Rotate the x tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_axis_names)):
        for j in range(len(x_axis_names)):
            value = values[i, j]

            # Get the background color of the cell
            bg_color = cmap(value / values.max())
            # Calculate text color based on luminance
            luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
            text_color = "white" if luminance < 0.5 else "black"

            ax.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                color=text_color,
            )

    if bias_homogeneities is not None and model_homogeneities is not None:
        # If the dtype of the values is not float, convert it to float
        if bias_homogeneities.dtype != float:
            bias_homogeneities = np.array(bias_homogeneities, dtype=float)
        if model_homogeneities.dtype != float:
            model_homogeneities = np.array(model_homogeneities, dtype=float)

        # Round to 3 decimal places
        bias_homogeneities = np.round(bias_homogeneities, 3)
        model_homogeneities = np.round(model_homogeneities, 3)

        # Append model homogeneities at the top
        for j in range(len(x_axis_names)):
            ax.text(
                j,
                -1,  # Position above the heatmap rows
                model_homogeneities[j],
                ha="center",
                va="center",
                color="black",
                fontsize=9,
                fontweight="bold",  # Make bold
            )

        # Append bias homogeneities to the right
        for i in range(len(y_axis_names)):
            ax.text(
                len(x_axis_names),  # Position after the heatmap columns
                i,
                bias_homogeneities[i],
                ha="center",
                va="center",
                color="black",
                fontsize=9,
                fontweight="bold",  # Make bold
            )

        # Set limits to account for the added values
        ax.set_xlim(-0.5, len(x_axis_names) + 0.5)
        ax.set_ylim(len(y_axis_names) - 0.5, -1.5)

        # Remove borders and ticks
        ax.spines[:].set_visible(False)  # Hide all borders
        ax.tick_params(top=False, bottom=False, left=False, right=False)

    # Add color bar
    cbar = fig.colorbar(cax, ax=ax, shrink=0.82)
    cbar.ax.set_ylabel("homogeneity", rotation=-90, va="bottom")

    fig.tight_layout()
    plt.show()

    if save:
        # Save as svg
        fig.savefig(
            "/Users/mAx/Documents/Master/04/Master_Thesis/02_Thesis/Chapters/04_Results/Homogeneity/homogeneity_heatmap.svg",
            format="svg",
            bbox_inches="tight",
        )


if __name__ == "__main__":
    import inquirer

    # Ask the user for filter options
    print(
        "If you want to compute the homogeneity on a more-detailed level, confirm the desired levels. Otherwise, the homogeneity will be computed across the average of all experiments."
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
        results: pd.DataFrame = pd.DataFrame()
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

            # Concate to results
            append_row: pd.DataFrame = (
                pd.Series(
                    {
                        **row,
                        "homogeneity": homogeneity,
                        "homogeneity_mod": homegeneity_mod,
                    }
                )
                .to_frame()
                .T
            )
            results = pd.concat([results, append_row], axis=0)

            # Print
            print(f"\033[1mCombination: {', '.join(row.tolist())}\033[0m")
            print(f"Homogeneity: {homogeneity}")
            print(f"Homogeneity with modified effect size: {homegeneity_mod}")

        if levels_list == ["bias", "model"]:
            # Ask whether to plot homogeneity or homogeneity_mod
            questions: List = [
                inquirer.Confirm(
                    name="homogeneity_q",
                    message="Plot unmodified homogeneity?",
                    default=True,
                ),
            ]
            homogeneity_q: Dict[str, bool] | None = inquirer.prompt(questions)

            if homogeneity_q is not None and homogeneity_q["homogeneity_q"]:
                which_homogeneity: str = "homogeneity"
            else:
                which_homogeneity: str = "homogeneity_mod"

            # Ask whether to store the homogeneity heatmap
            questions: List = [
                inquirer.Confirm(
                    name="save",
                    message="Save the homogeneity heatmap (plot)?",
                    default=False,
                ),
            ]
            save: Dict[str, bool] | None = inquirer.prompt(questions)

            # Prepare data for plotting
            x_axis_names: List[str] = results["model"].unique().tolist()
            y_axis_names: List[str] = results["bias"].unique().tolist()
            values: np.ndarray = (
                results[which_homogeneity]
                .to_numpy()
                .reshape(len(y_axis_names), len(x_axis_names))
            )

            # Ask whether to add the overall bias and model homogeneities
            questions: List = [
                inquirer.Confirm(
                    name="add",
                    message="Add overall bias and model homogeneities to the heatmap?",
                    default=False,
                ),
            ]
            add: Dict[str, bool] | None = inquirer.prompt(questions)

            if add is not None and add["add"]:
                # Get homogeneity for the model and bias levels
                homog_bias: pd.DataFrame = get_homogeneity_one_level(level="bias")
                bias_homogeneities: np.ndarray = homog_bias[
                    which_homogeneity
                ].to_numpy()
                homog_model: pd.DataFrame = get_homogeneity_one_level(level="model")
                model_homogeneities: np.ndarray = homog_model[
                    which_homogeneity
                ].to_numpy()

                # Plot homogeneity heatmap
                if save is not None:
                    plot_homogeneity_heatmap(
                        x_axis_names=x_axis_names,
                        y_axis_names=y_axis_names,
                        values=values,
                        bias_homogeneities=bias_homogeneities,
                        model_homogeneities=model_homogeneities,
                        save=save["save"],
                    )

            else:
                # Plot homogeneity heatmap
                if save is not None:
                    plot_homogeneity_heatmap(
                        x_axis_names=x_axis_names,
                        y_axis_names=y_axis_names,
                        values=values,
                        save=save["save"],
                    )
