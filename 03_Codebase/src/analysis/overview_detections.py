import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.data.db_helpers import Database
from typing import Dict, List, Tuple


def fetch_data(
    group_by: List[str] = ["bias", "model"],
) -> Tuple[List[str], List[str], np.ndarray]:
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

    # Get the unique values of the group_by columns
    x_axis_names: List[str] = detected_biases[group_by[1]].unique().tolist()
    y_axis_names: List[str] = detected_biases[group_by[0]].unique().tolist()

    # Create a matrix of the bias values
    values: np.ndarray = np.zeros((len(y_axis_names), len(x_axis_names)))
    values: np.ndarray = (
        detected_biases["bias_detected"]
        .to_numpy()
        .reshape(len(y_axis_names), len(x_axis_names))
    )

    return x_axis_names, y_axis_names, values


def plot_bias_heatmap(
    x_axis_names: List[str],
    y_axis_names: List[str],
    values: np.ndarray,
    save: bool = False,
) -> None:
    """
    Plot a heatmap of detected bias values
    """
    # If the dtype of the values is not float, convert it to float
    if values.dtype != float:
        values = np.array(values, dtype=float)

    # Round to 3 decimal places
    values = np.round(values, 3)

    # Configure matplotlib to use LaTeX fonts
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    fig, ax = plt.subplots()
    cax = ax.imshow(values, cmap="coolwarm")

    # Set ticks
    ax.set_xticks(np.arange(len(x_axis_names)), labels=x_axis_names)
    ax.set_yticks(np.arange(len(y_axis_names)), labels=y_axis_names)

    # Rotate the x tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_axis_names)):
        for j in range(len(x_axis_names)):
            ax.text(
                j,
                i,
                values[i, j],
                ha="center",
                va="center",
                color="w",
            )

    # Add color bar
    fig.colorbar(cax, ax=ax)

    fig.tight_layout()
    plt.show()

    if save:
        # Save as svg
        fig.savefig(
            "/Users/mAx/Documents/Master/04/Master_Thesis/02_Thesis/Chapters/04_Results/Overview/heatmap_detections.svg",
            format="svg",
            bbox_inches="tight",
        )


if __name__ == "__main__":
    import inquirer

    # Ask the user if they want to save the plot
    questions: List = [
        inquirer.Confirm(
            name="save",
            message="Save the homogeneity heatmap (plot)?",
            default=False,
        ),
    ]
    save: Dict[str, bool] | None = inquirer.prompt(questions)

    if save is not None:
        x_axis_names, y_axis_names, values = fetch_data()
        plot_bias_heatmap(x_axis_names, y_axis_names, values, save=save["save"])
