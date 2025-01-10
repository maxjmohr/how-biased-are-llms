import matplotlib.pyplot as plt
import pandas as pd
from src.data.db_helpers import Database
import seaborn as sns
from typing import List, Dict


def get_data() -> pd.DataFrame:
    "Function to get bias detections grouped by bias, models, scenarios"
    # Connect to the database
    db = Database()
    db.connect()

    # Query
    data: pd.DataFrame = db.fetch_data(total_object="t_bias_detections")

    # Disconnect from the database
    db.disconnect()

    # Rename some columns
    data = data.rename(
        columns={
            "bias_detected_mod": "bias_detected_capped",
        }
    )

    return data


def plot(data: pd.DataFrame, save: bool = False):
    "Function to plot the bias detections distributions"
    # Get portion of bias_detected == 0
    portion: float = len(data[data["bias_detected"] == 0]) / len(data)
    print(f"Portion of bias_detected == 0: {portion}")
    # Portion above 0
    portion: float = len(data[data["bias_detected"] > 0]) / len(data)
    print(f"Portion of bias_detected > 0: {portion}")
    # Portion below 0
    portion: float = len(data[data["bias_detected"] < 0]) / len(data)
    print(f"Portion of bias_detected < 0: {portion}")

    # Filter bias_detected for visualization purposes
    filtered_data: pd.DataFrame = pd.DataFrame(
        data[(data["bias_detected"] > -1.7) & (data["bias_detected"] < 1.7)]
    )
    portion: float = len(filtered_data) / len(data)
    print(f"Portion of data: {portion}")

    # Configure matplotlib to use LaTeX fonts
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams.update({"font.size": 18})

    # Create a figure and axes for two subplots
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    # Plot distribution of bias_detected
    sns.histplot(
        data=filtered_data,
        x="bias_detected",
        kde=True,
        line_kws={"linewidth": 3},
    )

    plt.xlabel("bias_detected", fontsize=18)
    plt.ylabel("frequency", fontsize=18)

    # Adjust layout for better appearance
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Optionally save the plot as SVG
    if save:
        fig.savefig(
            "/Users/mAx/Documents/Master/04/Master_Thesis/02_Thesis/Chapters/04_Results/Overview/bias_detections_distribution.svg",
            format="svg",
            bbox_inches="tight",
        )


if __name__ == "__main__":
    import inquirer

    # Ask the user if they want to save the plot
    questions: List = [
        inquirer.Confirm(
            name="save",
            message="Save the detection heatmap (plot)?",
            default=False,
        ),
    ]
    save: Dict[str, bool] | None = inquirer.prompt(questions)

    # Plot
    if save is not None:
        # Get data
        data = get_data()

        # Plot
        plot(data, save=save["save"])
