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

    # Rename values in scenario column
    data["scenario"] = data["scenario"].replace(
        {
            "0_normal": "1_no_persona",
            "1_persona": "0_normal",
        }
    )
    # Sort data asc by scenario
    data = data.sort_values(by="scenario", ascending=True)

    # Rename some columns
    data = data.rename(
        columns={
            "bias_detected_mod": "bias_detected_capped",
        }
    )

    return data


def plot(data: pd.DataFrame, save: bool = False):
    "Function to plot the bias detections per scenario"
    # Filter bias_detected for visualization purposes
    data = pd.DataFrame(
        data[(data["bias_detected"] > -5) & (data["bias_detected"] < 5)]
    )

    # Configure matplotlib to use LaTeX fonts
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams.update({"font.size": 18})

    # Create a figure and axes for the plot
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the size as needed

    # Create the plot
    sns.violinplot(
        data=data,
        x="bias_detected",
        y="scenario",
        density_norm="width",
    )

    # Add labels and title
    plt.xlabel("bias detected")
    plt.ylabel("")

    # Adjust layout for LaTeX integration
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Optionally save the plot as SVG
    if save:
        fig.savefig(
            "/Users/mAx/Documents/Master/04/Master_Thesis/02_Thesis/Chapters/04_Results/Scenarios/scenario_detections.svg",
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
