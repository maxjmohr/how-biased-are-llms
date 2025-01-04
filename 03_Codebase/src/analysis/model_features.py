import os
import pandas as pd
from src.data.db_helpers import Database
import statsmodels.formula.api as smf
from typing import List


def get_data(filter: bool = False, create_dummies: bool = False) -> pd.DataFrame:
    "Function to get bias detections grouped by bias, models, scenarios"
    # Connect to the database
    db = Database()
    db.connect()

    # Fetch all bias detections
    data: pd.DataFrame = db.fetch_data(total_object="t_bias_detections")

    # Read excel with additional model features
    current_path = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(
        current_path, "./../../res/db_objects_content/models_features.xlsx"
    )
    model_features: pd.DataFrame = pd.read_excel(target_path, sheet_name="filled")

    # Inner join on model
    data = data.join(
        model_features.set_index("model"), on="model", how="inner", rsuffix="_mod"
    )

    # Turn temperature into float
    data["temperature"] = data["temperature"].astype(float)

    # Disconnect from the database
    db.disconnect()

    # Filter out the rows where bias = category size bias, gamblers fallacy, or sunk cost fallacy
    if filter:
        data = pd.DataFrame(
            data[
                ~data["bias"].isin(
                    ["category size bias", "gamblers fallacy", "sunk cost fallacy"]
                )
            ]
        )

    # Create dummies
    if create_dummies:
        data = pd.get_dummies(
            data, columns=["bias", "model", "scenario"], drop_first=False
        )

        # Clean column names for Patsy compatibility
        data.columns = data.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)

    # Rename some columns
    data = data.rename(
        columns={
            "bias_detected_mod": "bias_detected_capped",
        }
    )

    return data


def regression(
    data: pd.DataFrame,
    regression: str = "bias_detected ~ C(scenario)",
    reg_type: str = "ols",
    file_name: str = "regression",
) -> None:
    "Regression to find out impact of scenarios on bias detections"
    # Regression
    if reg_type == "ols":
        model = smf.ols(regression, data=data)
    elif reg_type == "mixedlm":
        # Create column with bias + model
        data["biasmodel"] = data["bias"] + data["model"]

        model = smf.mixedlm(regression, data, groups=data["biasmodel"])
    else:
        raise ValueError("Invalid regression type")

    # Fit the model
    results = model.fit()

    # Print the formula bold
    print(
        "\n#----------------------------------------------------------------------------#"
    )
    print("####### Regression formula: #######")
    print(f"\033[1m{regression}\033[0m")
    print("\n####### Regression results: #######")
    print(results.summary())

    # Save the results to a latex file
    with open(file_name, "w") as f:
        f.write(results.summary().as_latex())


if __name__ == "__main__":
    # Get data
    data = get_data(filter=False, create_dummies=False)

    # Regression
    formulas: List[str] = [
        "bias_detected ~ temperature + release_date_diff + last_updated_date_diff + training_data_cutoff_date_diff + number_parameters + context_length + mmlu + chatbot_arena",
        "bias_detected_capped ~ temperature + release_date_diff + last_updated_date_diff + training_data_cutoff_date_diff + number_parameters + context_length + mmlu + chatbot_arena",
        # "bias_detected ~ release_date_diff + last_updated_date_diff + training_data_cutoff_date_diff + number_parameters + context_length + mmlu + chatbot_arena",
        # "bias_detected_mod ~ release_date_diff + last_updated_date_diff + training_data_cutoff_date_diff + number_parameters + context_length + mmlu + chatbot_arena",
        # "bias_detected ~ C(bias) + C(model) + C(scenario) + temperature",
        # "bias_detected_mod ~ C(bias) + C(model) + C(scenario) + temperature",
    ]

    for formula in formulas:
        # Get the first word uintil " ~" to create the file name for the latex output
        # Get current working directory
        current_path: str = os.path.dirname(os.path.abspath(__file__))
        target_path: str = os.path.join(
            current_path, "../../../02_Thesis/Chapters/06_Appendix/"
        )

        file_name: str = (
            target_path + "regression_modelfeats_" + formula.split(" ~")[0] + ".tex"
        )

        regression(data, regression=formula, reg_type="ols", file_name=file_name)
