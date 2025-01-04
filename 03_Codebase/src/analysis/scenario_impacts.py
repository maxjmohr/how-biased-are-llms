import matplotlib.pyplot as plt
import os
import pandas as pd
from src.data.db_helpers import Database
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List


def get_data(filter: bool = False, create_dummies: bool = False) -> pd.DataFrame:
    "Function to get bias detections grouped by bias, models, scenarios"
    # Connect to the database
    db = Database()
    db.connect()

    # Query
    sql = """
        SELECT
            bias,
            model,
            scenario,
            temperature,
            AVG(bias_detected) as bias_detected,
            AVG(bias_detected_mod) as bias_detected_mod
        FROM
            t_bias_detections
        GROUP BY
            bias,
            model,
            scenario,
            temperature
    """
    data: pd.DataFrame = db.fetch_data(sql=sql)

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

    # Rename values in scenario column
    data["scenario"] = data["scenario"].replace(
        {
            "0_normal": "1_no_persona",
            "1_persona": "0_normal",
        }
    )

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
):
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

    return results


def regression_diagnostics(model):
    "Daignose the regression model with residual plot, QQ plot and variance inflation factor"
    # Residual plot
    sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals vs Fitted")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.show()

    # QQ plot
    sm.qqplot(model.resid, line="s")
    plt.title("Q-Q Plot")
    plt.show()

    # Variance Inflation Factor
    variables = model.model.exog
    vif = pd.DataFrame()
    vif["VIF"] = [
        variance_inflation_factor(variables, i) for i in range(variables.shape[1])
    ]
    vif["feature"] = model.model.exog_names
    print(vif)


if __name__ == "__main__":
    # Get data
    data = get_data(filter=False, create_dummies=False)

    # Regressions scenarios
    formulas: List[str] = [
        "bias_detected ~ C(scenario)",
        # "bias_detected ~ C(scenario) + C(bias) + C(model)",
        "bias_detected_capped ~ C(scenario)",
        # "bias_detected ~ C(scenario) + C(bias) + C(model) + C(scenario):C(bias) + C(scenario):C(model) + C(bias):C(model)",
        # f"bias_detected ~ {' + '.join(data.columns[3:])}",
        # "bias_detected_mod ~ C(scenario) + C(bias) + C(model) + temperature",
    ]
    for formula in formulas:
        # Get the first word uintil " ~" to create the file name for the latex output
        # Get current working directory
        current_path: str = os.path.dirname(os.path.abspath(__file__))
        target_path: str = os.path.join(
            current_path, "../../../02_Thesis/Chapters/06_Appendix/"
        )

        file_name: str = (
            target_path + "regression_scenarios_" + formula.split(" ~")[0] + ".tex"
        )

        fitted_model = regression(
            data, regression=formula, reg_type="ols", file_name=file_name
        )
        print("\n\n")
        # regression_diagnostics(fitted_model)

    # Regressions all
    formulas: List[str] = [
        "bias_detected ~ C(bias) + C(model) + C(scenario) + temperature",
        "bias_detected_capped ~ C(bias) + C(model) + C(scenario) + temperature",
    ]
    for formula in formulas:
        # Get the first word uintil " ~" to create the file name for the latex output
        # Get current working directory
        current_path: str = os.path.dirname(os.path.abspath(__file__))
        target_path: str = os.path.join(
            current_path, "../../../02_Thesis/Chapters/06_Appendix/"
        )

        file_name: str = (
            target_path + "regression_allexper_" + formula.split(" ~")[0] + ".tex"
        )

        fitted_model = regression(
            data, regression=formula, reg_type="ols", file_name=file_name
        )
        print("\n\n")
        # regression_diagnostics(fitted_model)
