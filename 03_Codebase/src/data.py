from datetime import datetime
import json
import os
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict


class ExperimentConfig(BaseModel):
    # Configuration for experiments stored in csv
    bias: str
    experiment: str
    model: str
    local: bool
    temperature: float
    top_p: float
    content: str
    variables: Dict[str, str | float | int | bool] | None


class DataInteractor:
    def __init__(self, config_class: BaseModel = ExperimentConfig) -> None:
        """Initialize the data interactor class
        Parameters:
        config_class: BaseModel
            Pydantic model to use for configuration
        """
        # Check if any modifications to master data files
        files = ["bias_params.xlsx", "model_params.xlsx"]
        if DataInteractor.check_if_any_mods(files):
            # Merge the files
            experiments = self.merge_master_data(files)
        else:
            experiments = pd.read_excel(
                "./03_Codebase/res/experiments_config/experiments.xlsx"
            )

        # Assert that excel is not empty
        assert not experiments.empty, f"{datetime.now()} | No experiments found"

        # Convert the variables column to a dictionary
        experiments["variables"] = experiments["variables"].apply(json.loads)

        # Convert the excel to a list of dictionaries with type checking
        self.experiments: List[ExperimentConfig] = [
            ExperimentConfig(**experiment)
            for experiment in experiments.to_dict(orient="records")
        ]

    @staticmethod
    def check_mod_date(file_path: str) -> float:
        """Check the modification date of a file
        Parameters:
        file_path: str
            Path to the file
        Outputs:
        float
            Modification date
        """
        return os.path.getmtime(file_path)

    @staticmethod
    def check_if_any_mods(files: List[str]) -> bool:
        """Check if any of the files have been modified since the last merge
        Parameters:
        files: List[str]
            List of files to check
        Outputs:
        bool
            True if any of the files have been modified
        """
        # Get modification dates for all separated files
        mod_dates: List[float] = [
            DataInteractor.check_mod_date(
                f"./03_Codebase/res/experiments_config/{file}"
            )
            for file in files
        ]

        # Get modification date for the entire master data file, if it exists
        if not os.path.exists("./03_Codebase/res/experiments_config/experiments.xlsx"):
            return True
        master_data_date: float = DataInteractor.check_mod_date(
            "./03_Codebase/res/experiments_config/experiments.xlsx"
        )

        # Check if any of the modification dates are younger than the master data file
        if any([date > master_data_date for date in mod_dates]):
            return True

        return False

    def merge_master_data(self, files: List[str], save: bool = True) -> pd.DataFrame:
        """Merge the master data file with the modified files
        Parameters:
        self: DataInteractor
            DataInteractor class
        files: List[str]
            List of files to merge
        save: bool
            Save the merged file
        Outputs:
        pd.DataFrame
            Merged data
        """
        # Create the Cartesian product of the master data
        master_data = pd.DataFrame({"helper": [1]})
        for file in files:
            data = pd.read_excel(f"./03_Codebase/res/experiments_config/{file}")
            data["helper"] = 1
            master_data = master_data.merge(data, on="helper")
        master_data = master_data.drop("helper", axis=1)

        # Save the merged file
        if save:
            master_data.to_excel(
                "./03_Codebase/res/experiments_config/experiments.xlsx", index=False
            )

        return master_data


if __name__ == "__main__":
    DataInteractor()
