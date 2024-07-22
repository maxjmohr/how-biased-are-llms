"""
DISCLAIMER: THIS FILE IS MEANT TO BE USED IF THE DATA IS STORED LOCALLY IN EXCEL FILES.
The script was not further developed since the data was stored in a database.
"""

from datetime import datetime
import json
from numpy import NAN
import os
import pandas as pd
from pydantic import BaseModel
from typing import Dict, List, Tuple


class ExperimentConfig(BaseModel):
    # Configuration for experiments stored in excel

    experiment_id: int
    bias_id: int
    model_id: int
    bias: str
    experiment_type: str
    model: str
    local: bool
    temperature: float
    system: str
    total_content: str
    target_choice: str
    updated_at: datetime

    class Config:
        protected_namespaces = ()  # model_ otherwise protected by pydantic


class DataInteractor:
    def __init__(self, config_class: type[BaseModel] = ExperimentConfig) -> None:
        """Initialize the data interactor class
        Parameters:
        config_class: BaseModel
            Pydantic model to use for configuration
        """
        # Check if any modifications to master data files
        files = ["bias_params.xlsx", "model_params.xlsx"]
        if DataInteractor.check_if_any_mods(files):
            # Merge the files
            print(f"{datetime.now()} | Merging master data files")
            experiments = self.merge_master_data(files)
        else:
            print(
                f"{datetime.now()} | No modifications to master data, reading experiments from excel"
            )
            experiments = pd.read_excel(
                "./03_Codebase/res/experiments_config/experiments.xlsx"
            )

        # Assert that excel is not empty
        assert not experiments.empty, f"{datetime.now()} | No experiments found"

        # Convert the excel to a list of dictionaries with type checking
        self.experiments: List[ExperimentConfig] = [
            ExperimentConfig(**experiment)
            for experiment in experiments.to_dict(orient="records")
        ]

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
            os.path.getmtime(f"./03_Codebase/res/experiments_config/{file}")
            for file in files
        ]

        # Get modification date for the entire master data file, if it exists
        if not os.path.exists("./03_Codebase/res/experiments_config/experiments.xlsx"):
            return True
        master_data_date: float = os.path.getmtime(
            "./03_Codebase/res/experiments_config/experiments.xlsx"
        )

        # Check if any of the modification dates are younger than the master data file
        if any([date > master_data_date for date in mod_dates]):
            return True

        return False

    @staticmethod
    def combine_content_variables(content: str, variables: str) -> str:
        """Combine the content and variables into a single string
        Parameters:
        content: str
            Content of the experiment
        variables: str
            Variables of the experiment
        Outputs:
        str
            Combined content and variables
        """
        # Check if the variables are empty
        if variables is NAN:
            return content

        # Convert the variables column to a dictionary
        try:
            variables_dict: Dict[str, str | float | int | bool] = json.loads(variables)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from variables: {variables}") from e

        # Replace the variables in the content
        for key, value in variables_dict.items():
            content = content.replace(f"{key}", str(value))

        return content

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

        # Create the experiment_id by combining the bias_id and model_id
        master_data["experiment_id"] = (
            master_data["bias_id"].astype(str) + master_data["model_id"].astype(str)
        ).astype(int)

        # Combine the content and variables
        print(f"{datetime.now()} | Replacing variables in content")
        master_data["total_content"] = master_data.apply(
            lambda x: DataInteractor.combine_content_variables(
                x["content"], x["variables"]
            ),
            axis=1,
        )
        # Drop content and variables
        master_data = master_data.drop(["content", "variables"], axis=1)

        # Add the updated_at column
        master_data["updated_at"] = datetime.now()

        # Make sure the order of the columns is correct
        master_data = master_data.loc[
            :,
            [
                "experiment_id",
                "bias_id",
                "model_id",
                "bias",
                "experiment_type",
                "model",
                "local",
                "temperature",
                "top_p",
                "total_content",
                "target_choice",
                "updated_at",
            ],
        ]

        # Save the merged file
        if save:
            master_data.to_excel(
                "./03_Codebase/res/experiments_config/experiments.xlsx", index=False
            )

        return master_data

    # TODO: def remove_ran_experiments
    # TODO: def partition_data
    # TODO: def get_specific_experiment
    # TODO: def filter_experiments

    @staticmethod
    def check_create_directory(directory: str) -> None:
        """Check if the directory exists, if not create it
        Parameters:
        directory: str
            Directory to check
        """
        if not os.path.exists(directory):
            print(f"{datetime.now()} | Creating directory: {directory}")
            os.makedirs(directory)

    @staticmethod
    def save_responses(
        results: Dict[str, List[Tuple[str, str]]], filename: str, save_as: str = "json"
    ) -> None:
        """Save the responses to the specified directory
        Parameters:
        results: Dict[str, List[Tuple[str, str]]]
            Results to save
        filename: str
            Filename to save the results as
        save_as: str
            Format to save the results as
        """
        assert results != {}, "Results are empty"
        assert filename != "", "Filename is empty"
        assert save_as in ["json", "csv"], "Invalid save_as format (either json or csv)"

        # Check if the directory exists, if not create it
        DataInteractor.check_create_directory("./03_Codebase/res/results")

        # Save the results
        print(
            f"{datetime.now()} | Saving results as: ./03_Codebase/res/results/{filename}.{save_as}"
        )
        if save_as == "json":
            with open(f"./03_Codebase/res/results/{filename}.json", "w") as f:
                json.dump(results, f, indent=4)
        elif save_as == "csv":
            df = pd.DataFrame(results)
            df.to_csv(f"./03_Codebase/res/results/{filename}.csv", index=False)

    # TODO: def save_concat_results (use BiasDetector and then store the concatenated results)

    @staticmethod
    def concat_results():
        # Load the responses
        # Open all files in ./03_Codebase/res/results
        # Concatenate the results
        pass


if __name__ == "__main__":
    DataInteractor()
    # print(DataInteractor().experiments)
