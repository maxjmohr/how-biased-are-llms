import argparse
from datetime import datetime
import json
from numpy import NAN
import pandas as pd
from typing import Dict


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


def filter_parser_args(
    experiments: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    "Filter the experiments based on the parser arguments (if given)"
    # Get all args that are not None
    args_dict = {k: v for k, v in vars(args).items() if v is not None}

    # If no args, return original df
    if not args_dict:
        return experiments

    # Filter the experiments based on the args
    print(f"{datetime.now()} | Filtering experiments for these arguments: {args_dict}")
    # If key is "test", delete here (it is solely needed to activate test mode)
    if "test" in args_dict:
        del args_dict["test"]
    # Filter
    for key, value in args_dict.items():
        experiments = experiments.loc[experiments[key] == value]

    return experiments


def calc_remaining_loops(target_loops: int, correct_runs: int) -> int:
    "Calculate the remaining loops each exercise should run"
    return target_loops - correct_runs
