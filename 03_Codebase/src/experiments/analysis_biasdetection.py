from datetime import datetime
import pandas as pd
from src.data.db_helpers import Database
from typing import List


class BiasDetector:
    "Class to detect biases by computing a binary target variable for experiments with responses == threshold"

    def __init__(
        self,
        bias: str = "",
        scenario: str = "",
        model: str = "",
        temperature: float = -1,
        threshold: int = 100,
    ) -> None:
        self.bias = bias
        self.scenario = scenario
        self.model = model
        self.temperature = temperature
        self.threshold = threshold

        # Initialize database
        self.db = Database()
        self.db.connect()

        # Construct where clause
        self.contruct_where_clause()

    def contruct_where_clause(self) -> None:
        "Construct the where clause based on the filters provided"
        where: List = []
        if self.bias:
            where.append(f"bias = {str(self.bias)}")
        if self.scenario:
            where.append(f"scenario = {str(self.scenario)}")
        if self.model:
            where.append(f"model = {str(self.model)}")
        if self.temperature != -1:
            where.append(f"temperature = {str(self.temperature)}")
        self.where_clause: str = "WHERE " + " AND ".join(where) if where else ""

    def fetch_computable_experiments(self) -> None:
        "Check 1) whether we have enough responses for the bias model combination and if yes, 2) when the last response was generated (ran_date) and 3) if it was after the last computation, otherwise we don't need to recompute"

        print(
            f"{datetime.now()} | Fetching computable experiments for bias: {self.bias}, scenario: {self.scenario}, model: {self.model}, temperature: {self.temperature}"
        )
        # Select all the bias model combinations that are ready for computation
        sql: str = f"""
            WITH

            computable_experiments AS (
                SELECT *
                FROM v_experiments
                {self.where_clause + " AND " if self.where_clause else "WHERE "}
                correct_ran_loops = {str(self.threshold)}
                ),

            calculated_detections AS (
                SELECT *
                FROM t_bias_detections
                {self.where_clause}
            )

            SELECT
                ce.experiment_id, ce.bias_id, ce.model_id, ce.bias, ce.experiment_type, ce.scenario, ce.model, ce.temperature, ce.part, ce.parts_total, ce.response_type
            FROM
                computable_experiments ce
                LEFT JOIN calculated_detections cd  ON cd.bias = ce.bias
                                                    AND cd.scenario = ce.scenario
                                                    AND cd.model = ce.model
                                                    AND cd.temperature = ce.temperature
            WHERE
                ce.ran_date > cd.updated_at
                OR cd.updated_at IS NULL
            ;
            """
        self.computable_experiments: pd.DataFrame = self.db.fetch_data(sql=sql)
        print(self.computable_experiments[["bias", "scenario", "model", "temperature"]])

    def compute_1q_choice(self):
        "Binary target variable computation for 1 question choice experiments"
        # TODO
        # Binary output?
        return "bias detected"

    def compute_2q_choice(self):
        "Binary target variable computation for 2 question choice experiments"
        # TODO
        # Binary output?
        return "bias detected"

    def compute_2q_numeric(self):
        "Binary target variable computation for 2 question value estimation experiments"
        # TODO
        # Binary output?
        return "bias detected"

    def detect_bias(self):
        # TODO
        # Binary output?
        return "bias detected"


if __name__ == "__main__":
    import inquirer
    import os
    import sys

    # Add total codebase
    parent_dir: str = os.path.dirname(os.path.realpath(__file__ + "../../"))
    sys.path.append(parent_dir)

    # Lists to filter
    biases: List[str] = [
        "all",
        "anchoring",
        "category size bias",
        "endowment effect",
        "framing effect",
        "gambler's fallacy",
        "loss aversion",
        "sunk cost fallacy",
        "transaction utility",
    ]
    scenarios: List[str] = ["all", "0_normal"]
    models: List[str] = [
        "all",
        "gemma2",
        "gemma2:27b",
        "gpt-4o-mini",
        "gpt-4o",
        "llama3.1",
        "llama3.1:70b",
        "phi3.5",
        "phi3:medium",
    ]
    temperatures: List[float] = [-1, 0.7]

    # Ask the user for filter options
    print(
        "Please select the bias, scenario, model and temperature to filter the experiments you want to calculate the bias detection for."
    )
    questions: List = [
        inquirer.List(
            name="bias",
            message="Which bias?",
            choices=biases,
        ),
        inquirer.List(
            name="scenario",
            message="Which scenario?",
            choices=scenarios,
        ),
        inquirer.List(
            name="model",
            message="Which model?",
            choices=models,
        ),
        inquirer.List(
            name="temperature",
            message="Which temperature?",
            choices=temperatures,
        ),
    ]
    answers: dict | None = inquirer.prompt(questions)
    if answers is None:
        print("No function selected.")
        sys.exit()

    # If answer "all", then convert to ""
    for key, value in answers.items():
        if value == "all":
            answers[key] = ""

    # Initialize the BiasDetector class
    bd = BiasDetector(
        bias=answers["bias"],
        scenario=answers["scenario"],
        model=answers["model"],
        temperature=answers["temperature"],
    )
    print(bd.fetch_computable_experiments())
