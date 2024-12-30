from datetime import datetime
from llama_index.llms.ollama.base import Tuple
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from src.data.db_helpers import Database
from typing import List, Dict


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
            where.append(f"bias = '{str(self.bias)}'")
        if self.scenario:
            where.append(f"scenario = '{str(self.scenario)}'")
        if self.model:
            where.append(f"model = '{str(self.model)}'")
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
            ),

            pre_selection AS (
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
            ),

            check_if_all_parts_are_ready AS (
                SELECT
                    bias, model, scenario, temperature,
                    COUNT(DISTINCT part) AS parts_ready
                FROM
                    pre_selection
                GROUP BY
                    bias, model, scenario, temperature
            )

            SELECT
                ce.experiment_id, ce.bias_id, ce.model_id, ce.bias, ce.experiment_type, ce.scenario, ce.model, ce.temperature, ce.part, ce.parts_total, ce.response_type
            FROM
                pre_selection ce
                INNER JOIN check_if_all_parts_are_ready c   ON c.bias = ce.bias
                                                            AND c.model = ce.model
                                                            AND c.scenario = ce.scenario
                                                            AND c.temperature = ce.temperature
                                                            AND c.parts_ready = ce.parts_total
            ORDER BY
                ce.bias, ce.scenario, ce.model, ce.temperature, ce.part
            ;
            """
        self.computable_experiments: pd.DataFrame = self.db.fetch_data(sql=sql)
        print(
            self.computable_experiments[
                ["bias", "scenario", "model", "temperature"]
            ].drop_duplicates()
        )

    def cohens_d_2g(self, group1: NDArray, group2: NDArray) -> float:
        """Compute Cohen's d effect size for two-group design
        Group 1 is control, Group 2 is treatment
        This is the formula we use (regardless if considered as two-group or repeated measures design):
        d = (M2 - M1) / sqrt(((n1 - 1) * std1^2 + (n2 - 1) * std2^2) / (n1 + n2 - 2))
        If same sample size: d = (M2 - M1) / sqrt((s1^2 + s2^2) / 2)
        """
        # Compute means
        mean1: float = float(np.mean(group1))
        mean2: float = float(np.mean(group2))

        # Compute standard deviations
        std1: float = float(np.std(group1))
        std2: float = float(np.std(group2))

        # Compute sample sizes
        n1: int = len(group1)
        n2: int = len(group2)
        if n1 + n2 == 2:
            return 0.0

        # Pooled standard deviation
        std_pooled: float = np.sqrt(
            ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        )
        if std_pooled == 0:
            return 0.0

        # Compute Cohen's d
        return (mean2 - mean1) / std_pooled

    def compute_2q_choice(
        self,
        q1_counts: pd.DataFrame,
        q2_counts: pd.DataFrame,
        map_answers: Dict[str, int],
    ) -> Tuple[float, int, int]:
        """Binary target variable computation for 2 question choice experiments
        Group 1 is control, Group 2 is treatment
        If Hypothesis test: Fisher's exact test: difference between observed and expected frequencies
        If Effect Size: Characters into numeric values (A = 0, B = 1, ...)*their occurences and compute Cohen's d
        """
        # Create an array per question (= group). Map each letter to a value and create an array with this value and the length as the number of occurences
        q1_group: NDArray = np.repeat(
            [int(map_answers[letter]) for letter in q1_counts["response"]],
            q1_counts["count"],
        )
        q2_group: NDArray = np.repeat(
            [int(map_answers[letter]) for letter in q2_counts["response"]],
            q2_counts["count"],
        )

        # Calulcate Cohen's d
        return self.cohens_d_2g(q1_group, q2_group), len(q1_group), len(q2_group)

    def compute_2q_numeric(
        self, q1_counts: pd.DataFrame, q2_counts: pd.DataFrame
    ) -> Tuple[float, int, int]:
        """Binary target variable computation for 2 question value estimation experiments
        Group 1 is control, Group 2 is treatment
        If Hypothesis test: Wilcoxon signed-rank test: difference between observed and expected values (median)
        If Effect Size: Compute Cohen's d
        """
        # Create an array per question (= group). Create an array with the valuse and the length for each value as the number of occurences
        q1_group: NDArray = np.repeat(
            [float(value) for value in q1_counts["response"]], q1_counts["count"]
        )
        q2_group: NDArray = np.repeat(
            [float(value) for value in q2_counts["response"]], q2_counts["count"]
        )

        # Calulcate Cohen's d
        return self.cohens_d_2g(q1_group, q2_group), len(q1_group), len(q2_group)

    def fetch_group_responses(self, experiment_id: int) -> pd.DataFrame:
        "Fetch the responses and their occurences for the control/treatment group"
        sql: str = f"SELECT response, count FROM v_responses_grouped WHERE experiment_id = {str(experiment_id)} AND response != 'Failed prompt';"
        return self.db.fetch_data(sql=sql)

    def sampling_variance(
        self, sample_size_g1: int, sample_size_g2: int, population_effect_size: float
    ) -> float:
        """Compute the sampling variance for an effect size
        Sample size is the number of observations in each group
        Population effect size of all experiments
        The formula to compute the approximate sampling variance of Cohen's d is:
            N = sample_size_g1 + sample_size_g2
            ñ = (sample_size_g1*sample_size_g2) / (sample_size_g1+sample_size_g2)
            c(x) = 1 - 3 / (4 * (N-2) - 1)
        var =   (1 / ñ ) * ((N-2)/(N-4)) * (1 + ñ * pop_effect_size^2) - (pop_effect_size^2 / (c(N-2))^2)
        """
        # Compute N, ñ, c(x)
        N: int = sample_size_g1 + sample_size_g2
        n_: float = (sample_size_g1 * sample_size_g2) / (
            sample_size_g1 + sample_size_g2
        )
        c: float = 1 - 3 / (4 * (N - 2) - 1)

        # Compute the sampling variance
        return (1 / n_) * ((N - 2) / (N - 4)) * (1 + n_ * population_effect_size**2) - (
            population_effect_size**2 / (c) ** 2
        )

    def detect_bias(self) -> None:
        "Detect bias for the given bias/scenario/model/temperature combination by computing the binary target variable"
        ######## FIRST SOME FILTERS ########
        # For 2 question experiments, determine which question is the control and which is the treatment (group 1 should be control or the group which should estimate a lower value)
        # True means that group 1 is control
        # False means that group 2 is control
        g1_is_control: Dict[str, bool] = {
            "anchoring": True,  # Group 1 should estimate a lower proportion (control) than group 2
            "category size bias": True,  # Group 1 should estimate a lower percentage (control) than group 2
            "endowment effect": False,  # Group 2 should estimate a lower value (control) than group 1
            "framing effect": True,  # Group 1 lost a random bill (control), group 2 lost the ticket
            "gamblers fallacy": False,  # Group 2 should estimate 50% (control) and possibly lower than group 1
            "loss aversion": True,  # Group 1 is gain scenario (control), group 2 is loss scenario
            "sunk cost fallacy": True,  # Group 1 has same valued tickets (control), group 2 the different valued tickets
            "transaction utility": False,  # Group 2 has the scenario with higher prices (should remain at store) (control) and group 1 has lower prices and should switch
        }

        # For the choice experiments, create a mapping of the answers to numeric values (if we expect higher B answers to show the bias, we should map B to 1)
        map_choices: Dict[str, Dict[str, int]] = {
            "framing effect": {"A": 0, "B": 1},
            "loss aversion": {"A": 0, "B": 1},
            "sunk cost fallacy": {"A": 1, "B": 0},
            "transaction utility": {"A": 0, "B": 1},
        }

        # For some experiments, a negative Cohen's d also indicates the bias, in these cases take the absolute value
        absolute_d: List[str] = [
            "framing effect",
        ]

        # Get unique combinations of bias, scenario, model, temperature
        unique_combinations: pd.DataFrame | pd.Series = self.computable_experiments[
            [
                "bias",
                "scenario",
                "model",
                "temperature",
                "model_id",
                "response_type",
            ]
        ].drop_duplicates()

        ######## NOW THE CALCULATION ########
        results: pd.DataFrame = pd.DataFrame()

        # Loop through each unique experiment combination
        for _, row in unique_combinations.iterrows():
            print(
                f"{datetime.now()} | Detecting bias for {row['bias']} ({row['scenario']}) on {row['model']} (temp={row['temperature']})"
            )
            # Filter the experiment ids of the unique combination
            experiment_ids: List[int] = self.computable_experiments[
                (self.computable_experiments["bias"] == row["bias"])
                & (self.computable_experiments["scenario"] == row["scenario"])
                & (self.computable_experiments["model"] == row["model"])
                & (self.computable_experiments["temperature"] == row["temperature"])
            ]["experiment_id"].tolist()

            # Get relevant filters
            g1_is_control_filter: bool = g1_is_control.get(str(row["bias"]), True)
            map_choices_filter: Dict[str, int] | None = map_choices.get(
                str(row["bias"]), None
            )
            absolute_d_filter: bool = True if row["bias"] in absolute_d else False

            # Control group
            control_group_id: int = (
                min(experiment_ids) if g1_is_control_filter else max(experiment_ids)
            )
            control_group_counts: pd.DataFrame = self.fetch_group_responses(
                control_group_id
            )
            # Treatment group
            treatment_group_id: int = (
                max(experiment_ids) if g1_is_control_filter else min(experiment_ids)
            )
            treatment_group_counts: pd.DataFrame = self.fetch_group_responses(
                treatment_group_id
            )

            # Choice or numeric experiment?
            response_type: str = str(row["response_type"])
            assert (
                response_type
                in [
                    "choice",
                    "numerical",
                ]
            ), f"Invalid response type (should be either 'choice' or 'numerical'). Currently it is {response_type}."

            # Start calculation
            if response_type == "choice" and map_choices_filter is not None:
                bias_detected, sample_size_1, sample_size_2 = self.compute_2q_choice(
                    q1_counts=control_group_counts,
                    q2_counts=treatment_group_counts,
                    map_answers=map_choices_filter,
                )
            else:
                bias_detected, sample_size_1, sample_size_2 = self.compute_2q_numeric(
                    q1_counts=control_group_counts,
                    q2_counts=treatment_group_counts,
                )

            # Take absolute value if necessary
            if absolute_d_filter:
                bias_detected = abs(bias_detected)

            # Append to results df
            results = pd.concat(
                [
                    results,
                    pd.DataFrame(
                        [
                            {
                                "bias": row["bias"],
                                "scenario": row["scenario"],
                                "model": row["model"],
                                "temperature": row["temperature"],
                                "model_id": row["model_id"],
                                "sample_size_1": sample_size_1,
                                "sample_size_2": sample_size_2,
                                "bias_detected": bias_detected,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

        # Compute the sampling variance
        # Population effect size as mean of all biases detected
        population_effect_size: float = float(results["bias_detected"].mean())

        results["sampling_variance"] = results.apply(
            lambda x: self.sampling_variance(
                sample_size_g1=x["sample_size_1"],
                sample_size_g2=x["sample_size_2"],
                population_effect_size=population_effect_size,
            ),
            axis=1,
        )

        # Insert the results
        print(f"{datetime.now()} | Inserting results into database")
        print(
            results[
                [
                    "bias",
                    "scenario",
                    "model",
                    "temperature",
                    "bias_detected",
                    "sampling_variance",
                ]
            ]
        )
        self.db.update_data(
            object="t_bias_detections",
            data=results,
            update_cols=["bias_detected", "sampling_variance"],
        )

    def modify_detected_biases(self) -> None:
        "Function to set negative biases to 0 and all values larger than 1 to 1"
        print(f"{datetime.now()} | Modifying detected biases")
        # Fetch the current bias_detected
        table_data: pd.DataFrame = self.db.fetch_data(total_object="t_bias_detections")

        # Drop updated_at
        table_data.drop(columns=["updated_at"], inplace=True)

        # Create bias_detected_mod (0 <= bias_detected_mod <= 1)
        table_data["bias_detected_mod"] = table_data["bias_detected"].apply(
            lambda x: 0 if x < 0 else (1 if x > 1 else x)
        )

        # Compute the sampling variance of modified bias_detected
        population_effect_size: float = float(table_data["bias_detected_mod"].mean())

        table_data["sampling_variance_mod"] = table_data.apply(
            lambda x: self.sampling_variance(
                sample_size_g1=x["sample_size_1"],
                sample_size_g2=x["sample_size_2"],
                population_effect_size=population_effect_size,
            ),
            axis=1,
        )

        # Assert that we have all columns left
        assert set(table_data.columns) == {
            "bias",
            "scenario",
            "model",
            "temperature",
            "model_id",
            "sample_size_1",
            "sample_size_2",
            "bias_detected",
            "sampling_variance",
            "bias_detected_mod",
            "sampling_variance_mod",
        }, "Columns are not as expected."

        # Update the table
        self.db.update_data(
            object="t_bias_detections",
            data=table_data,
            update_cols=["bias_detected_mod", "sampling_variance_mod"],
        )
        # self.db.delete_data(total_object="t_bias_detections")
        # self.db.insert_data(table="t_bias_detections", data=table_data, updated_at=True)


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
        "gamblers fallacy",
        "loss aversion",
        "sunk cost fallacy",
        "transaction utility",
    ]
    scenarios: List[str] = [
        "all",
        "0_normal",
        "1_persona",
        "2_odd_numbers",
        "3_large_numbers",
    ]
    models: List[str] = [
        "all",
        "claude-3-haiku",
        "claude-3.5-sonnet",
        "gemma2",
        "gemma2:27b",
        "gpt-4o-mini",
        "gpt-4o",
        "llama3.1",
        "llama3.1:70b",
        "phi3.5",
        "phi3:medium",
    ]
    temperatures: List[float] = [-1, 0.2, 0.7, 1, 1.3, 1.8]

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
    bd.fetch_computable_experiments()

    # Run the bias detection
    if (
        input(
            "\nAre you sure you want to detect the biases for these combinations? (y/n): "
        ).lower()
        == "y"
    ):
        bd.detect_bias()
        bd.modify_detected_biases()
