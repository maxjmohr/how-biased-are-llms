import argparse
from datetime import datetime
import pandas as pd
import platform
import os
from src.data.db_helpers import Database
from src.data.manipulations import (
    calc_remaining_loops,
    combine_content_variables,
    filter_parser_args,
)
from src.experiments.run import run_experiment
from typing import List
# import faulthandler

if __name__ == "__main__":
    # faulthandler.enable()
    # Parse arguments
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="Run Bias Experiments",
        description="Run one of the experiments for a specific bias on one of the models. If you want to run any of the experiments automized, don't use any of the filter tags.",
        epilog="The resonses will be saved in t_responses in the PostgreSQL database.",
        usage="%(prog)s [options]",
    )
    parser.add_argument(
        "-b",
        "--bias",
        type=str,
        choices=[
            "anchoring",
            "category size bias",
            "endowment effect",
            "framing effect",
            "gamblers fallacy",
            "loss aversion",
            "sunk cost fallacy",
            "transaction utility",
        ],
        help="optional filter for bias",
        required=False,
    )
    parser.add_argument(
        "-bid",
        "--bias_id",
        type=int,
        help="optional filter if bias id is known",
        required=False,
    )
    parser.add_argument(
        "-c",
        "--cluster",
        type=bool,
        choices=[True, False],
        help="optional flag to indicate if experiment is run on cluster",
        required=False,
    )
    parser.add_argument(
        "-eid",
        "--experiment_id",
        type=int,
        help="optional filter if experiment id is known",
        required=False,
    )
    parser.add_argument(
        "-l",
        "--local",
        type=bool,
        choices=[True, False],
        help="optional filter to run experiment locally",
        required=False,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=[
            "gemma2",
            "gemma2:27b",
            "gpt-4o-mini",
            "gpt-4o",
            "llama3.1",
            "llama3.1:70b",
            "phi3.5",
            "phi3:medium",
        ],
        help="optional filter for model",
        required=False,
    )
    parser.add_argument(
        "-mid",
        "--model_id",
        type=int,
        help="optional filter if model id is known",
        required=False,
    )
    parser.add_argument(
        "-n",
        "--n_loops",
        type=int,
        help="optional filter how often an experiment should correctly be run",
        required=False,
    )
    parser.add_argument(
        "-r",
        "--reason",
        type=bool,
        choices=[True, False],
        help="optional model output to include reason for choice",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--scenario",
        type=str,
        choices=["0_normal", "1_persona"],
        help="optional filter for experiment scenario",
        required=False,
    )
    parser.add_argument(
        "-t",
        "--test",
        type=bool,
        choices=[True, False],
        help="activate test mode",
        required=False,
    )
    args: argparse.Namespace = parser.parse_args()

    # Initialize and connect to the database
    db: Database = Database()
    db.connect()

    try:
        # Get all runnable experiments
        n: int = 100
        if args.n_loops:
            n = args.n_loops
        if args.cluster:
            experiments: pd.DataFrame = db.fetch_next_experiments(
                n_loops=n, check_system=False
            )
        elif args.model and (
            # Ask user in terminal if they are sure to run the model on the selected device
            input(
                "Are you sure you want to run the experiments on the selected model on this device (y/n)? "
            ).lower()
            == "y"
        ):
            experiments: pd.DataFrame = db.fetch_next_experiments(
                n_loops=n, check_system=False
            )
        else:
            experiments: pd.DataFrame = db.fetch_next_experiments(n_loops=n)

        # Perhaps activate test mode
        test: bool = False
        if args.test:
            test = args.test

        # Check whether reason is requested
        reason: bool = False
        if args.reason:
            reason = args.reason

        # Filter for experiments if arguments parsed
        experiments = filter_parser_args(experiments, args)

        # Create total_content of experiment by combining content and variables
        experiments["total_content"] = experiments.apply(
            lambda row: combine_content_variables(row["content"], row["variables"]),
            axis=1,
        )

        # Try to run through as many experiments as possible
        while not experiments.empty:
            # Get first experiment
            experiment: pd.DataFrame = experiments.head(1)

            # Check if currently any experiments are running and if so, if it is the same experiment
            currently_running: List[int] = list(
                db.fetch_data(total_object="t_currently_running")["experiment_id"]
            )
            if experiment["experiment_id"].iloc[0] in currently_running:
                # Remove experiment from df
                experiments = experiments.iloc[1:]
                continue
            else:  # Add experiment to currently running
                print(f"{datetime.now()} | Adding experiment to t_currently_running")
                db.insert_data(
                    table="t_currently_running",
                    data=pd.DataFrame(
                        {
                            "experiment_id": experiment["experiment_id"],
                            "system": platform.system(),
                        }
                    ),
                    updated_at=True,
                )

            # Get remaining loops
            print(f"{datetime.now()} | Calculating remaining loops")
            # Fetch the ran loops (perhaps in the meantime, the experiment was run by another process)
            sql = f"SELECT correct_ran_loops FROM v_ran_experiments WHERE experiment_id = {experiment['experiment_id'].iloc[0]}"
            correct_ran_loops: int | None = int(
                db.fetch_data(sql=sql)["correct_ran_loops"].iloc[0]
            )
            if correct_ran_loops is None:
                correct_ran_loops = 0
            n_remain: int = calc_remaining_loops(
                target_loops=n, correct_runs=correct_ran_loops
            )

            # Some checks
            # print(f"total_content: {experiment['total_content'].iloc[0]}")
            # print(f"experiment_id: {experiment['experiment_id'].iloc[0]}")

            # Run the experiment
            responses, reasons, correct_runs = run_experiment(
                bias=experiment["bias"].iloc[0],
                scenario=experiment["scenario"].iloc[0],
                response_type=experiment["response_type"].iloc[0],
                total_content=experiment["total_content"].iloc[0],
                model=experiment["model"].iloc[0],
                local=experiment["local"].iloc[0],
                temperature=experiment["temperature"].iloc[0],
                n=n_remain,
                test=test,
                reason=reason,
            )

            # Add missing columns and insert responses into database
            responses_df: pd.DataFrame = pd.DataFrame(
                {
                    "experiment_id": experiment["experiment_id"].iloc[0],
                    "bias_id": experiment["bias_id"].iloc[0],
                    "model_id": experiment["model_id"].iloc[0],
                    "response_type": experiment["response_type"].iloc[0],
                    "response": responses,
                    "reason": reasons,
                    "correct_run": correct_runs,
                }
            )
            # Save intermediate responses
            responses_df.to_csv(
                f"{experiment['experiment_id'].iloc[0]}.csv", index=False
            )
            if test:  # If test mode, break here
                print(f"response: {responses_df['response'].iloc[0]}")
                print(f"reason: {responses_df['reason'].iloc[0]}")

                # Delete experiment from currently running
                db.delete_data(
                    total_object="t_currently_running",
                    sql=f"""
                    DELETE FROM t_currently_running
                    WHERE experiment_id = '{experiment["experiment_id"].iloc[0]}';
                    """,
                    definitely_delete=True,
                )
                break

            if not responses_df.empty:
                # If process beforehand was too long, reconnect to database
                try:
                    db.insert_data(
                        table="t_responses", data=responses_df, updated_at=True
                    )
                except Exception:
                    db.connect()
                    db.insert_data(
                        table="t_responses", data=responses_df, updated_at=True
                    )

            # Delete intermediate responses
            os.remove(f"{experiment['experiment_id'].iloc[0]}.csv")

            # Delete experiment from currently running
            db.delete_data(
                total_object="t_currently_running",
                sql=f"""
                DELETE FROM t_currently_running
                WHERE experiment_id = {experiment["experiment_id"].iloc[0]};
                """,
                definitely_delete=True,
            )

            # Remove experiment from df
            experiments = experiments.iloc[1:]

    finally:
        # Close the database connection
        print(
            f"{datetime.now()} | Closing database connection and deleting currently running experiments of this system"
        )
        sql = f"""
        DELETE
        FROM t_currently_running
        WHERE system = '{platform.system()}';
        """
        db.execute_sql(sql=sql)
        db.disconnect()
