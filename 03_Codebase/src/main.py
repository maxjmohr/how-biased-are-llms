import argparse

if __name__ == "__main__":
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
            "category size bias",
            "endowment effect",
            "loss aversion",
            "sunk cost fallacy",
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
        default=False,
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
            "phi3:mini",
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
        "-s",
        "--scenario",
        type=str,
        choices=["00_normal", "01_odd_numbers"],
        help="optional filter for experiment scenario",
        required=False,
    )
    parser.add_argument(
        "-t",
        "--test",
        type=bool,
        default=False,
        choices=[True, False],
        help="activate test mode",
        required=False,
    )
    args: argparse.Namespace = parser.parse_args()

    # If there are parser arguments, get the right data, else just get the next experiment
