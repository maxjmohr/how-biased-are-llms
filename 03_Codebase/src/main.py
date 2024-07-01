import argparse

if __name__ == "__main__":
    # Parse arguments
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="Run Bias Experiments",
        description="Run one of the experiments for a specific bias on one of the models.",
        epilog="Output saved in the 'output' folder.",
        usage="%(prog)s [options]",
    )
    parser.add_argument(
        "-b",
        "--bias",
        type=str,
        choices=["a", "b", "c"],
        help="bias to study",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        choices=["standard", "odd_numbers", "c"],
        help="experiment to run",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--local",
        type=bool,
        default=False,
        choices=[True, False],
        help="run the experiment locally (default: False)",
        required=False,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["gpt-3.5-turbo", "gpt-4", "llama2", "llama3"],
        help="model to use",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--test",
        type=bool,
        default=False,
        choices=[True, False],
        help="pipeline test mode (default: False)",
        required=False,
    )
    args: argparse.Namespace = parser.parse_args()
