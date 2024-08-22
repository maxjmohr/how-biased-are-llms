from datetime import datetime
from experiments.models import ModelInteractor
from tqdm import trange
from typing import List, Tuple


def run_experiment(
    bias: str,
    scenario: str,
    total_content: str,
    model: str,
    local: bool = False,
    temperature: float = 0.7,
    n: int = 100,
    test: bool = False,
) -> Tuple[List[str | int], List[str], List[int]]:
    """Run an experiment for a specific bias on one of the models
    Parameters:
    bias: str
        Bias to study
    scenario: str
        Scenario of the experiment
    local: bool,
        Run the model locally
    model: str,
        Model to use
    temperature: float
        Temperature for model
    n: int
        Number of loops
    test: bool
        Whether to activate test mode
    """
    # Initialize the model interactor
    mi = ModelInteractor(model=model, local=local, temperature=temperature)

    # Check if test mode is activated
    if test:
        print(
            f"{datetime.now()} | Running experiment scenario {scenario} for bias {bias} on model {model} in test mode (loops=1)"
        )
        n = 1
    else:
        print(
            f"{datetime.now()} | Running experiment scenario {scenario} for bias {bias} on model {model} (loops={n})"
        )

    # Store responses and whether they are in the correct format
    responses: List[str | int] = [] * n
    reasons: List[str] = [""] * n
    correct_runs: List[int] = [0] * n

    # Run the experiment
    for i in trange(n, desc=f"Scenario {scenario} for bias {bias} on model {model}"):
        total_response, correct_run = mi.prompt(total_content)
        responses[i] = total_response.response
        reasons[i] = total_response.reason
        correct_runs[i] = correct_run

    return responses, reasons, correct_runs
