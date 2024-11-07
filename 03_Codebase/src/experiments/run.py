from datetime import datetime
import os
from src.experiments.models import ModelInteractor
from tqdm import trange
from typing import List, Tuple


def run_experiment(
    bias: str,
    scenario: str,
    response_type: str,
    total_content: str,
    model: str,
    local: bool = False,
    temperature: float = 0.7,
    n: int = 100,
    test: bool = False,
    reason: bool = False,
) -> Tuple[List[str], List[str], List[int]]:
    """Run an experiment for a specific bias on one of the models
    Parameters:
    bias: str
        Bias to study
    scenario: str
        Scenario of the experiment
    response_type: str
        Type of response
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
    reason: bool
        Whether to activate reasoning mode
    """
    # Add additional message to prompt (e.g. persona)
    if scenario in ["1_persona", "2_odd_numbers", "3_large_numbers"]:
        # Read txt file
        current_script_directory: str = os.path.dirname(os.path.realpath(__file__))
        path: str = "../../res/prompt_additions/persona.txt"
        total_path: str = os.path.join(current_script_directory, path)
        with open(total_path, "r") as f:
            additional_system_message: str = f.read()
    else:
        additional_system_message: str = ""

    # Initialize the model interactor
    print(f"{datetime.now()} | Initializing model interactor for model {model}")
    mi = ModelInteractor(
        model=model,
        local=local,
        temperature=temperature,
        persona=additional_system_message,
    )
    print(f"{datetime.now()} | Initialized model interactor for model {model}")

    # Check if test mode is activated
    if test:
        print(
            f"{datetime.now()} | Bias {bias} ({scenario}) on model {model} (temp={temperature}) in test mode (loops=1)"
        )
        n = 1
    else:
        print(
            f"{datetime.now()} | Bias {bias} ({scenario}) on model {model} (temp={temperature}) for loops={n}"
        )

    # Store responses and whether they are in the correct format
    responses: List[str] = [""] * n
    reasons: List[str] = [""] * n
    correct_runs: List[int] = [0] * n

    # Run the experiment
    for i in trange(
        n, desc=f"Bias {bias} ({scenario}) on model {model} (temp={temperature})"
    ):
        # Prompt the model
        if reason:
            total_response, correct_run = mi.prompt(
                total_content, additional_system_message=additional_system_message
            )
        else:
            total_response, correct_run = mi.prompt_unstructured(
                total_content,
                additional_system_message=additional_system_message,
                response_type=response_type,
            )
        try:
            responses[i] = str(total_response.response)
            reasons[i] = total_response.reason
            correct_runs[i] = correct_run
        except IndexError:
            print(f"Index {i} is out of range. List length: {len(responses)}")
            continue

    print(
        f"{datetime.now()} | Finished experiment scenario {scenario} for bias {bias} on model {model}"
    )
    return responses, reasons, correct_runs
