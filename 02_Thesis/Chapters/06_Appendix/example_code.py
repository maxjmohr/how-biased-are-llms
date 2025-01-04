from llama_index.llms.ollama import Ollama
import ollama
from typing import List

####-- SETTINGS --####
model: str = "llama3.1:70b"  # Model to use
temperature: float = 0.7  # Model temperature
max_tokens: int = 2  # Number of tokens to predict
n: int = 100  # Number of runs
response_type: str = "choice"  # Choice or numerical

####-- PROMPTS --####
##- System message -##
# Persona
persona: str = "You are a customer with median income and average education. You are selected at random to participate in a survey. You can only choose one of the presented options or assign a value. Behave humanlike and choose instinctively. You can and are advised to make a subjective decision! There is no right or wrong, but you HAVE TO DECIDE."

# Additional system message
system_message: str = f"You will be asked to make choices. Please blank out that some information might be missing or that you might not be able to make a choice. I have initialized you in a way that you can only generate {max_tokens} tokens. The only valid answer is A SINGLE LETTER OR NUMBER."

# Combine messages
system_message += f" {persona}" if persona != "" else ""

##- User message -##
# User message
user_message: str = "You are forced to choose! Answer the experiment by only giving the letter of the answer options (e.g. A, B, C, ...) or a numerical value (80, 100, 1000, ...). Do not state anything else! Do not hallucinate."

# Question
question: str = "Imagine that you have decided to see a play where admission is $10. As you enter the theater, you discover that you have lost a $10 bill. Would you still pay $10 for a ticket to the play? Choose between A) Yes or B) No. __"

# Tell the model what type of response we are expecting
if response_type == "choice":
    output_message: str = "Your output should only be a LETTER (A, B, C, ...)."
else:
    output_message: str = "Your output should only be a NUMBER (9, 80, 100, 1000, ...)."
# Combine messages
entire_user_message: str = (
    user_message
    + "\n---------------------\n"
    + question
    + "\n---------------------\n"
    + output_message
)

####-- MODEL INITIALIZATION --####
# Check if model is downloaded, else pull
try:
    ollama.show(model)
except ollama.ResponseError:
    ollama.pull(model)

# Initialize the model
model_interactor: Ollama = Ollama(
    model=model,
    temperature=temperature,
    system_prompt=system_message,
    additional_kwargs={
        "num_predict": max_tokens,
    },
)

####-- EXPERIMENT --####
# Store responses and whether they are in the correct format
responses: List[str] = [""] * n
correct_runs: List[int] = [0] * n

# Run the experiment
for i in range(n):
    # Prompt the model and store the response
    response: str = str(model_interactor.complete(entire_user_message)).strip()

    # Make sure if there are letters, it is only one letter
    if (
        (len(str(response)) == 1 and str(response).isalpha())  # Single letter
        or str(response).isdigit()  # Valid number
    ):
        correct_run: int = 1
    else:
        correct_run: int = 0

    responses[i] = response
    correct_runs[i] = correct_run

# Close the model interactor
ollama.generate(model=model, prompt="Goodbye!", keep_alive=0)
