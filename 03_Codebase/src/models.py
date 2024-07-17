from datetime import datetime
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.replicate import Replicate
import ollama
import os
from pydantic import BaseModel
from pydantic_core import ValidationError


class ExperimentOutput(BaseModel):
    # Output format for experiments

    choice: str
    reason: str


class ModelInteractor:
    def __init__(
        self,
        model: str = "",
        api_key: str = "",
        local: bool = False,
        temperature: float = 0.7,
        request_timeout: int = 600,
    ) -> None:
        """Initialize the model class
        Parameters:
        model: str
            Model to use
        api_key: str
            API key to access OpenAI API
        local: bool
            Run the model locally
        temperature: float
            Temperature for sampling
        request_timeout: int
            Timeout for the request in seconds
        """
        assert model in [
            "gemma2",
            "gemma2:27b",
            "gpt-3.5-turbo",
            "gpt-4o",
            "llama3",
            "llama3:70b",
            "phi3",
            "phi3:medium",
        ], f"{datetime.now()} | Model is required"
        if model not in ["gpt-3.5-turbo", "gpt-4o"] and local:
            self.llm = self.ollama(model=model, temperature=temperature, request_timeout=request_timeout)
        elif model in ["gpt-3.5-turbo", "gpt-4o"] and not local:
            self.llm = self.replicate(model=model)
        elif model in ["gpt-3.5-turbo", "gpt-4o"]:
            self.llm = self.openai(
                model=model, api_key=api_key, temperature=temperature
            )

    @staticmethod
    def ollama(model: str = "", temperature: float = 0.7, request_timeout: int = 600) -> Ollama:
        """Initialize the Ollama class
        Parameters:
        model: str
            Model to use
        temperature: float
            Temperature for sampling
        request_timeout: int
            Timeout for the request in seconds
        Returns:
        Ollama
            Model class
        """
        # Check if model is downloaded, else pull
        try:
            ollama.show(model)
        except ollama.ResponseError:
            ollama.pull(model)

        # Initialize the model
        return Ollama(model=model, temperature=temperature, request_timeout=request_timeout)

    @staticmethod
    def openai(model: str = "", api_key: str = "", temperature: float = 0.7) -> OpenAI:
        """Initialize the OpenAI class
        Parameters:
        model: str
            Model to use for the API
        apikey: str
            API key to access OpenAI API
        temperature: float
            Temperature for sampling
        Returns:
        OpenAI
            OpenAI class
        """
        if api_key == "":
            api_key = str(os.getenv("OPENAI_API_KEY"))
        assert api_key != "", f"{datetime.now()} | API key is required"

        return OpenAI(model=model, api_key=api_key, temperature=temperature)

    @staticmethod
    def replicate(model: str = "") -> Replicate:
        """Initialize the Replicate class
        Parameters:
        model: str
            Model to use
        Returns:
        Replicate
            Replicate class
        """
        # Map the model to the correct model name
        map = {
            "llama2": "meta/llama-2-70b-chat",
            "llama3": "meta/meta-llama-3-70b-instruct",
        }
        model = map[model]

        api_key = str(os.getenv("REPLICATE_API_TOKEN"))
        assert (
            api_key != ""
        ), f"{datetime.now()} | API key is required, please set global environment variable REPLICATE_API_TOKEN"

        return Replicate(model=model)

    def prompt(
        self,
        total_content: str,
        system_message: str = "",
        output_class: type[BaseModel] = ExperimentOutput,
    ) -> ExperimentOutput:
        """Get the prompt for the experiment
        Parameters:
        total_content: str
            Content for the experiment
        system_message: str
            System message for the experiment
        output_class: type[BaseModel]
            Output class for the experiment
        Returns:
        PromptTemplate
            Prompt for the experiment
        """
        assert total_content != "", f"{datetime.now()} | Experiment content is required"

        if system_message != "":
            system_message = "Please answer the experiment by only giving the letter of the answer options (e.g. 'A', 'B', 'C', ...) without stating anything else! Afterwards, state a short reason in 1-2 sentences for your choice. \n"
        entire_message = system_message + "---------------------\n" + "{total_content}"
        prompt = PromptTemplate(entire_message)

        # Try the structured prediciton, sometimes it doesnt work with smaller models, then just use the completion
        try:
            response = self.llm.structured_predict(
                output_class, prompt, total_content=total_content
            )
        except ValidationError as e:
            print(f"Pydantic Validation Error: {e}")
            entire_message = f"system_message---------------------\n{total_content}"
            response = ExperimentOutput(
                choice=str(self.llm.complete(entire_message)), reason=""
            )
        return response


if __name__ == "__main__":
    # Gemma2
    gemma2 = ModelInteractor(model="gemma2", local=True)
    response = gemma2.prompt(
        "What is the capital of France? A. Paris B. London C. Berlin"
    )
    print(f"GEMMA2 // Choice: {response.choice}, Reason: {response.reason}")

    # Gemma2:27b
    gemma2_27b = ModelInteractor(model="gemma2:27b", local=True)
    response = gemma2_27b.prompt(
        "What is the capital of France? A. Paris B. London C. Berlin"
    )
    print(f"GEMMA2 27B // Choice: {response.choice}, Reason: {response.reason}")

    # Llama3
    llama3 = ModelInteractor(model="llama3", local=True)
    response = llama3.prompt(
        "What is the capital of France? A. Paris B. London C. Berlin"
    )
    print(f"LLAMA3 // Choice: {response.choice}, Reason: {response.reason}")

    # Llama3:70b
    llama3_70b = ModelInteractor(model="llama3:70b", local=True)
    response = llama3_70b.prompt(
        "What is the capital of France? A. Paris B. London C. Berlin"
    )
    print(f"LLAMA3 70B // Choice: {response.choice}, Reason: {response.reason}")

    # Phi3
    phi3 = ModelInteractor(model="phi3", local=True)
    response = phi3.prompt(
        "What is the capital of France? A. Paris B. London C. Berlin"
    )
    print(f"PHI3 // Choice: {response.choice}, Reason: {response.reason}")

    # Phi3 Medium
    phi3_medium = ModelInteractor(model="phi3:medium", local=True)
    response = phi3_medium.prompt(
        "What is the capital of France? A. Paris B. London C. Berlin"
    )
    print(f"PHI3 MEDIUM // Choice: {response.choice}, Reason: {response.reason}")
