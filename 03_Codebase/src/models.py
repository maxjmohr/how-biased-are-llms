from datetime import datetime
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.replicate import Replicate
import os
from pydantic import BaseModel


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
        """
        assert model in [
            "gemma2",
            "gemma2:27b",
            "gpt-3.5-turbo",
            "gpt-4o",
            "llama3",
            "llama3:70b",
            "phi3:mini",
            "phi3:medium",
        ], f"{datetime.now()} | Model is required"
        if model not in ["gpt-3.5-turbo", "gpt-4o"] and local:
            self.llm = self.llama(model=model, temperature=temperature)
        elif model in ["gpt-3.5-turbo", "gpt-4o"] and not local:
            self.llm = self.replicate(model=model)
        elif model in ["gpt-3.5-turbo", "gpt-4o"]:
            self.llm = self.openai(
                model=model, api_key=api_key, temperature=temperature
            )

    @staticmethod
    def llama(model: str = "", temperature: float = 0.7) -> Ollama:
        """Initialize the Llama class
        Parameters:
        model: str
            Model to use
        temperature: float
            Temperature for sampling
        Returns:
        Ollama
            Llama class
        """
        return Ollama(model=model, temperature=temperature)

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
    ):
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
            system_message = "Please answer the experiment by only giving the letter of the answer options (e.g. A, B, C, ...). Afterwards, state a short reason in 1-2 sentences for your choice. \n"
        entire_message = system_message + "---------------------\n" + "{total_content}"
        prompt = PromptTemplate(entire_message)

        return self.llm.structured_predict(
            output_class, prompt, total_content=total_content
        )
