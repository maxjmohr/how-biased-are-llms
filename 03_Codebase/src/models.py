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
    # Class to interact with large language models

    def __init__(self, model: str = "", api_key: str = "", local: bool = False) -> None:
        """Initialize the model class
        Parameters:
        model: str
            Model to use
        api_key: str
            API key to access OpenAI API
        """
        assert model in [
            "gpt-3.5-turbo",
            "-gpt4",
            "gpt-4o",
            "llama2",
            "llama3",
        ], f"{datetime.now()} | Model is required"
        if model in ["llama2", "llama3"] and local:
            self.llm = self.llama(model=model)
        elif model in ["llama2", "llama3"] and not local:
            self.llm = self.replicate(model=model)
        elif model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]:
            self.llm = self.openai(model=model, api_key=api_key)

    @staticmethod
    def llama(model: str = "") -> Ollama:
        """Initialize the Llama class
        Parameters:
        model: str
            Model to use
        Returns:
        Ollama
            Llama class
        """
        return Ollama(model=model)

    @staticmethod
    def openai(model: str = "", api_key: str = "") -> OpenAI:
        """Initialize the OpenAI class
        Parameters:
        model: str
            Model to use for the API
        apikey: str
            API key to access OpenAI API
        Returns:
        OpenAI
            OpenAI class
        """
        if api_key == "":
            api_key = str(os.getenv("OPENAI_API_KEY"))
        assert api_key != "", f"{datetime.now()} | API key is required"

        return OpenAI(model=model, api_key=api_key)

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
        experiment: str,
        system_message: str = "",
        output_class: BaseModel = ExperimentOutput,
    ):
        """Get the prompt for the experiment
        Parameters:
        experiment: str
            Experiment to run
        user_message: str
            User message to send to the chatbot
        Returns:
        PromptTemplate
            Prompt for the experiment
        """
        assert experiment != "", f"{datetime.now()} | Experiment is required"

        if system_message != "":
            system_message = (
                "Please answer by only giving the letter of the answer option A or B."
            )
        entire_message = system_message + " {experiment}"
        prompt = PromptTemplate(entire_message)

        return self.llm.structured_predict(
            ExperimentOutput, prompt, experiment=experiment
        )
