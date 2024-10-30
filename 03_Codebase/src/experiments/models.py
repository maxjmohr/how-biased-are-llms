from datetime import datetime
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.replicate import Replicate
import ollama
import os
from pydantic import BaseModel
from pydantic_core import ValidationError
import re
from typing import Literal


class ExperimentOutput(BaseModel):
    # Output format for experiments

    response: str | int
    reason: str


class ModelInteractor:
    def __init__(
        self,
        model: str = "",
        api_key: str = "",
        local: bool = False,
        temperature: float = 0.7,
        request_timeout: int = 36000, # 10 hours
        persona: str = "",
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
            "gpt-4o-mini",
            "gpt-4o",
            "llama3.1",
            "llama3.1:70b",
            "phi3.5",
            "phi3:medium",
        ], f"{datetime.now()} | Model is required"

        # Create system prompt
        system_prompt: str = "You will be asked to make choices. Please blank out that some information might be missing or that you might not be able to make a choice. I have initialized you in a way that you can only generate one token. The only valid answer is A SINGLE LETTER OR NUMBER."
        system_prompt += f" {persona}" if persona != "" else ""

        # Initialize the model
        if local:
            self.llm = self.ollama(
                model=model,
                temperature=temperature,
                request_timeout=request_timeout,
                system_prompt=system_prompt,
            )
        elif model not in ["gpt-4o-mini", "gpt-4o"]:
            self.llm = self.replicate(model=model)
        elif model in ["gpt-4o-mini", "gpt-4o"]:
            self.llm = self.openai(
                model=model,
                api_key=api_key,
                temperature=temperature,
                system_prompt=system_prompt,
            )

    @staticmethod
    def ollama(
        model: str = "",
        temperature: float = 0.7,
        request_timeout: int = 600,
        system_prompt: str = "",
    ) -> Ollama:
        """Initialize the Ollama class
        Parameters:
        model: str
            Model to use
        temperature: float
            Temperature for sampling
        request_timeout: int
            Timeout for the request in seconds
        system_prompt: str
            System prompt to use
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
        return Ollama(
            model=model,
            temperature=temperature,
            request_timeout=request_timeout,
            system_prompt=system_prompt,
            additional_kwargs={
                "num_predict": 2,  # Number of tokens to predict
                # "repeat_last_n": 0 # Number of last responses (0 means no repetition)
            },
        )

    @staticmethod
    def openai(
        model: str = "",
        api_key: str = "",
        temperature: float = 0.7,
        system_prompt: str = "",
    ) -> OpenAI:
        """Initialize the OpenAI class
        Parameters:
        model: str
            Model to use for the API
        apikey: str
            API key to access OpenAI API
        temperature: float
            Temperature for sampling
        system_prompt: str
            System prompt to use
        Returns:
        OpenAI
            OpenAI class
        """
        if api_key == "":
            api_key = str(os.getenv("OPENAI_API_KEY"))
        assert api_key != "", f"{datetime.now()} | API key is required"

        return OpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            system_prompt=system_prompt,
        )

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
        additional_system_message: str = "",
    ) -> tuple[ExperimentOutput, Literal[1, 0]]:
        """Get the prompt for the experiment
        Parameters:
        total_content: str
            Content for the experiment
        system_message: str
            System message for the experiment
        additional_system_message: str
            Additional system message for the experiment
        Returns:
        PromptTemplate
            Prompt for the experiment
        """
        assert total_content != "", f"{datetime.now()} | Experiment content is required"

        if system_message == "":
            system_message = "You are forced to choose! Answer the experiment by only giving the letter of the answer options (e.g. 'A', 'B', 'C', ...) or a numerical value (80, 100, 1000, ...). Do not state anything else! Afterwards, state a short reason in 1-2 sentences for your choice (hard cap at 2 sentences). Do not halucinate. Your 'response' output is only the single letter/integer (such as A/B/C... or 80, 100, 1000,)!\n"
        # OVERWRITE SYSTEM_MESSAGE AS IT WAS FIRST IMPLEMENTED FALSELY, SO THERE WAS NO MESSAGE
        system_message = ""
        system_message += additional_system_message if additional_system_message else ""

        entire_message: str = (
            system_message + "\n---------------------\n" + total_content
        )
        prompt = PromptTemplate(entire_message)

        # Try the structured prediciton, sometimes it doesnt work with smaller models, then just use the completion
        try:
            print(f"{datetime.now()} | Trying structured prediction")
            total_response = ExperimentOutput.parse_obj(
                self.llm.structured_predict(
                    output_cls=ExperimentOutput,  # type: ignore
                    prompt=prompt,
                )
            )

            # Make sure if there are letters, it is only one letter
            if (
                len(str(total_response.response)) != 1
                and str(total_response.response).isalpha()
            ):
                print(f"{datetime.now()} | Structured prediction failed")
                raise ValueError("Structured prediction response format failed")
            else:
                correct_run = 1

        except (ValueError, ValidationError) as e:
            print(f"Error encountered: {e}")  # Debugging line

            # First we try to extract the outputs manually
            if isinstance(e, ValidationError):
                print(f"{datetime.now()} | Trying manual extraction")
                match_response = re.search(r'"response"\s*:\s*"([^"]+)"', str(e))
                match_reason = re.search(r'"reason"\s*:\s*"([^"]+)"', str(e))

                # Especially the response matters
                if match_response:
                    response = match_response.group(1).strip()
                    # Make sure if there are letters, it is only one letter
                    if len(response) == 1 and response.isalpha():
                        reason = match_reason.group(1) if match_reason else ""
                        correct_run = 1 if reason != "" else 0
                        total_response = ExperimentOutput(
                            response=response, reason=reason
                        )

                        return total_response, correct_run

            # If we cannot extract the response, we try to complete the prompt without reasoning
            try:
                print(f"{datetime.now()} | Trying completion without reasoning")
                system_message_onlyresponse: str = "You are forced to choose! Answer the experiment by only giving the letter of the answer options (e.g. 'A', 'B', 'C', ...) or a numerical value (80, 100, 1000, ...). Do not state anything else! Do not halucinate. Your output is only the single letter/integer (such as A/B/C... or 80, 100, 1000,).\n"
                system_message_onlyresponse += (
                    additional_system_message if additional_system_message else ""
                )

                entire_message_onlyresponse: str = (
                    system_message_onlyresponse
                    + "\n---------------------\n"
                    + total_content
                )
                total_response = ExperimentOutput(
                    response=str(
                        self.llm.complete(entire_message_onlyresponse)
                    ).strip(),
                    reason="Prompt without reasoning",
                )

                # Make sure if there are letters, it is only one letter
                if (
                    len(str(total_response.response)) != 1
                    and str(total_response.response).isalpha()
                ):
                    print(f"{datetime.now()} | Structured prediction failed")
                    raise ValueError("Structured prediction response format failed")
                else:
                    correct_run = 1

            except (Exception, ValueError) as e:
                print(f"Error: {e}")
                total_response = ExperimentOutput(
                    response="Failed prompt", reason="Failed prompt"
                )
                correct_run = 0

        return total_response, correct_run

    def prompt_unstructured(
        self,
        total_content: str,
        system_message: str = "",
        additional_system_message: str = "",
    ) -> tuple[ExperimentOutput, Literal[1, 0]]:
        """Prompt the model without reasoning output and without structured prediction
        Parameters:
        total_content: str
            Content for the experiment
        system_message: str
            System message for the experiment
        additional_system_message: str
            Additional system message for the experiment
        Returns:
        PromptTemplate
            Prompt for the experiment
        """
        assert total_content != "", f"{datetime.now()} | Experiment content is required"

        if system_message == "":
            system_message = "You are forced to choose! Answer the experiment by only giving the letter of the answer options (e.g. 'A', 'B', 'C', ...) or a numerical value (80, 100, 1000, ...). Do not state anything else! Do not halucinate. You can and are advised to make a SUBJECTIVE decision! There is no right or wrong, but you HAVE TO DECIDE.\n"
        # OVERWRITE SYSTEM_MESSAGE AS IT WAS FIRST IMPLEMENTED FALSELY, SO THERE WAS NO MESSAGE
        system_message = ""
        system_message += additional_system_message if additional_system_message else ""

        entire_message: str = (
            system_message
            + "\n---------------------\n"
            + total_content
            + "\n---------------------\n"
            + "Example question 1: 'What is the best color? A) Red B) Blue C) Green __'\nC\nExample question 2: 'What value would you sell this car for? $ __'\n8\nUnderstand if you have to choose a letter or answer with a number and then ONLY OUTPUT THE LETTER/NUMBER. BUT DO NOT ATTACH YOUR ANSWER TO THE EXAMPLE QUESTIONS.\n"
        )
        # print(entire_message)

        # Try the structured prediciton, sometimes it doesnt work with smaller models, then just use the completion
        try:
            print(
                f"{datetime.now()} | Trying completion without reasoning (no structured prediction)"
            )
            total_response = ExperimentOutput(
                response=str(self.llm.complete(entire_message)).strip(),
                reason="Prompt without reasoning",
            )
            print(f"Initial response: '{total_response.response}'")

            # Make sure if there are letters, it is only one letter
            if (
                (
                    len(str(total_response.response)) == 1
                    and str(total_response.response).isalpha()
                )  # Single letter
                or str(total_response.response).isdigit()  # Valid number
            ):
                correct_run = 1
            else:
                print(
                    f"{datetime.now()} | Structured prediction failed, trying extraction with LLM extractor"
                )
                total_response.response = self.response_extractor(
                    str(total_response.response)
                )
                print(f"Extracted response: '{total_response.response}'")
                if (
                    (
                        len(str(total_response.response)) == 1
                        and str(total_response.response).isalpha()
                    )  # Single letter
                    or str(total_response.response).isdigit()  # Valid number
                ):
                    correct_run = 1
                else:
                    print(f"{datetime.now()} | Structured prediction failed")
                    raise ValueError("Structured prediction response format failed")

        except (Exception, ValueError) as e:
            print(f"Error: {e}")
            print(f"{datetime.now()} | Completion without reasoning failed")
            total_response = ExperimentOutput(
                response="Failed prompt", reason=f"Response was '{total_response.response}'"
            )
            correct_run = 0
        print(f"Final response: '{total_response.response}'")

        return total_response, correct_run

    @staticmethod
    def response_extractor(response: str, model: str = "llama3.1") -> str:
        "Leverage a LLM to extract the letter/number of a model response)"
        system_prompt = "You will be given a model response to an experimental question. There are two response types: either a choice (A, B, C, ...) or an integer (0, 4, 26, ...). Some models however add some symbols or other letters to their answer. Extract the letter or integer from the response and only output that. Do not halucinate."
        ollama = Ollama(
            model=model,
            system_prompt=system_prompt,
            additional_kwargs={
                "num_predict": 1,  # Number of tokens to predict
                # "repeat_last_n": 0 # Number of last responses (0 means no repetition)
            },
        )
        entire_message = (
            system_prompt
            + "\n"
            + "This was the model's response: "
            + response
            + "\nOnly output the letter or number of the response. Examples: 'A', 'B', 'C', '1', '2', '3', ... DO NOT OUTPUT ANYTHING ELSE BESIDES THE ONE LETTER OR INTEGER."
        )
        extracted_response = str(ollama.complete(entire_message)).strip()
        return extracted_response



if __name__ == "__main__":
    """
    # Gemma2
    gemma2 = ModelInteractor(model="gemma2", local=True)
    total_response = gemma2.prompt(
        "What is the capital of France? A. Paris B. London C. Berlin"
    )
    print(f"GEMMA2 // Choice: {total_response.response}, Reason: {total_response.reason}")

    # Gemma2:27b
    gemma2_27b = ModelInteractor(model="gemma2:27b", local=True)
    total_response = gemma2_27b.prompt(
        "What is the capital of France? A. Paris B. London C. Berlin"
    )
    print(f"GEMMA2 27B // Choice: {total_response.response}, Reason: {total_response.reason}")

    # Llama3
    llama3 = ModelInteractor(model="llama3.1", local=True)
    total_response = llama3.prompt_unstructured(
        "What is the capital of France? A. Paris B. London C. Berlin"
    )
    print(f"LLAMA3.1 // Response: {total_response}")
    """

    # Llama3:70b
    llama3_70b = ModelInteractor(model="llama3.1:70b", local=True)
    total_response = llama3_70b.prompt_unstructured(
        "What is the capital of France? A. Paris B. London C. Berlin"
    )
    print(f"LLAMA3.1 70B // Response: {total_response}")

"""
    # Phi3
    phi3 = ModelInteractor(model="phi3", local=True)
    total_response = phi3.prompt(
        "What is the capital of France? A. Paris B. London C. Berlin"
    )
    print(f"PHI3 // Choice: {total_response.response}, Reason: {total_response.reason}")

    # Phi3 Medium
    phi3_medium = ModelInteractor(model="phi3:medium", local=True)
    total_response = phi3_medium.prompt(
        "What is the capital of France? A. Paris B. London C. Berlin"
    )
    print(f"PHI3 MEDIUM // Choice: {total_response.response}, Reason: {total_response.reason}")
"""
