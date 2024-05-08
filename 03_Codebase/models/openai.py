from datetime import datetime
from openai import OpenAI
from typing import List, Dict


class OpenAIInteractor:
    "Class to interact with OpenAI API"

    def __init__(self, api_key: str = "", model: str = "") -> None:
        """Initialize the OpenAIGPT class
        Parameters:
        api_key: str
            API key to access OpenAI API
        model: str
            Model to use for the API
        """
        assert api_key != "", f"{datetime.now()} | API key is required"
        assert model != "", f"{datetime.now()} | Model is required"
        self.api_key = api_key
        # self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    @staticmethod
    def format_system_message(system_message: str) -> Dict[str, str]:
        """Format the system message
        Parameters:
        system_message: str
            System message to send to the chatbot
        Returns:
        dict
            Formatted system message
        """
        return {"role": "system", "content": system_message}

    @staticmethod
    def format_user_message(user_message: str) -> Dict[str, str]:
        """Format the user message
        Parameters:
        user_message: str
            User message to send to the chatbot
        Returns:
        dict
            Formatted user message
        """
        return {"role": "user", "content": user_message}

    def get_response(
        self, system_message: str, user_message: str, other_parameters: Dict[str, float]
    ) -> str:
        """Get the response of the chatbot
        Parameters:
        system_message: str
            System message to send to the chatbot
        user_message: str
            User message to send to the chatbot
        other_parameters: dict
            Other parameters to send to the chatbot
        Returns:
        str
            Response from the chatbot
        """
        # Retrieve and format messages
        assert user_message != "", f"{datetime.now()} | User message is required"
        messages: List[Dict[str, str]] = [
            self.format_system_message(system_message),
            self.format_user_message(user_message),
        ]

        # Retrieve OpenAI specific parameters
        temperature: float = other_parameters.get("temperature", 0.5)

        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature
        )
        response_content = response.choices[0].message.content
        if response_content is not None:
            return response_content.strip()
        else:
            return ""
