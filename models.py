from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> tuple[str, int]:
        """
        Generates content based on the prompt.
        Returns:
            tuple[str, int]: (response_text, token_usage)
        """
        pass

class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        # Uses "GEMINI_API_KEY" in .env
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate(self, prompt: str, system_prompt: str = "") -> tuple[str, int]:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config= types.GenerateContentConfig(
                system_instruction=system_prompt
                )
            )
            
            # Extract tokens from usage_metadata
            usage = response.usage_metadata
            token_count = usage.total_token_count if usage else 0
            
            return response.text, token_count
        except Exception as e:
            print(f"Error in Gemini generation: {e}")
            return "", 0
