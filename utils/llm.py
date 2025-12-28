from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv
from google import genai
from openai import OpenAI

load_dotenv()

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt, system_prompt=""):
        pass

class GeminiLLM(BaseLLM):
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model_name = model_name
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate(self, prompt, system_prompt=""):
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config= genai.types.GenerateContentConfig(
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


class KimiLLM(BaseLLM):
    def __init__(self, model_name="kimi-k2-0905-preview"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv("KIMI_API_KEY"),
            base_url="https://api.moonshot.ai/v1"
        )

    def generate(self, prompt, system_prompt=""):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            
            # Extract tokens from usage
            usage = response.usage
            token_count = usage.total_tokens if usage else 0
            
            return response.choices[0].message.content, token_count
        except Exception as e:
            print(f"Error in Kimi generation: {e}")
            return "", 0
