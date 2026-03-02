import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from .logging import get_logger

load_dotenv()

logger = get_logger(__name__)


class GeminiLLM:
    def __init__(self) -> None:
        self.model = "gemini-3-flash-preview"
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.last_tokens = 0
        self.last_elapsed = 0.0

    def generate(self, prompt: str) -> str:
        try:
            start = time.time()
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="minimal")
            )

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            elapsed = time.time() - start

            usage = response.usage_metadata
            token_count = usage.total_token_count if usage else 0

            self.last_tokens = token_count
            self.last_elapsed = elapsed

            logger.info(f"llm_call | tokens={token_count} | elapsed={elapsed:.2f}s\n"
                        f"--- PROMPT ---\n{prompt}\n"
                        f"--- RESPONSE ---\n{response.text}\n"
                        f"--- END ---")

            return response.text
        except Exception as e:
            print(f"Error in Gemini generation: {e}")
            return ""

class OpenAILLM:
    def __init__(self) -> None:
        self.model = "gpt-5.2"
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="http://14.103.68.46/v1")
        self.last_tokens = 0
        self.last_elapsed = 0.0

    def generate(self, prompt: str) -> str:
        try:
            start = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.time() - start

            token_count = response.usage.total_tokens if response.usage else 0
            text = response.choices[0].message.content or ""

            self.last_tokens = token_count
            self.last_elapsed = elapsed

            logger.info(f"llm_call | tokens={token_count} | elapsed={elapsed:.2f}s\n"
                        f"--- PROMPT ---\n{prompt}\n"
                        f"--- RESPONSE ---\n{text}\n"
                        f"--- END ---")

            return text
        except Exception as e:
            print(f"Error in OpenAI generation: {e}")
            return ""