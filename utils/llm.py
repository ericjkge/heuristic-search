import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from .logging import get_logger
import time

load_dotenv()

logger = get_logger(__name__)

class GeminiLLM:
    def __init__(self, model_name="gemini-3-flash-preview"):
        self.model_name = model_name
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.last_tokens = 0
        self.last_elapsed = 0.0

    def generate(self, messages, system_prompt=""):
        try:
            # Conver to Gemini format ("model" instead of "assistant")
            contents = []
            for msg in messages:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})

            start = time.time()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config= genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                thinking_config=types.ThinkingConfig(thinking_level="minimal") # Minimal thinking with Gemini 3.0 Flash
                )
            )
            elapsed = time.time() - start

            # Extract tokens from usage_metadata
            usage = response.usage_metadata
            token_count = usage.total_token_count if usage else 0
            
            self.last_tokens = token_count
            self.last_elapsed = elapsed
            
            logger.info(f"llm_call | tokens={token_count} | elapsed={elapsed:.2f}s\n"
                        f"--- SYSTEM PROMPT ---\n{system_prompt}\n"
                        f"--- PROMPT ---\n{contents}\n"
                        f"--- RESPONSE ---\n{response.text}\n"
                        f"--- END ---")

            return response.text
        except Exception as e:
            print(f"Error in Gemini generation: {e}")
            return ""