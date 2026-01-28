import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from .logging import get_logger
import time

load_dotenv()

logger = get_logger(__name__)

class GeminiLLM:
    def __init__(self):
        self.model = "gemini-3-flash-preview"
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.last_tokens = 0
        self.last_elapsed = 0.0

    def generate(self, messages):
        try:
            # Build contents using types.Content (matches AI Studio format)
            contents = []
            for msg in messages:
                contents.append(
                    types.Content(
                        role=msg["role"],
                        parts=[types.Part.from_text(text=msg["content"])],
                    )
                )

            start = time.time()
            config = genai.types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="MINIMAL")
            )

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )
            elapsed = time.time() - start

            # Extract tokens from usage_metadata
            usage = response.usage_metadata
            token_count = usage.total_token_count if usage else 0
            
            self.last_tokens = token_count
            self.last_elapsed = elapsed
            
            logger.info(f"llm_call | tokens={token_count} | elapsed={elapsed:.2f}s\n"
                        f"--- PROMPT ---\n{contents}\n"
                        f"--- RESPONSE ---\n{response.text}\n"
                        f"--- END ---")

            return response.text
        except Exception as e:
            print(f"Error in Gemini generation: {e}")
            return ""