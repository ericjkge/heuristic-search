from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from .logging import get_logger
import time

load_dotenv()

logger = get_logger(__name__)

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
            start = time.time()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config= genai.types.GenerateContentConfig(
                system_instruction=system_prompt
                )
            )
            elapsed = time.time() - start

            # Extract tokens from usage_metadata
            usage = response.usage_metadata
            token_count = usage.total_token_count if usage else 0
            
            logger.info(f"llm_call | tokens={token_count} | elapsed={elapsed:.2f}s | prompt={prompt}")

            return response.text
        except Exception as e:
            print(f"Error in Gemini generation: {e}")
            return ""


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
            
            start = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            elapsed = time.time() - start
            
            # Extract tokens from usage
            usage = response.usage
            token_count = usage.total_tokens if usage else 0

            logger.info(f"llm_call | tokens={token_count} | elapsed={elapsed:.2f}s | prompt={prompt[:100]}... | system_prompt={system_prompt[:100]}...")

            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in Kimi generation: {e}")
            return ""


class QwenLLM(BaseLLM):
    def __init__(self, model_name="qwen-plus"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" # China endpoint
        )

    def generate(self, prompt, system_prompt=""):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            start = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            elapsed = time.time() - start
            
            # Extract tokens from usage
            usage = response.usage
            token_count = usage.total_tokens if usage else 0

            logger.info(f"llm_call | tokens={token_count} | elapsed={elapsed:.2f}s | prompt={prompt[:100]}... | system_prompt={system_prompt[:100]}...")

            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in Qwen generation: {e}")
            return ""
