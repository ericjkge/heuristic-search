import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from .logging import get_logger

load_dotenv()

logger = get_logger(__name__)


class GeminiLLM:
    def __init__(self) -> None:
        self.model = "gemini-3-flash-preview"
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.last_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
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
            input_tokens = usage.prompt_token_count if usage else 0
            output_tokens = usage.candidates_token_count if usage else 0

            self.last_tokens = token_count
            self.last_input_tokens = input_tokens
            self.last_output_tokens = output_tokens
            self.last_elapsed = elapsed

            logger.info(f"llm_call | tokens={token_count} | elapsed={elapsed:.2f}s\n"
                        f"--- PROMPT ---\n{prompt}\n"
                        f"--- RESPONSE ---\n{response.text}\n"
                        f"--- END ---")

            return response.text
        except Exception as e:
            logger.error(f"GeminiLLM exception: {e}")
            return ""

class HarvardGeminiLLM:
    """Gemini via Harvard HUIT API gateway.

    Ref: https://portal.apis.huit.harvard.edu/docs/ais-gemini-llm/1/overview
    Requires HUIT_API_KEY in .env.
    """

    BASE_URL = "https://go.apis.huit.harvard.edu/ais-google-gemini/v1beta/models"

    def __init__(self) -> None:
        import requests as _requests
        self._requests = _requests
        self.model = "gemini-3-flash-preview"
        self.api_key: str = os.getenv("HUIT_API_KEY", "")
        self.last_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_elapsed = 0.0

    def generate(self, prompt: str) -> str:
        import json as _json

        url = f"{self.BASE_URL}/{self.model}:generateContent"
        payload = _json.dumps({
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]},
            ],
            "generationConfig": {
                "thinkingConfig": {
                    "thinkingLevel": "minimal",
                },
            },
        })
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        try:
            start = time.time()
            resp = self._requests.post(url, headers=headers, data=payload)
            data = resp.json()
            elapsed = time.time() - start

            # Extract text from response
            candidates = data.get("candidates", [])
            text = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts)

            # Extract token count
            usage = data.get("usageMetadata", {})
            token_count = usage.get("totalTokenCount", 0)
            input_tokens = usage.get("promptTokenCount", 0)
            output_tokens = usage.get("candidatesTokenCount", 0)

            self.last_tokens = token_count
            self.last_input_tokens = input_tokens
            self.last_output_tokens = output_tokens
            self.last_elapsed = elapsed

            logger.info(f"llm_call | tokens={token_count} | elapsed={elapsed:.2f}s\n"
                        f"--- PROMPT ---\n{prompt}\n"
                        f"--- RESPONSE ---\n{text}\n"
                        f"--- END ---")

            return text
        except Exception as e:
            logger.error(f"HarvardGeminiLLM exception: {e}")
            return ""

class GPTLLM:
    """GPT models via OpenAI Responses API.

    Requires OPENAI_API_KEY in .env.
    reasoning_effort: "none" | "low" | "medium" | "high"
    """

    def __init__(self, model: str = "gpt-5.2", reasoning_effort: str = "none") -> None:
        from openai import OpenAI as _OpenAI
        self._client = _OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=600)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.last_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_elapsed = 0.0

    def generate(self, prompt: str) -> str:
        try:
            start = time.time()
            response = self._client.responses.create(
                model=self.model,
                reasoning={"effort": self.reasoning_effort},
                input=[{"role": "user", "content": prompt}],
            )
            elapsed = time.time() - start

            token_count = response.usage.total_tokens if response.usage else 0
            input_tokens = response.usage.input_tokens if response.usage else 0
            output_tokens = response.usage.output_tokens if response.usage else 0
            text = response.output_text

            self.last_tokens = token_count
            self.last_input_tokens = input_tokens
            self.last_output_tokens = output_tokens
            self.last_elapsed = elapsed

            logger.info(
                f"llm_call | tokens={token_count} | elapsed={elapsed:.2f}s\n"
                f"--- PROMPT ---\n{prompt}\n"
                f"--- RESPONSE ---\n{text}\n"
                f"--- END ---"
            )
            return text
        except Exception as e:
            logger.error(f"GPTLLM exception: {e}")
            return ""


class QwenLLM:
    """Qwen3 models via DashScope OpenAI-compatible interface.

    Requires DASHSCOPE_API_KEY in .env.
    thinking: True to enable extended thinking, False to disable.
    """

    BASE_URL = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"

    def __init__(self, model: str = "qwen3-235b-a22b", thinking: bool = False) -> None:
        from openai import OpenAI as _OpenAI
        self._client = _OpenAI(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url=self.BASE_URL,
        )
        self.model = model
        self.thinking = thinking
        self.last_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_elapsed = 0.0

    def generate(self, prompt: str) -> str:
        try:
            start = time.time()
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"enable_thinking": self.thinking},
            )
            elapsed = time.time() - start

            token_count = completion.usage.total_tokens if completion.usage else 0
            input_tokens = completion.usage.prompt_tokens if completion.usage else 0
            output_tokens = completion.usage.completion_tokens if completion.usage else 0
            text = completion.choices[0].message.content or ""

            self.last_tokens = token_count
            self.last_input_tokens = input_tokens
            self.last_output_tokens = output_tokens
            self.last_elapsed = elapsed

            logger.info(
                f"llm_call | tokens={token_count} | elapsed={elapsed:.2f}s\n"
                f"--- PROMPT ---\n{prompt}\n"
                f"--- RESPONSE ---\n{text}\n"
                f"--- END ---"
            )
            return text
        except Exception as e:
            logger.error(f"QwenLLM exception: {e}")
            return ""


class HarvardGPTLLM:
    """GPT via Harvard HUIT API gateway.

    Ref: https://portal.apis.huit.harvard.edu/docs/ais-openai-direct/1/overview
    Requires HUIT_API_KEY in .env.
    """

    BASE_URL = "https://go.apis.huit.harvard.edu/ais-openai-direct/v1/responses"

    def __init__(self, reasoning_effort: str = "high") -> None:
        import requests as _requests
        self._requests = _requests
        self.model = "gpt-5.2"
        self.reasoning_effort = reasoning_effort
        self.api_key: str = os.getenv("HUIT_API_KEY", "")
        self.last_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_elapsed = 0.0

    def generate(self, prompt: str) -> str:
        import json as _json

        payload = _json.dumps({
            "model": self.model,
            "reasoning": {"effort": self.reasoning_effort},
            "input": [
                {"role": "user", "content": prompt},
            ],
        })
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        try:
            start = time.time()
            resp = self._requests.post(self.BASE_URL, headers=headers, data=payload, timeout=600)
            data = resp.json()
            elapsed = time.time() - start

            # Extract text from response
            text = ""
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            text += content.get("text", "")

            # Extract token count
            usage = data.get("usage", {})
            token_count = usage.get("total_tokens", 0)
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            self.last_tokens = token_count
            self.last_input_tokens = input_tokens
            self.last_output_tokens = output_tokens
            self.last_elapsed = elapsed

            if token_count == 0:
                logger.warning(f"HarvardGPTLLM zero tokens | status={resp.status_code} | response={data}")

            logger.info(f"llm_call | tokens={token_count} | elapsed={elapsed:.2f}s\n"
                        f"--- PROMPT ---\n{prompt}\n"
                        f"--- RESPONSE ---\n{text}\n"
                        f"--- END ---")

            return text
        except Exception as e:
            logger.error(f"HarvardGPTLLM exception: {e}")
            return ""