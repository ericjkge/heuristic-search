"""Async LLM client with global concurrency control, retries, and usage tracking.

Routes to one of two OpenAI-compatible providers:
    openrouter  https://openrouter.ai/api/v1        (OPENROUTER_API_KEY)
    abaka       https://app-us.ppapi.ai/v1          (ABAKA_API_KEY, lab endpoint)

Usage:
    from llm import generate, generate_many

    text = await generate("What is 2+2?")
    texts = await generate_many(["q1", ...], model="gemini-3.1-flash-lite-preview",
                                provider="abaka")

Configuration comes from .env / environment:
    OPENROUTER_API_KEY / ABAKA_API_KEY   per-provider keys
    LLM_PROVIDER         default provider (default: openrouter)
    LLM_MODEL            default model (default: openai/gpt-4.1-mini)
    LLM_MAX_CONCURRENCY  max in-flight requests across the process (default: 16)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from tqdm.asyncio import tqdm_asyncio

load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger("llm")

PROVIDERS = {
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    "abaka": {"base_url": "https://app-us.ppapi.ai/v1", "key_env": "ABAKA_API_KEY"},
}
DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "abaka")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "qwen3-8b")
MAX_CONCURRENCY = int(os.environ.get("LLM_MAX_CONCURRENCY", "16"))
MAX_RETRIES = 6

_clients: dict[str, AsyncOpenAI] = {}


def _get_client(provider: str) -> AsyncOpenAI:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; options: {sorted(PROVIDERS)}")
    if provider not in _clients:
        spec = PROVIDERS[provider]
        _clients[provider] = AsyncOpenAI(
            base_url=spec["base_url"],
            api_key=os.environ.get(spec["key_env"]),
            timeout=180.0,
            max_retries=0,  # we handle retries ourselves
        )
    return _clients[provider]

# One semaphore shared by every call in the process, created lazily so it binds
# to the running event loop.
_semaphore: asyncio.Semaphore | None = None
_semaphore_loop: asyncio.AbstractEventLoop | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore, _semaphore_loop
    loop = asyncio.get_running_loop()
    if _semaphore is None or _semaphore_loop is not loop:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        _semaphore_loop = loop
    return _semaphore


@dataclass
class Usage:
    """Cumulative token/cost accounting across all calls in the process."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, model: str, usage: Any) -> None:
        self.requests += 1
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        cost = getattr(usage, "cost", 0.0) or 0.0
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.cost_usd += cost
        m = self.by_model.setdefault(
            model, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
        )
        m["requests"] += 1
        m["prompt_tokens"] += pt
        m["completion_tokens"] += ct
        m["cost_usd"] += cost

    def summary(self) -> str:
        return (
            f"{self.requests} requests | {self.prompt_tokens:,} prompt tok | "
            f"{self.completion_tokens:,} completion tok | ${self.cost_usd:.4f}"
        )


USAGE = Usage()


async def generate(
    prompt: str | list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    system: str | None = None,
    temperature: float = 1.0,
    max_tokens: int | None = None,
    json_mode: bool = False,
    tools: list[dict] | None = None,
    tool_choice: Any = None,
    return_response: bool = False,
) -> Any:
    """Single chat completion. `prompt` is a user string or a full messages list.

    Returns the completion text, or the raw response object if return_response=True.
    Retries on rate limits, timeouts, and 5xx with exponential backoff + jitter.
    """
    if isinstance(prompt, str):
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
    else:
        assert system is None, "pass system inside the messages list"
        messages = prompt

    kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    extra_body: dict[str, Any] = {}
    if provider == "openrouter":
        extra_body["usage"] = {"include": True}  # report cost in usage
    if model.startswith("qwen3"):
        extra_body["enable_thinking"] = False  # disable qwen3 native thinking
    if extra_body:
        kwargs["extra_body"] = extra_body
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    client = _get_client(provider)
    sem = _get_semaphore()
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await client.chat.completions.create(**kwargs)
            if resp.usage is not None:
                USAGE.add(f"{provider}:{model}", resp.usage)
            if return_response:
                return resp
            return resp.choices[0].message.content
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            last_err = e
        except APIStatusError as e:
            if e.status_code < 500:
                raise
            last_err = e
        delay = min(60.0, 2.0**attempt) * (1 + random.random())
        logger.warning("retry %d/%d after %s (sleeping %.1fs)", attempt + 1, MAX_RETRIES, type(last_err).__name__, delay)
        await asyncio.sleep(delay)
    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} retries") from last_err


async def generate_json(
    prompt: str | list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    system: str | None = None,
    temperature: float = 1.0,
    max_tokens: int | None = None,
    max_parse_retries: int = 2,
) -> Any:
    """generate() with JSON mode, parsed. Re-samples on unparseable output."""
    for _ in range(max_parse_retries + 1):
        text = await generate(
            prompt, model=model, provider=provider, system=system,
            temperature=temperature, max_tokens=max_tokens, json_mode=True,
        )
        try:
            return json.loads(_strip_code_fence(text))
        except (json.JSONDecodeError, TypeError):
            logger.warning("JSON parse failed, resampling")
    raise ValueError(f"unparseable JSON after {max_parse_retries + 1} attempts: {text[:500]!r}")


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -3]
    return text.strip()


async def generate_many(
    prompts: list[str | list[dict[str, str]]],
    *,
    progress: bool = True,
    desc: str = "llm",
    return_exceptions: bool = False,
    **kwargs: Any,
) -> list[Any]:
    """Run many generate() calls concurrently (bounded by the global semaphore)."""
    tasks = [generate(p, **kwargs) for p in prompts]
    if progress:
        return await tqdm_asyncio.gather(*tasks, desc=desc, return_exceptions=return_exceptions)
    return await asyncio.gather(*tasks, return_exceptions=return_exceptions)


if __name__ == "__main__":
    async def _smoke_test() -> None:
        out = await generate("Reply with exactly the word: ok", temperature=0.0, max_tokens=10)
        print(f"provider={DEFAULT_PROVIDER!r} model={DEFAULT_MODEL!r} response={out!r}")
        outs = await generate_many(
            [f"Reply with exactly the number {i}" for i in range(4)],
            temperature=0.0, max_tokens=10, desc="smoke",
        )
        print("batch:", outs)
        print("usage:", USAGE.summary())

    asyncio.run(_smoke_test())
