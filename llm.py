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
TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT", "180"))
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
            timeout=TIMEOUT_S,
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
        # chat completions report prompt/completion tokens; responses report input/output
        pt = (getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0)) or 0
        ct = (getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0)) or 0
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
    reasoning_effort: str | None = None,
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
        # strip Responses-API bookkeeping keys (e.g. _reasoning) before a chat call
        messages = [{k: v for k, v in m.items() if not k.startswith("_")}
                    for m in prompt]

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
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
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


def _to_response_items(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Chat-style messages -> (instructions, Responses input items). Assistant
    messages may carry '_reasoning' (opaque reasoning items from a previous
    Responses turn) and 'tool_calls'; tool messages become function_call_output."""
    instructions = None
    items: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            instructions = m.get("content") or ""
        elif role == "tool":
            items.append({"type": "function_call_output",
                          "call_id": m.get("tool_call_id", ""),
                          "output": m.get("content") or ""})
        elif role == "assistant":
            items.extend(m.get("_reasoning") or [])
            if m.get("content"):
                items.append({"role": "assistant", "content": m["content"]})
            for tc in m.get("tool_calls") or []:
                items.append({"type": "function_call", "call_id": tc["id"],
                              "name": tc["function"]["name"],
                              "arguments": tc["function"]["arguments"]})
        else:
            items.append({"role": role or "user", "content": m.get("content") or ""})
    return instructions, items


def _flatten_tools(tools: list[dict] | None) -> list[dict] | None:
    """Chat-style {'type':'function','function':{...}} -> internally tagged."""
    if tools is None:
        return None
    return [
        {"type": "function", **t["function"]} if "function" in t else t
        for t in tools
    ]


async def generate_agentic(
    messages: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    tools: list[dict] | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """One agent turn via the Responses API, with reasoning preserved across
    tool calls (store=false + encrypted reasoning items carried in the returned
    message's '_reasoning' key — pass the message back in `messages` on the
    next turn and it round-trips automatically).

    Returns a chat-style assistant message dict: {role, content, tool_calls?,
    _reasoning?}. Note: reasoning models reject temperature; none is sent.
    """
    instructions, items = _to_response_items(messages)
    kwargs: dict[str, Any] = dict(
        model=model, input=items, store=False,
        include=["reasoning.encrypted_content"],
    )
    if instructions:
        kwargs["instructions"] = instructions
    if tools is not None:
        kwargs["tools"] = _flatten_tools(tools)
    if reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    if max_tokens is not None:
        kwargs["max_output_tokens"] = max_tokens

    client = _get_client(provider)
    sem = _get_semaphore()
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await client.responses.create(**kwargs)
            if resp.usage is not None:
                USAGE.add(f"{provider}:{model}", resp.usage)
            msg: dict[str, Any] = {"role": "assistant", "content": "", "_reasoning": []}
            tool_calls = []
            for item in resp.output:
                if item.type == "reasoning":
                    msg["_reasoning"].append(item.model_dump(exclude_none=True))
                elif item.type == "function_call":
                    tool_calls.append({"id": item.call_id, "type": "function",
                                       "function": {"name": item.name,
                                                    "arguments": item.arguments}})
                elif item.type == "message":
                    msg["content"] += "".join(
                        c.text for c in item.content if getattr(c, "text", None))
            if tool_calls:
                msg["tool_calls"] = tool_calls
            if not msg["_reasoning"]:
                del msg["_reasoning"]
            return msg
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            last_err = e
        except APIStatusError as e:
            if "invalid_encrypted_content" in str(e):
                # gateway routed to a different upstream key; reasoning items
                # from the prior turn can't be decrypted — drop them and go on
                logger.warning("dropping stale encrypted reasoning items")
                kwargs["input"] = [i for i in kwargs["input"]
                                   if i.get("type") != "reasoning"]
                last_err = e
                continue
            if e.status_code < 500:
                raise
            last_err = e
        delay = min(60.0, 2.0**attempt) * (1 + random.random())
        logger.warning("retry %d/%d after %s (sleeping %.1fs)", attempt + 1, MAX_RETRIES,
                       type(last_err).__name__, delay)
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
    reasoning_effort: str | None = None,
    max_parse_retries: int = 2,
) -> Any:
    """generate() with JSON mode, parsed. Re-samples on unparseable output."""
    for _ in range(max_parse_retries + 1):
        text = await generate(
            prompt, model=model, provider=provider, system=system,
            temperature=temperature, max_tokens=max_tokens, json_mode=True,
            reasoning_effort=reasoning_effort,
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
