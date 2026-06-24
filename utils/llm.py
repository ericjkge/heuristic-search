"""OpenRouter (Qwen3-8B) wrapper with full JSONL call logging."""

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "qwen/qwen3-8b"
BASE_URL = "https://openrouter.ai/api/v1"
RUNS_DIR = Path("runs")
MAX_CONCURRENCY = 100  # global cap on in-flight API calls (across all fan-out)


class LLM:
    """Wraps an OpenRouter client; every call() appends a full trace row to trace.jsonl.

    Qwen3 thinking is controlled via OpenRouter's `reasoning={"enabled": ...}`
    (on by default); reasoning tokens come back in a separate `reasoning` field.
    """

    def __init__(self, run_dir: Path | None = None, model: str = MODEL,
                 max_concurrency: int = MAX_CONCURRENCY):
        self.client = OpenAI(
            base_url=BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"]
        )
        self.model = model
        self.run_dir = run_dir or (RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.run_dir / "trace.jsonl"
        self._lock = threading.Lock()  # guards the trace append
        self._sem = threading.BoundedSemaphore(max_concurrency)  # caps in-flight calls

    def call(
        self,
        messages: list[dict[str, str]],
        *,
        tags: dict[str, Any],
        temperature: float = 0.6,  # Qwen3 thinking-mode recommended default
        max_tokens: int | None = None,
        think: bool = True,
        max_retries: int = 4,
    ) -> str:
        """Send a chat completion, log it, and return the response text.

        `tags` carries experiment metadata (puzzle_id, condition, phase, iter,
        parent_id, ...) and is written verbatim into the trace row. With
        `think=True` the answer is in `content`; reasoning is logged separately.
        """
        extra_body = {"reasoning": {"enabled": think}}
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "extra_body": extra_body,
        }
        if max_tokens is not None:  # unlimited if omitted
            params["max_tokens"] = max_tokens

        t0 = time.time()
        message = usage = None
        with self._sem:  # bound total concurrent API calls regardless of fan-out
            for attempt in range(max_retries):
                try:
                    resp = self.client.chat.completions.create(**params)
                    # A malformed 200 (no choices) parses to choices=None; treat it
                    # as retryable here so it doesn't crash outside the loop.
                    if not getattr(resp, "choices", None):
                        raise ValueError("response had no choices")
                    message = resp.choices[0].message
                    usage = resp.usage
                    break
                except Exception as e:  # transient upstream 429s / 5xx / empty body
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)
        latency = time.time() - t0

        text = message.content or ""
        reasoning = getattr(message, "reasoning", None)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0  # includes reasoning tokens

        row = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "model": self.model,
            "tags": tags,
            "messages": messages,
            "response": text,
            "reasoning": reasoning,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "latency_s": round(latency, 3),
        }
        with self._lock:  # serialize concurrent trace appends
            with self.trace_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

        return text


# Smoke test LLM call
if __name__ == "__main__":
    llm = LLM()
    out = llm.call(
        [{"role": "user", "content": "Reply with exactly: hello world"}],
        tags={"phase": "smoke"},
    )
    print("response:", repr(out))
    print("trace:", llm.trace_path)
