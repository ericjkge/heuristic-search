"""Abaka (ppapi.ai) OpenAI-compatible client with full JSONL call logging.

Every call() appends a trace row to <run_dir>/trace.jsonl (same schema as
utils/llm.py, so both viewers can read it). An API error that survives all
retries RAISES -- no silent failure.
"""

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

MODEL = "gpt-5.4-mini"
REASONING_EFFORT = None  # None -> API default (medium); plumbing kept for overrides
BASE_URL = "https://app-us.ppapi.ai/v1"
RUNS_DIR = Path("runs")
MAX_CONCURRENCY = 100  # global cap on in-flight API calls


class LLM:
    def __init__(self, run_dir: Path | None = None, model: str = MODEL,
                 reasoning_effort: str | None = REASONING_EFFORT,
                 max_concurrency: int = MAX_CONCURRENCY):
        self.client = OpenAI(base_url=BASE_URL, api_key=os.environ["ABAKA_API_KEY"])
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.run_dir = run_dir or (RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.run_dir / "trace.jsonl"
        self._lock = threading.Lock()
        self._sem = threading.BoundedSemaphore(max_concurrency)

    def call(
        self,
        messages: list[dict[str, str]],
        *,
        tags: dict[str, Any],  # experiment metadata, written verbatim to the trace
        temperature: float | None = None,  # omitted by default (reasoning models)
        max_tokens: int | None = None,
        max_retries: int = 4,
    ) -> str:
        params: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        t0 = time.time()
        message = usage = None
        with self._sem:
            for attempt in range(max_retries):
                try:
                    resp = self.client.chat.completions.create(**params)
                    if not getattr(resp, "choices", None):
                        raise ValueError("response had no choices")
                    message = resp.choices[0].message
                    usage = resp.usage
                    break
                except Exception:  # transient 429s / 5xx / empty body
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)
        latency = time.time() - t0

        row = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "model": self.model,
            "tags": tags,
            "messages": messages,
            "response": message.content or "",
            "reasoning": getattr(message, "reasoning", None),
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "latency_s": round(latency, 3),
        }
        with self._lock:
            with self.trace_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

        return row["response"]


if __name__ == "__main__":
    llm = LLM()
    out = llm.call([{"role": "user", "content": "Reply with exactly: hello world"}],
                   tags={"phase": "smoke"})
    print("response:", repr(out))
    print("trace:", llm.trace_path)
