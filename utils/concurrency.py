"""Run independent thunks concurrently, returning results in submission order.

The real concurrency ceiling is the per-call semaphore in utils.llm, so these
pools are sized to the work itself; threads that exceed the cap simply block on
that semaphore. Exceptions propagate (one exhausted-retry call fails the batch).
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")


def run_parallel(thunks: list[Callable[[], T]]) -> list[T]:
    """Execute each zero-arg thunk concurrently; results stay in input order."""
    if not thunks:
        return []
    with ThreadPoolExecutor(max_workers=len(thunks)) as ex:
        return list(ex.map(lambda f: f(), thunks))
