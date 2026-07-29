"""MiniLM sentence embeddings for semantic verifier scoring (LLM-free).

Semantic verifiers are natural-language statements scored as
    max_j cos(embed(statement), embed(query_j))
over the search queries issued along a trajectory, clamped to [0, 1].
Embeddings are cached by exact text, so repeated scoring is nearly free.
"""

from __future__ import annotations

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            embs = self._load().encode(
                missing, normalize_embeddings=True, show_progress_bar=False
            )
            for t, e in zip(missing, embs):
                self._cache[t] = e
        return np.stack([self._cache[t] for t in texts])

    def max_sim(self, statement: str, queries: list[str]) -> float:
        """max over queries of cosine(statement, query), clamped to [0, 1]."""
        queries = [q for q in queries if q]
        if not statement or not queries:
            return 0.0
        embs = self.encode([statement] + queries)
        return float(min(1.0, max(0.0, (embs[1:] @ embs[0]).max())))


_default: Embedder | None = None


def get_embedder() -> Embedder:
    global _default
    if _default is None:
        _default = Embedder()
    return _default
