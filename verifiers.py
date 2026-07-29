"""LLM-written, LLM-free-to-evaluate semantic verifiers.

A Verifier is a natural-language statement (phrased like a search query)
describing information the solver should have sought. Its score for a node is
    max_j cos(embed(statement), embed(search_j))
over the search queries issued along the node's trajectory, clamped to [0, 1]
(MiniLM embeddings, see embeddings.py).

All verifiers are desired properties (positive only); a node's value is the
mean score over active verifiers. Zero-variance verifiers across the node
population get deactivated (kept inactive, not deleted).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any


@dataclass
class Verifier:
    name: str
    description: str
    statement: str  # search-query-like phrase, embedded for scoring
    kind: str = "seed"  # seed | evolved
    active: bool = True

    def score(self, eval_vars: dict[str, Any], embedder: Any) -> float:
        try:
            return embedder.max_sim(self.statement, eval_vars.get("searches", []))
        except Exception:
            return 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "statement": self.statement, "kind": self.kind, "active": self.active,
        }


class VerifierSet:
    def __init__(
        self, verifiers: list[Verifier] | None = None, embedder: Any = None
    ):
        self.verifiers: list[Verifier] = []
        self.embedder = embedder
        self.log: list[dict[str, Any]] = []  # add/drop events for analysis
        for v in verifiers or []:
            self.add(v)

    @property
    def active(self) -> list[Verifier]:
        return [v for v in self.verifiers if v.active]

    def add(self, v: Verifier) -> None:
        # dedupe by name (keep first)
        if any(x.name == v.name for x in self.verifiers):
            return
        self.verifiers.append(v)
        self.log.append({"event": "add", "verifier": v.to_json()})

    def score_node(self, eval_vars: dict[str, Any]) -> dict[str, float]:
        return {v.name: v.score(eval_vars, self.embedder) for v in self.active}

    def aggregate(self, scores: dict[str, float]) -> float:
        """Mean score over active verifiers."""
        vals = [scores[v.name] for v in self.active if v.name in scores]
        return sum(vals) / len(vals) if vals else 0.0

    def all_satisfied(self, scores: dict[str, float], threshold: float) -> bool:
        return all(scores.get(v.name, 0.0) >= threshold for v in self.active)

    def drop_no_variance(
        self, population_scores: list[dict[str, float]], max_active: int = 8
    ) -> dict[str, Any]:
        """Deactivate verifiers with zero std across the population, then cap
        actives at max_active by std desc."""
        stds: dict[str, float] = {}
        for v in self.active:
            vals = [s.get(v.name, 0.0) for s in population_scores]
            stds[v.name] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

        dropped = []
        for v in self.active:
            if stds[v.name] == 0.0:
                v.active = False
                dropped.append(v.name)

        survivors = sorted(self.active, key=lambda v: stds[v.name], reverse=True)
        for v in survivors[max_active:]:
            v.active = False
            dropped.append(v.name)

        event = {"event": "drop", "dropped": dropped, "stds": stds}
        self.log.append(event)
        return event

    def to_json(self) -> list[dict[str, Any]]:
        return [v.to_json() for v in self.verifiers]


# Shared documentation injected into every verifier-generation prompt.
VERIFIER_RULES = """\
Each verifier is a `statement`: a short search-query-like phrase describing
information the solver should have sought. It is scored automatically as the
maximum cosine similarity (sentence embeddings) between the statement and each
<search> query the solver has issued so far — so it costs nothing to evaluate,
but it can only measure what the solver SEARCHED for, not what it read or
answered. Phrase every statement like a search query itself, not a sentence
about the solver; meta-language dilutes the embedding.
The solver searches with BM25 (keyword overlap), so its queries are short
keyword phrases — distinctive names plus a relation word, not full questions.
Phrase statements in that same keyword style so they embed close to the
solver's queries.

Decomposing a multi-hop question — example: "When was the baseball team
winning the World Series in 2015 created?" Answering requires finding the 2015
World Series winner first, then searching THAT team's founding date. Write one
verifier per hop:
  1. "2015 World Series winner"
     Hop 1 is directly searchable — name its entities and relation exactly.
  2. "baseball team founded"
     Hop 2 depends on hop 1's answer, which is unknown when verifiers are
     written. Never guess the entity. Write the statement at the META level:
     the relation plus the entity's known TYPE ("baseball team"), so it still
     matches the eventual entity-specific query (e.g. "Kansas City Royals
     founded"). Keep meta statements short and relation-focused — padding
     them with descriptors of the unknown entity ("...the World Series
     winning team") weakens the match with the real future query.

GOOD statement: "Inception film director"
GOOD statement: "baseball team founded"                  (later-hop, meta level)
BAD statement:  "the agent should search for the director" (meta-language)
BAD statement:  "relevant information about the question" (matches anything)"""
