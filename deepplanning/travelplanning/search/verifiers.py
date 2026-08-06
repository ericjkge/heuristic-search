"""Semantic verifiers for DeepPlanning travel planning.

A Verifier is a statement describing ONE tool call the agent should make,
scored max_j cos(embed(statement), embed(rendered_tool_call_j)) over the
trajectory's tool calls (MiniLM). Node value = mean over the set.

The set grows during search (add-only): seeds come from the query text;
evolution adds verifiers referencing entities discovered in tool results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Verifier:
    name: str
    description: str
    statement: str  # rendered-tool-call-style phrase, embedded for scoring
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
    def __init__(self, verifiers: list[Verifier] | None = None, embedder: Any = None):
        self.verifiers: list[Verifier] = []
        self.embedder = embedder
        for v in verifiers or []:
            self.add(v)

    @property
    def active(self) -> list[Verifier]:
        return [v for v in self.verifiers if v.active]

    def add(self, v: Verifier) -> None:
        if any(x.name == v.name for x in self.verifiers):
            return
        self.verifiers.append(v)

    def score_node(self, eval_vars: dict[str, Any]) -> dict[str, float]:
        return {v.name: v.score(eval_vars, self.embedder) for v in self.active}

    def aggregate(self, scores: dict[str, float]) -> float:
        vals = [scores[v.name] for v in self.active if v.name in scores]
        return sum(vals) / len(vals) if vals else 0.0

    def all_satisfied(self, scores: dict[str, float], threshold: float) -> bool:
        return all(scores.get(v.name, 0.0) >= threshold for v in self.active)

    def to_json(self) -> list[dict[str, Any]]:
        return [v.to_json() for v in self.verifiers]


RULES = """\
The agent plans travel by calling these database tools (? = optional argument):
  query_train_info(origin, destination, depDate, seatClassName?)
  query_flight_info(origin, destination, depDate, seatClassName?)
  query_hotel_info(destination, checkinDate, checkoutDate, hotelStar?, hotelBrands?)
  query_attraction_details(attraction_name)
  recommend_attractions(city, attraction_type?)
  search_location(place_name)
  query_road_route_info(origin, destination)
  recommend_restaurants(latitude, longitude)
  query_restaurant_details(restaurant_name)

Each verifier checks whether a specific tool call was made. Scoring is fully automatic: your `statement` is
embedded and compared (cosine similarity) against every tool call the agent
has executed, rendered as the tool name followed by its argument values:
  "query_train_info Hefei Nanjing 2025-11-12"
  "query_hotel_info Nanjing 2025-11-12 2025-11-13 3"
  "recommend_restaurants 32.041002 118.784478"
The verifier takes its best match. The search uses these scores to decide
which partial trajectories to extend, so each verifier acts as pressure
toward making that tool call.

Therefore a verifier is ONLY meaningful if it describes a concrete tool call.
Properties of the final plan — total cost within budget, schedule timing,
itinerary structure — are NOT observable in tool calls; never write verifiers
for them. Write every statement in the rendered-call format: tool name plus
argument values, using exact entity names and dates.
GOOD statement: "query_train_info Hefei Nanjing 2025-11-12"
GOOD statement: "query_hotel_info Nanjing 3 star"
BAD statement:  "total cost within 3000 yuan budget"  (plan property, no such call)
BAD statement:  "the agent should search for hotels"  (meta-language, not a call)"""


def seed_prompt(query: str) -> str:
    return f"""\
You are writing verifiers to guide an LLM travel-planning agent's search.

# INITIAL QUERY
{query}

# VERIFIER RULES
{RULES}

# TASK
Write 8-12 verifiers covering the distinct tool calls this task requires —
what must be looked up for the itinerary to be correct and complete. Use the
task's exact entities, dates, and filters.

# OUTPUT FORMAT
Return JSON: {{"verifiers": [{{"name": "snake_case_id", "description": "...",
"statement": "..."}}]}}"""


def evolve_prompt(query: str, vset: VerifierSet, frontier_desc: str) -> str:
    existing = "\n".join(f'- {v.name}: "{v.statement}"' for v in vset.active)
    return f"""\
You are adding verifiers to guide an LLM travel-planning agent's search,
using what the search has discovered so far.

# INITIAL QUERY
{query}

# VERIFIER RULES
{RULES}

# CURRENT VERIFIERS
{existing}

# CURRENT SEARCH FRONTIER (full tool call trajectory of the top 2 highest-scoring nodes)
{frontier_desc}

# TASK
Propose 1-4 NEW verifiers for tool calls that are still missing but now
identifiable — especially calls on entities DISCOVERED in the results above
(e.g. detail lookups on returned hotels/restaurants/attractions, route
queries between found coordinates, comparison lookups across returned
candidates). Do not duplicate existing verifiers. If nothing useful is
missing, return an empty list.

# OUTPUT FORMAT
Return JSON: {{"verifiers": [{{"name": "snake_case_id", "description": "...",
"statement": "..."}}]}}"""
