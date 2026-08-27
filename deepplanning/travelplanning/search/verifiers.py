"""Rubric verifiers for DeepPlanning travel planning.

A Verifier is one rubric item: a statement about the final plan phrased so
that 1 = fully satisfied / optimal and 0 = poor. An LLM judge scores every
active item in one call, given the user request and the agent's full
conversation (tool calls, results, plan). Node value = mean over the set.

The set grows during search (add-only): seeds come from the request + the
official plan requirements; evolution adds items that separate the current
best plans from the worst one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prompts import get_system_prompt


@dataclass
class Verifier:
    name: str
    description: str
    statement: str  # rubric item, 1 = optimal, 0 = poor
    kind: str = "seed"  # seed | evolved
    active: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "statement": self.statement, "kind": self.kind, "active": self.active,
        }


class VerifierSet:
    def __init__(self, verifiers: list[Verifier] | None = None):
        self.verifiers: list[Verifier] = []
        for v in verifiers or []:
            self.add(v)

    @property
    def active(self) -> list[Verifier]:
        return [v for v in self.verifiers if v.active]

    def add(self, v: Verifier) -> bool:
        if any(x.name == v.name for x in self.verifiers):
            return False
        self.verifiers.append(v)
        return True

    def aggregate(self, scores: dict[str, float]) -> float:
        vals = [scores[v.name] for v in self.active if v.name in scores]
        return sum(vals) / len(vals) if vals else 0.0

    def to_json(self) -> list[dict[str, Any]]:
        return [v.to_json() for v in self.verifiers]


def plan_requirements() -> str:
    """Section II rule body of the official agent system prompt, verbatim."""
    sp = get_system_prompt("en")
    start = sp.index("**A. Content & Logic Rigor**")
    end = sp.index("COMPLETE EXAMPLE")
    return sp[start:end].rstrip().rstrip("=").rstrip()


def parse_verifiers(data: Any, kind: str) -> list[Verifier]:
    items = data.get("verifiers", []) if isinstance(data, dict) else []
    return [
        Verifier(name=str(i["name"]), description=str(i.get("description", "")),
                 statement=str(i["statement"]), kind=kind)
        for i in items
        if isinstance(i, dict) and i.get("name") and i.get("statement")
    ]


RULES = """\
Each verifier is a rubric item about the FINAL PLAN. An LLM judge reads the user request, the agent's full conversation (every tool call, every tool result, the plan), and scores each item in [0, 1]. The search uses the mean score to decide which plans to revise and shows the per-item scores to the agent as feedback.

Phrase every statement so that 1 means the plan fully satisfies it and 0 means it clearly does not. Items may target things the plan must do OR mistakes it must avoid, but always word them in the direction where 1 is optimal.
GOOD statement: "Every hotel stay uses a hotel returned by query_hotel_info with star rating 4 or higher"
GOOD statement: "The itinerary's total cost, summed from the plan's own cost lines, does not exceed the 3000 CNY budget"
GOOD statement: "The chosen hotel has the highest rating among all hotels returned by the agent's query_hotel_info calls"
GOOD statement: "No activity starts before the previous activity of the same day ends, including travel time between locations"
BAD statement:  "The plan is good"  (not checkable, not specific)
BAD statement:  "Does the plan exceed the budget?"  (a question, and 1 would mean poor)
BAD statement:  "The agent should search for hotels"  (about the process in the abstract, not a checkable property of the final plan and its evidence)

A few other tips:
- Items should come from the user request and the plan requirements, not from any candidate plan. Never require a specific restaurant, hotel, train, or route that a candidate plan happened to choose unless the user named it.
- When a request constraint has an ambiguous boundary, the verifier should accept any reasonable reading, not just the strictest one (e.g. "at least a 4-star hotel" is satisfied by a 4-star hotel, and "opened after 2010" is satisfied by a hotel whose listed year is 2010)
- One rule per item. Each item checks exactly one thing, so that a single violation anywhere in the plan makes it clearly fail. Do not bundle several checks into one statement.
- The plan requirements above define what a valid plan looks like (daily structure, meals, sightseeing, timing, hours, pricing, budget); make sure the items cover them, not only the user's explicit requests.
- A note is not a fix. The plan should contain only the itinerary and budget; an item that a plan could satisfy by adding an explanation, disclaimer, or caveat is invalid."""


def seed_prompt(query: str, n_verifiers: str = "8-12") -> str:
    return f"""\
You are writing rubric verifiers to guide an LLM travel-planning agent's search.

# USER REQUEST
{query}

# PLAN REQUIREMENTS (verbatim from the agent's instructions)
{plan_requirements()}

# VERIFIER RULES
{RULES}

# TASK
Write {n_verifiers} verifiers that define what a correct and complete plan for THIS request must satisfy. Use the request's specific entities, dates, and numbers. Ensure each verifier is distinct and avoid any redundant or overlapping criteria.

# OUTPUT FORMAT
Return JSON: {{"verifiers": [{{"name": "snake_case_id", "description": "...", "statement": "..."}}]}}"""


def evolve_prompt(query: str, vset: VerifierSet, frontier_desc: str,
                  n_verifiers: str = "1-3") -> str:
    existing = "\n".join(f'- {v.name}: "{v.statement}"' for v in vset.active)
    return f"""\
You are adding rubric verifiers to guide an LLM travel-planning agent's search, using what the search has produced so far.

# USER REQUEST
{query}

# PLAN REQUIREMENTS (verbatim from the agent's instructions)
{plan_requirements()}

# VERIFIER RULES
{RULES}

# CURRENT VERIFIERS
{existing}

# CURRENT SEARCH FRONTIER
The two highest-scoring plans and the lowest-scoring plan, each with the agent's full conversation and its current per-verifier scores.

{frontier_desc}

# TASK
Propose up to {n_verifiers} NEW verifiers that would separate the better plans from the worse one, or that catch a real problem the current verifiers miss. Describe what any correct plan must satisfy; do not write items around the particular choices the plans above made. Do not duplicate or rephrase existing verifiers. If the current verifiers already cover everything that matters, return an empty list.

# OUTPUT FORMAT
Return JSON: {{"verifiers": [{{"name": "snake_case_id", "description": "...", "statement": "..."}}]}}"""


def judge_prompt(query: str, conversation: str, verifiers: list[Verifier]) -> str:
    items = "\n".join(f'- {v.name}: "{v.statement}"' for v in verifiers)
    return f"""\
You are scoring a travel plan produced by an LLM agent against a rubric.

# USER REQUEST
{query}

# AGENT CONVERSATION (score ONLY the final plan; use the rest as evidence to check facts, prices, ratings, and availability)
{conversation}

# RUBRIC (1 means full satisfied)
{items}

# TASK
For every rubric item, give a score in [0, 1] continous (e.g. 0.3 if partially satisfied) and a one-sentence reason citing the evidence from the conversation. Score 1 only if the item holds everywhere in the plan. Do not credit explanations, disclaimers, or caveats in the plan.

# OUTPUT FORMAT
Return JSON: {{"scores": {{"<name>": {{"score": 0.0, "reason": "..."}}, ...}}}} with an entry for every rubric item name above."""


def parse_judge(data: Any, verifiers: list[Verifier]) -> tuple[dict[str, float], dict[str, str]]:
    """-> (scores clamped to [0,1], reasons); missing items score 0."""
    raw = data.get("scores", {}) if isinstance(data, dict) else {}
    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for v in verifiers:
        entry = raw.get(v.name)
        if isinstance(entry, dict):
            try:
                scores[v.name] = max(0.0, min(1.0, float(entry.get("score", 0.0))))
            except (TypeError, ValueError):
                scores[v.name] = 0.0
            reasons[v.name] = str(entry.get("reason", ""))
        elif isinstance(entry, (int, float)):
            scores[v.name] = max(0.0, min(1.0, float(entry)))
            reasons[v.name] = ""
        else:
            scores[v.name] = 0.0
            reasons[v.name] = "(missing from judge output)"
    return scores, reasons
