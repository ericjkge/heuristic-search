"""Self-improving verifier search (SVS) over complete travel plans.

Node = a full agent conversation (system, user, tool-call turns, tool results)
that ends in a <plan>...</plan>. Seeds are n_seed independent episodes; an
expansion appends rubric feedback + a revise instruction to the parent's
conversation and runs another episode (tool turns allowed) that ends in a
revised plan. Every node is scored by one LLM-judge call over its full
conversation against the current rubric; node value = mean item score.
Parents ~ softmax((V - deg_coef*children)/tau). Every evolve_every rounds the
rubric grows (add-only) from the top-2 and lowest-V conversations, and all
nodes are rescored on the new items. Search runs for max_rounds rounds and
returns the best-V plan.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import random
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

SEARCH_DIR = Path(__file__).resolve().parent
TP_DIR = SEARCH_DIR.parent
ROOT = TP_DIR.parent.parent
for p in (str(ROOT), str(TP_DIR), str(TP_DIR / "agent"), str(TP_DIR / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import llm
from prompts import get_system_prompt

from verifiers import (Verifier, VerifierSet, evolve_prompt, judge_prompt,
                       parse_judge, parse_verifiers, seed_prompt)

PLAN_NOW = (
    "You have used all available tool-call turns. Based on the information "
    "gathered so far, produce your final complete travel plan now, strictly "
    "following the required format."
)

GUIDANCE_TAG = "[Automated search guidance — not a user message]"

REVISE = "Revise your plan above: fix each item that scores below 1 and re-check any part you now believe was wrong. You may call the database tools first to look up any information you need. When ready, output the complete revised plan wrapped in <plan></plan> tags."

_PLAN = re.compile(r"<plan>(.*?)</plan>", re.S | re.I)

# tool name -> per-instance database file (from their tools_fn_agent.py)
DB_MAPPING = {
    "query_train_info": "trains/trains.csv",
    "query_flight_info": "flights/flights.csv",
    "query_hotel_info": "hotels/hotels.csv",
    "query_attraction_details": "attractions/attractions.csv",
    "recommend_attractions": "attractions/attractions.csv",
    "search_location": "locations/locations_coords.csv",
    "query_road_route_info": "transportation/distance_matrix.csv",
    "recommend_restaurants": "restaurants/restaurants.csv",
    "query_restaurant_details": "restaurants/restaurants.csv",
}


def load_tools(sample_id: str, language: str = "en") -> tuple[list[dict], dict[str, Any]]:
    """Load OpenAI tool schemas + per-instance tool objects (their loader logic)."""
    schema_path = TP_DIR / "tools" / f"tool_schema_{language}.json"
    raw = json.loads(schema_path.read_text())
    schemas = raw if isinstance(raw, list) else raw.get("tools", [raw])
    openai_tools = [
        s if s.get("type") == "function"
        else {"type": "function", "function": {"name": s.get("name"),
                                               "description": s.get("description", ""),
                                               "parameters": s.get("parameters", {})}}
        for s in schemas
    ]

    import tools  # noqa: F401  (registers BaseTravelTool subclasses)
    base = importlib.import_module("tools.base_travel_tool").BaseTravelTool
    sample_db = TP_DIR / "database" / f"database_{language}" / f"id_{sample_id}"
    instances: dict[str, Any] = {}
    for cls in base.__subclasses__():
        try:
            name = getattr(cls, "name", "")
            cfg: dict[str, Any] = {"language": language}
            if name in DB_MAPPING and (sample_db / DB_MAPPING[name]).exists():
                cfg["database_path"] = str(sample_db / DB_MAPPING[name])
            inst = cls(cfg=cfg)
            inst_name = getattr(inst, "name", None) or name
            if inst_name:
                instances[inst_name] = inst
        except Exception:
            continue
    if not instances:
        raise RuntimeError(f"no tools loaded for sample {sample_id}")
    return openai_tools, instances


def render_tool_call(name: str, arguments: str) -> str:
    """Tool call -> compact one-line text."""
    try:
        vals = " ".join(str(v) for v in json.loads(arguments or "{}").values())
    except Exception:
        vals = str(arguments)
    return f"{name} {vals}".strip()


def message_to_dict(msg: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"role": getattr(msg, "role", "assistant"),
                         "content": getattr(msg, "content", None) or ""}
    tcs = getattr(msg, "tool_calls", None)
    if tcs:
        d["tool_calls"] = [
            {"id": tc.id or f"call_{uuid.uuid4().hex[:24]}", "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in tcs
        ]
    return d


def render_messages(msgs: list[dict], cap: int = 3000) -> str:
    """Human-readable prompt render for logging; long tool results truncated."""
    parts = []
    for m in msgs:
        body = m.get("content") or ""
        if len(body) > cap:
            body = body[:cap] + f" ...[truncated {len(body) - cap} chars]"
        for tc in m.get("tool_calls") or []:
            body += f"\n<tool_call> {tc['function']['name']}({tc['function']['arguments']})"
        parts.append(f"[{m.get('role', '?')}]\n{body}")
    return "\n\n".join(parts)


def render_conversation(msgs: list[dict], include_guidance: bool = True) -> str:
    """Full conversation for the judge / rubric writer: user request, every tool
    call and complete tool result, every assistant text (plans). The system
    prompt is omitted; search-guidance messages optionally."""
    parts = []
    for m in msgs:
        role = m.get("role")
        body = m.get("content") or ""
        if role == "system":
            continue
        if role == "user":
            if body.startswith(GUIDANCE_TAG) and not include_guidance:
                continue
            parts.append(f"[user]\n{body}")
        elif role == "assistant":
            calls = [render_tool_call(tc["function"]["name"], tc["function"]["arguments"])
                     for tc in m.get("tool_calls") or []]
            if calls:
                parts.append("[assistant tool calls]\n" + "\n".join(calls))
            if body.strip():
                parts.append(f"[assistant]\n{body}")
        elif role == "tool":
            parts.append(f"[tool result: {m.get('name', 'tool')}]\n{body}")
    return "\n\n".join(parts)


def extract_plan(text: str) -> str:
    if not text:
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    parts = [m.strip() for m in _PLAN.findall(text) if m.strip()]
    return "\n\n".join(parts)


@dataclass
class Node:
    id: int
    parent: int | None
    depth: int  # number of revisions from a seed
    state: list[dict]  # full message list, ends with the plan turn
    plan: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    value: float = 0.0
    children_count: int = 0
    born_round: int = 0
    n_turns: int = 0  # tool-call turns in the episode that produced this node
    forced: bool = False  # plan was forced after the turn budget
    prompt: str = ""  # rendered messages the episode started from


@dataclass
class SearchConfig:
    n_seed: int = 3
    n_parents: int = 1
    k_children: int = 1
    max_rounds: int = 10
    evolve_every: int = 3        # add verifiers every R rounds; 0 disables
    tau: float = 0.2
    deg_coef: float = 0.3        # V-point price per child in sampling
    feedback: bool = True        # show per-item scores to the agent on revision
    seed_verifiers: str = "8-12"
    evolve_verifiers: str = "1-3"
    max_turns: int = 100         # tool-call turns per episode before a forced plan
    max_tokens: int | None = None  # max_output_tokens per call; None = uncapped
    seed: int = 0


@dataclass
class SearchResult:
    plan: str
    reason: str  # budget | empty
    rounds: int
    n_nodes: int
    best_node_id: int | None
    verifiers: list[dict]


def softmax_sample(rng: random.Random, pool: list[Node], n: int, tau: float,
                   deg_coef: float = 0.3) -> list[Node]:
    pool = list(pool)
    picked: list[Node] = []
    while pool and len(picked) < n:
        logits = [(x.value - deg_coef * x.children_count) / max(tau, 1e-6) for x in pool]
        m = max(logits)
        weights = [math.exp(l - m) for l in logits]
        picked.append(pool.pop(rng.choices(range(len(pool)), weights=weights, k=1)[0]))
    return picked


class TravelSearch:
    def __init__(self, example: dict, model: str, provider: str, cfg: SearchConfig,
                 log: Callable[[dict], None] | None = None, language: str = "en",
                 reasoning_effort: str | None = "none"):
        self.example = example
        self.query: str = example["query"]
        self.sample_id = str(example["id"])
        self.model, self.provider = model, provider
        self.reasoning_effort = reasoning_effort
        self.cfg = cfg
        self.log = log or (lambda ev: None)
        self.openai_tools, self.tool_instances = load_tools(self.sample_id, language)
        self.system_prompt = get_system_prompt(language)
        self.parse_failures = 0
        self.llm_calls = 0
        self.vset = VerifierSet()

    # ------------------------------------------------------------------ state
    def initial_state(self) -> list[dict]:
        return [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.query}]

    def _exec_tool(self, name: str, arguments: str) -> str:
        inst = self.tool_instances.get(name)
        if inst is None:
            return json.dumps({"error": f"tool '{name}' not found"}, ensure_ascii=False)
        try:
            args = json.loads(arguments) if arguments else {}
        except Exception:
            args = {}
        try:
            res = inst.call(args)
            return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    # ---------------------------------------------------------------- episode
    async def episode(self, msgs: list[dict]) -> tuple[str, list[dict], list[dict], int, bool]:
        """Run the agent from `msgs` until it emits a plan (or the turn budget
        forces one). Returns (plan, full messages, tool results, n_turns, forced)."""
        msgs = list(msgs)
        tool_results: list[dict] = []
        n_turns = 0
        for _ in range(self.cfg.max_turns):
            msg = await llm.generate_agentic(
                msgs, model=self.model, provider=self.provider, tools=self.openai_tools,
                reasoning_effort=self.reasoning_effort, max_tokens=self.cfg.max_tokens)
            self.llm_calls += 1
            msgs.append(msg)
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                result = await asyncio.to_thread(
                    self._exec_tool, tc["function"]["name"], tc["function"]["arguments"])
                msgs.append({"role": "tool", "tool_call_id": tc["id"],
                             "name": tc["function"]["name"], "content": result})
                tool_results.append({"name": tc["function"]["name"], "content": result})
            if tool_calls:
                n_turns += 1
                continue
            plan = extract_plan(msg.get("content", ""))
            if plan:
                return plan, msgs, tool_results, n_turns, False
            break  # dead turn: no tools, no plan — force below
        msgs.append({"role": "user", "content": PLAN_NOW})
        msg = await llm.generate_agentic(
            msgs, model=self.model, provider=self.provider,
            reasoning_effort=self.reasoning_effort, max_tokens=self.cfg.max_tokens)
        self.llm_calls += 1
        msgs.append(msg)
        text = msg.get("content", "") or ""
        return extract_plan(text) or text, msgs, tool_results, n_turns, True

    # ---------------------------------------------------------------- scoring
    async def judge(self, node: Node, verifiers: list[Verifier]) -> None:
        """Score `node` on `verifiers` (one LLM call) and merge into node.scores."""
        if not verifiers:
            return
        try:
            data = await llm.generate_json(
                judge_prompt(self.query, render_conversation(node.state, include_guidance=False),
                             verifiers),
                model=self.model, provider=self.provider,
                reasoning_effort=self.reasoning_effort)
            self.llm_calls += 1
            scores, reasons = parse_judge(data, verifiers)
        except Exception as e:
            self.parse_failures += 1
            scores = {v.name: 0.0 for v in verifiers}
            reasons = {v.name: f"(judge error: {e!r})" for v in verifiers}
        node.scores.update(scores)
        node.reasons.update(reasons)
        node.value = self.vset.aggregate(node.scores)

    # -------------------------------------------------------------- expansion
    def guidance_message(self, node: Node) -> dict:
        """Persisted user turn: per-item scores (lowest first) + revise instruction."""
        if self.cfg.feedback and self.vset.active:
            lines = sorted(((node.scores.get(v.name, 0.0), v.name,
                             node.reasons.get(v.name, ""))
                            for v in self.vset.active), key=lambda x: x[0])
            body = "\n".join(f"{name} ({sc:.2f}): {reason}" for sc, name, reason in lines)
            text = (f"{GUIDANCE_TAG}\n"
                    f"Judge feedback on your plan above (V = {node.value:.2f}; 1 = fully satisfied, 0 = not at all), lowest first:\n"
                    + body + "\n\n" + REVISE)
        else:
            text = f"{GUIDANCE_TAG}\n{REVISE}"
        return {"role": "user", "content": text}

    async def expand(self, parent: Node | None, k: int, rnd: int, add_child) -> None:
        """k episodes from `parent` (None = seeds from the initial state)."""
        if parent is None:
            msgs = self.initial_state()
        else:
            msgs = list(parent.state) + [self.guidance_message(parent)]
        prompt_text = render_messages(msgs)

        async def one() -> None:
            try:
                plan, state, tool_results, n_turns, forced = await self.episode(msgs)
            except Exception:
                self.parse_failures += 1
                return
            if not plan:
                self.parse_failures += 1
                return
            await add_child(parent, state, plan, rnd, tool_results, n_turns, forced, prompt_text)

        await asyncio.gather(*(one() for _ in range(k)))

    # ----------------------------------------------------------------- evolve
    def describe_frontier(self, nodes: list[Node]) -> str:
        """Top-2 by V plus the lowest-V node, each with its full conversation
        and per-item scores."""
        ranked = sorted(nodes, key=lambda n: n.value, reverse=True)
        picked = list(zip(["BEST", "SECOND BEST"], ranked[:2]))
        if len(ranked) > 2:
            picked.append(("WORST", ranked[-1]))
        parts = []
        for label, n in picked:
            scores = "\n".join(f"  {n.scores.get(v.name, 0.0):.2f}  {v.name}"
                               for v in self.vset.active)
            convo = render_conversation(n.state)
            parts.append(f"## {label}: plan n{n.id} (V = {n.value:.2f}, {n.depth} revisions)\n"
                         f"Scores:\n{scores}\n\nConversation:\n{convo}")
        return "\n\n".join(parts)

    async def evolve(self, nodes: list[Node]) -> list[Verifier]:
        try:
            data = await llm.generate_json(
                evolve_prompt(self.query, self.vset, self.describe_frontier(nodes),
                              n_verifiers=self.cfg.evolve_verifiers),
                model=self.model, provider=self.provider,
                reasoning_effort=self.reasoning_effort)
            self.llm_calls += 1
            return parse_verifiers(data, "evolved")
        except Exception:
            self.parse_failures += 1
            return []

    # ------------------------------------------------------------------- run
    async def run(self) -> SearchResult:
        cfg = self.cfg
        rng = random.Random(cfg.seed)

        # 1. seed rubric from the request + plan requirements (one LLM call)
        try:
            data = await llm.generate_json(
                seed_prompt(self.query, n_verifiers=cfg.seed_verifiers),
                model=self.model, provider=self.provider,
                reasoning_effort=self.reasoning_effort)
            self.llm_calls += 1
            seed_vs = parse_verifiers(data, "seed")
        except Exception:
            self.parse_failures += 1
            seed_vs = []
        for v in seed_vs:
            self.vset.add(v)
        self.log({"event": "verifiers", "verifiers": self.vset.to_json(), "values": {}})

        nodes: list[Node] = []
        next_id = 1

        def node_json(n: Node) -> dict:
            return {"id": n.id, "parent": n.parent, "depth": n.depth, "terminal": True,
                    "value": round(n.value, 4),
                    "scores": {k: round(v, 3) for k, v in n.scores.items()},
                    "reasons": n.reasons, "children": n.children_count,
                    "born_round": n.born_round, "n_turns": n.n_turns, "forced": n.forced,
                    "output": n.plan, "plan": n.plan, "prompt": n.prompt}

        async def add_child(parent: Node | None, state, plan, rnd, tool_results,
                            n_turns, forced, prompt):
            nonlocal next_id
            child = Node(id=next_id, parent=parent.id if parent else None,
                         depth=parent.depth + 1 if parent else 0, state=state, plan=plan,
                         born_round=rnd, n_turns=n_turns, forced=forced, prompt=prompt)
            next_id += 1
            if parent is not None:
                parent.children_count += 1
            await self.judge(child, self.vset.active)
            nodes.append(child)
            self.log({"event": "node", **node_json(child), "information": tool_results})

        def finish(reason: str, rnd: int) -> SearchResult:
            best = max(nodes, key=lambda n: (n.value, n.id)) if nodes else None  # ties -> later node
            plan = best.plan if best else ""
            self.log({"event": "done", "reason": reason, "rounds": rnd,
                      "best_node_id": best.id if best else None, "plan": plan})
            return SearchResult(plan=plan, reason=reason, rounds=rnd, n_nodes=len(nodes),
                                best_node_id=best.id if best else None,
                                verifiers=self.vset.to_json())

        # 2. seed plans: n_seed independent episodes
        await self.expand(None, cfg.n_seed, 0, add_child)
        if not nodes:
            return finish("empty", 0)

        # 3. revise / evolve
        for rnd in range(1, cfg.max_rounds + 1):
            parents = softmax_sample(rng, nodes, cfg.n_parents, cfg.tau, cfg.deg_coef)
            await asyncio.gather(*(self.expand(p, cfg.k_children, rnd, add_child)
                                   for p in parents))
            if cfg.evolve_every > 0 and rnd % cfg.evolve_every == 0:
                added = [v for v in await self.evolve(nodes) if self.vset.add(v)]
                if added:
                    await asyncio.gather(*(self.judge(n, added) for n in nodes))
                    self.log({"event": "verifiers", "verifiers": self.vset.to_json(),
                              "values": {n.id: {"value": round(n.value, 4),
                                                "scores": {k: round(x, 3)
                                                           for k, x in n.scores.items()},
                                                "reasons": n.reasons}
                                         for n in nodes}})
            self.log({"event": "round", "round": rnd})

        return finish("budget", cfg.max_rounds)
