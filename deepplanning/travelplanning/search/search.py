"""Best-first search over travel-planning agent states (seed-once verifiers).

State = the OpenAI message list so far (system, user, assistant tool-call
turns, tool results). One expansion = one assistant turn: the model either
issues tool calls (executed inline against the task's sandbox databases) or
emits the final <plan>...</plan>, which makes the node terminal.

Search loop mirrors multihop: seed verifiers once from the query, expand seeds,
sample parents ~ softmax((V - deg)/tau), stop when a terminal node satisfies
every verifier at the threshold or budget ends (best terminal by V, else a
forced plan from the best node).
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
from embeddings import get_embedder
from prompts import get_system_prompt

from verifiers import Verifier, VerifierSet, evolve_prompt, seed_prompt

PLAN_NOW = (
    "You have used all available tool-call turns. Based on the information "
    "gathered so far, produce your final complete travel plan now, strictly "
    "following the required format."
)

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
    """Tool call -> keyword text that verifier statements embed against."""
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
    depth: int  # number of assistant turns
    state: list[dict]  # full message list
    is_terminal: bool
    plan: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    value: float = 0.0
    children_count: int = 0
    born_round: int = 0
    output: str = ""  # rendered assistant turn that produced this node
    prompt: str = ""  # rendered input messages that produced this node


@dataclass
class SearchConfig:
    n_seed: int = 2
    n_parents: int = 1
    k_children: int = 1
    max_rounds: int = 16
    evolve_every: int = 4        # add verifiers every R rounds; 0 disables
    tau: float = 0.2
    deg_coef: float = 0.3        # V-point price per child in sampling
    satisfy_threshold: float = 0.7
    expand_temperature: float = 0.8
    max_tokens: int = 4000
    seed: int = 0


@dataclass
class SearchResult:
    plan: str
    reason: str  # solved | budget_terminal | budget_forced
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
                 log: Callable[[dict], None] | None = None, language: str = "en"):
        self.example = example
        self.query: str = example["query"]
        self.sample_id = str(example["id"])
        self.model, self.provider = model, provider
        self.cfg = cfg
        self.log = log or (lambda ev: None)
        self.embedder = get_embedder()
        self.openai_tools, self.tool_instances = load_tools(self.sample_id, language)
        self.system_prompt = get_system_prompt(language)
        self.parse_failures = 0
        self.llm_calls = 0

    # ------------------------------------------------------------------ state
    def initial_state(self) -> list[dict]:
        return [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.query}]

    def eval_vars(self, node: Node) -> dict[str, Any]:
        searches = [
            render_tool_call(tc["function"]["name"], tc["function"]["arguments"])
            for m in node.state if m.get("role") == "assistant"
            for tc in m.get("tool_calls") or []
        ]
        return {"searches": searches, "plan": node.plan, "n_turns": node.depth}

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

    # -------------------------------------------------------------- expansion
    async def expand(self, node: Node, k: int, rnd: int, add_child) -> None:
        msgs = list(node.state)
        prompt_text = render_messages(msgs)

        async def one() -> None:
            try:
                resp = await llm.generate(
                    msgs, model=self.model, provider=self.provider,
                    temperature=self.cfg.expand_temperature,
                    max_tokens=self.cfg.max_tokens,
                    tools=self.openai_tools,
                    return_response=True,
                )
            except Exception:
                self.parse_failures += 1
                return
            self.llm_calls += 1
            msg = message_to_dict(resp.choices[0].message)
            new_state = msgs + [msg]
            tool_calls = msg.get("tool_calls") or []
            rendered = [render_tool_call(tc["function"]["name"], tc["function"]["arguments"])
                        for tc in tool_calls]
            tool_results = []
            for tc in tool_calls:
                result = await asyncio.to_thread(
                    self._exec_tool, tc["function"]["name"], tc["function"]["arguments"])
                tool_msg = {"role": "tool", "tool_call_id": tc["id"],
                            "name": tc["function"]["name"], "content": result}
                new_state.append(tool_msg)
                tool_results.append({"name": tc["function"]["name"], "content": result})
            plan = "" if tool_calls else extract_plan(msg.get("content", ""))
            if not tool_calls and not plan:
                self.parse_failures += 1  # neither tools nor a plan: dead turn
                return
            add_child(node, new_state, bool(plan), plan, rnd,
                      output=(msg.get("content", "") + ("\n" if rendered else "")
                              + "\n".join(rendered)).strip(),
                      information=tool_results, prompt=prompt_text)

        await asyncio.gather(*(one() for _ in range(k)))

    def describe_frontier(self, nodes: list[Node], k: int = 2,
                          cap_per_traj: int = 120_000) -> str:
        """The top-k distinct trajectories by V, with FULL tool history:
        every turn's rendered calls and complete tool responses."""
        by_id = {n.id: n for n in nodes}

        def path_ids(n: Node) -> set[int]:
            ids = set()
            cur: Node | None = n
            while cur is not None and cur.id != 0:
                ids.add(cur.id)
                cur = by_id.get(cur.parent)
            return ids

        ranked = sorted((n for n in nodes if n.id != 0),
                        key=lambda n: n.value, reverse=True)
        picked: list[Node] = []
        picked_paths: list[set[int]] = []
        for n in ranked:
            pids = path_ids(n)
            # skip nodes on an already-picked trajectory (ancestor/descendant)
            if any(n.id in pp or p.id in pids for p, pp in zip(picked, picked_paths)):
                continue
            picked.append(n)
            picked_paths.append(pids)
            if len(picked) >= k:
                break

        parts = []
        for n in picked:
            lines = [f"[trajectory n{n.id}, V={n.value:.2f}, {n.depth} turns]"]
            turn = 0
            for m in n.state:
                if m.get("role") == "assistant":
                    calls = [render_tool_call(tc["function"]["name"],
                                              tc["function"]["arguments"])
                             for tc in m.get("tool_calls") or []]
                    if calls:
                        turn += 1
                        lines.append(f"turn {turn} calls: " + "; ".join(calls))
                elif m.get("role") == "tool":
                    lines.append(f"  result ({m.get('name', 'tool')}): "
                                 + (m.get("content") or ""))
            block = "\n".join(lines)
            if len(block) > cap_per_traj:
                block = block[:cap_per_traj] + " ...[truncated]"
            parts.append(block)
        return "\n\n".join(parts)

    async def evolve(self, vset: VerifierSet, nodes: list[Node]) -> list[Verifier]:
        try:
            data = await llm.generate_json(
                evolve_prompt(self.query, vset, self.describe_frontier(nodes)),
                model=self.model, provider=self.provider, temperature=0.7,
                max_tokens=1500)
            self.llm_calls += 1
            return [
                Verifier(name=str(i["name"]), description=str(i.get("description", "")),
                         statement=str(i["statement"]), kind="evolved")
                for i in data.get("verifiers", [])
                if isinstance(i, dict) and i.get("name") and i.get("statement")
            ]
        except Exception:
            return []

    async def finalize(self, node: Node) -> tuple[str, str, str]:
        """Force a plan from `node`. Returns (plan, rendered_prompt, raw_text)."""
        msgs = list(node.state) + [{"role": "user", "content": PLAN_NOW}]
        prompt_text = render_messages(msgs)
        try:
            resp = await llm.generate(
                msgs, model=self.model, provider=self.provider, temperature=0.0,
                max_tokens=self.cfg.max_tokens, return_response=True)
            self.llm_calls += 1
            text = resp.choices[0].message.content or ""
            return extract_plan(text) or text, prompt_text, text
        except Exception:
            return "", prompt_text, ""

    # ------------------------------------------------------------------- run
    async def run(self) -> SearchResult:
        cfg = self.cfg
        rng = random.Random(cfg.seed)

        # 1. seed verifiers from the query (one LLM call, once)
        try:
            data = await llm.generate_json(
                seed_prompt(self.query), model=self.model, provider=self.provider,
                temperature=0.7, max_tokens=2500)
            self.llm_calls += 1
            seed_vs = [
                Verifier(name=str(i["name"]), description=str(i.get("description", "")),
                         statement=str(i["statement"]),
                         kind=str(i.get("kind", "constraint")))
                for i in data.get("verifiers", [])
                if isinstance(i, dict) and i.get("name") and i.get("statement")
            ]
        except Exception:
            seed_vs = []
        vset = VerifierSet(seed_vs, embedder=self.embedder)
        self.log({"event": "verifiers", "verifiers": vset.to_json(), "values": {}})

        root = Node(id=0, parent=None, depth=0, state=self.initial_state(), is_terminal=False)
        nodes: list[Node] = [root]
        next_id = 1

        def node_json(n: Node) -> dict:
            return {"id": n.id, "parent": n.parent, "depth": n.depth,
                    "terminal": n.is_terminal, "value": round(n.value, 4),
                    "scores": {k: round(v, 3) for k, v in n.scores.items()},
                    "children": n.children_count, "born_round": n.born_round,
                    "output": n.output, "plan": n.plan, "prompt": n.prompt}

        def add_child(parent: Node, state, terminal, plan, rnd, output, information,
                      prompt=""):
            nonlocal next_id
            child = Node(id=next_id, parent=parent.id, depth=parent.depth + 1,
                         state=state, is_terminal=terminal, plan=plan,
                         born_round=rnd, output=output, prompt=prompt)
            child.scores = vset.score_node(self.eval_vars(child))
            child.value = vset.aggregate(child.scores)
            parent.children_count += 1
            nodes.append(child)
            next_id += 1
            self.log({"event": "node", **node_json(child), "information": information})

        def solved() -> Node | None:
            for n in nodes:
                if n.is_terminal and vset.all_satisfied(n.scores, cfg.satisfy_threshold):
                    return n
            return None

        def finish(plan: str, reason: str, rnd: int, best: Node | None) -> SearchResult:
            self.log({"event": "done", "reason": reason, "rounds": rnd,
                      "best_node_id": best.id if best else None,
                      "plan": plan})
            return SearchResult(plan=plan, reason=reason, rounds=rnd,
                                n_nodes=len(nodes) - 1,
                                best_node_id=best.id if best else None,
                                verifiers=vset.to_json())

        # 2. seed solutions
        await self.expand(root, cfg.n_seed, 0, add_child)

        for rnd in range(1, cfg.max_rounds + 1):
            if (win := solved()) is not None:
                return finish(win.plan, "solved", rnd, win)
            expandable = [n for n in nodes if not n.is_terminal and n.id != 0]
            if not expandable:
                break
            parents = softmax_sample(rng, expandable, cfg.n_parents, cfg.tau, cfg.deg_coef)
            await asyncio.gather(*(self.expand(p, cfg.k_children, rnd, add_child)
                                   for p in parents))
            if cfg.evolve_every > 0 and rnd % cfg.evolve_every == 0:
                added = await self.evolve(vset, nodes)
                if added:
                    for v in added:
                        vset.add(v)
                    for n in nodes:
                        if n.id != 0:
                            n.scores = vset.score_node(self.eval_vars(n))
                            n.value = vset.aggregate(n.scores)
                    self.log({"event": "verifiers", "verifiers": vset.to_json(),
                              "values": {n.id: {"value": round(n.value, 4),
                                                "scores": {k: round(x, 3)
                                                           for k, x in n.scores.items()}}
                                         for n in nodes if n.id != 0}})
            self.log({"event": "round", "round": rnd})

        if (win := solved()) is not None:
            return finish(win.plan, "solved", cfg.max_rounds, win)
        terminals = [n for n in nodes if n.is_terminal]
        if terminals:
            best = max(terminals, key=lambda n: n.value)
            return finish(best.plan, "budget_terminal", cfg.max_rounds, best)
        candidates = [n for n in nodes if n.id != 0]
        if not candidates:
            return finish("", "budget_forced", cfg.max_rounds, None)
        best = max(candidates, key=lambda n: n.value)
        plan, prompt_text, raw = await self.finalize(best)
        # log the forced plan as a synthetic terminal node so it shows in the viewer
        forced_state = list(best.state) + [{"role": "user", "content": PLAN_NOW},
                                           {"role": "assistant", "content": raw}]
        add_child(best, forced_state, True, plan, cfg.max_rounds,
                  output=raw.strip(), information=[], prompt=prompt_text)
        return finish(plan, "budget_forced", cfg.max_rounds, nodes[-1])
