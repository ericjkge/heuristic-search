"""Best-first search over partial solution states with self-improving verifiers.

Algorithm (one question = one search):
  1. env.seed_verifiers(): decompose the problem into initial subgoal verifiers
  2. Expand the root into n_seed seed nodes
  Per round:
  3. Score every node with the active verifier set; if a terminal node
     satisfies all verifiers, return it
  4. Sample parents ~ softmax((V(n) - deg(n)) / tau) over expandable
     (non-terminal) nodes, where deg(n) = number of children
  5. Generate k children per parent
  Every evolve_every rounds:
  6. env.evolve_verifiers(): propose new verifiers from the frontier
  7. Drop zero-variance verifiers, cap actives (DR Tulu)
  At budget end: best terminal node by V, else env.finalize(best node).

The environment owns everything task-specific (state representation, expansion
LLM calls, eval_vars extraction, verifier-generation prompts).
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .verifiers import Verifier, VerifierSet


@dataclass
class Node:
    id: int
    parent: int | None
    depth: int
    state: Any
    is_terminal: bool
    scores: dict[str, float] = field(default_factory=dict)
    value: float = 0.0
    children_count: int = 0
    born_round: int = 0
    prompt: str = ""  # rendered input prompt that produced this node
    output: str = ""  # raw LLM completion that produced this node
    information: list = field(default_factory=list)  # docs retrieved at this step


class Environment(Protocol):
    def initial_state(self) -> Any: ...
    def eval_vars(self, state: Any) -> dict[str, Any]: ...
    def state_to_json(self, state: Any) -> Any: ...
    def builtin_verifiers(self) -> list[Verifier]: ...
    async def seed_verifiers(self) -> list[Verifier]: ...
    async def expand(self, node: Node, k: int) -> list[tuple[Any, bool, dict]]: ...
    async def evolve_verifiers(
        self, nodes: list[Node], vset: VerifierSet
    ) -> list[Verifier]: ...
    async def finalize(self, node: Node) -> str: ...


@dataclass
class SearchConfig:
    n_seed: int = 2          # seed nodes off the root
    n_parents: int = 1       # parents sampled per round
    k_children: int = 1      # children per parent
    max_rounds: int = 12
    evolve_every: int = 0    # verifier self-improvement every R rounds; 0 disables
    tau: float = 1.0         # softmax temperature over node priority
    satisfy_threshold: float = 0.55  # cosine scores rarely near 1.0; sweepable
    max_active_verifiers: int = 8
    seed: int = 0


@dataclass
class SearchResult:
    answer: str
    reason: str  # "solved" | "budget_terminal" | "budget_forced"
    rounds: int
    n_nodes: int
    best_node_id: int | None
    nodes: list[dict[str, Any]]
    verifier_log: list[dict[str, Any]]
    verifiers: list[dict[str, Any]]


def softmax_sample(
    rng: random.Random, items: list[Node], n: int, tau: float
) -> list[Node]:
    """Sample up to n distinct nodes ~ softmax((value - children_count)/tau)."""
    pool = list(items)
    picked: list[Node] = []
    while pool and len(picked) < n:
        logits = [(x.value - x.children_count) / max(tau, 1e-6) for x in pool]
        m = max(logits)
        weights = [math.exp(l - m) for l in logits]
        chosen = rng.choices(range(len(pool)), weights=weights, k=1)[0]
        picked.append(pool.pop(chosen))
    return picked


async def best_first_search(
    env: Environment,
    cfg: SearchConfig,
    log: Callable[[dict[str, Any]], None] | None = None,
) -> SearchResult:
    """Run the search. `log`, if given, receives append-only JSONL-able events
    (node / verifiers / round / done) recording all LLM I/O as it happens —
    this is what the live viewer reads."""
    rng = random.Random(cfg.seed)

    def emit(ev: dict[str, Any]) -> None:
        if log is not None:
            log(ev)

    vset = VerifierSet(env.builtin_verifiers(), embedder=getattr(env, "embedder", None))
    for v in await env.seed_verifiers():
        vset.add(v)
    emit({"event": "verifiers", "verifiers": vset.to_json(), "values": {}})

    root = Node(id=0, parent=None, depth=0, state=env.initial_state(), is_terminal=False)
    nodes: list[Node] = [root]
    next_id = 1

    def rescore_all() -> None:
        for node in nodes:
            if node.id == 0:
                continue
            node.scores = vset.score_node(env.eval_vars(node.state))
            node.value = vset.aggregate(node.scores)

    def solved_node() -> Node | None:
        for node in nodes:
            if node.is_terminal and node.id != 0 and vset.all_satisfied(
                node.scores, cfg.satisfy_threshold
            ):
                return node
        return None

    def node_json(n: Node) -> dict[str, Any]:
        return {
            "id": n.id, "parent": n.parent, "depth": n.depth,
            "terminal": n.is_terminal, "value": round(n.value, 4),
            "scores": {k: round(v, 3) for k, v in n.scores.items()},
            "children": n.children_count, "born_round": n.born_round,
            "prompt": n.prompt, "output": n.output, "information": n.information,
            "state": env.state_to_json(n.state),
        }

    async def add_children(parents: list[Node], k: int, rnd: int) -> None:
        nonlocal next_id
        results = await asyncio.gather(*(env.expand(p, k) for p in parents))
        for parent, expansions in zip(parents, results):
            parent.children_count += len(expansions)
            for state, terminal, meta in expansions:
                child = Node(
                    id=next_id, parent=parent.id, depth=parent.depth + 1,
                    state=state, is_terminal=terminal, born_round=rnd,
                    prompt=meta.get("prompt", ""), output=meta.get("output", ""),
                    information=meta.get("information", []),
                )
                child.scores = vset.score_node(env.eval_vars(state))
                child.value = vset.aggregate(child.scores)
                nodes.append(child)
                next_id += 1
                emit({"event": "node", **node_json(child)})

    def finish(answer: str, reason: str, rnd: int, best: Node | None) -> SearchResult:
        emit({
            "event": "done", "answer": answer, "reason": reason, "rounds": rnd,
            "best_node_id": best.id if best else None,
        })
        return SearchResult(
            answer=answer, reason=reason, rounds=rnd, n_nodes=len(nodes) - 1,
            best_node_id=best.id if best else None,
            nodes=[node_json(n) for n in nodes],
            verifier_log=vset.log,
            verifiers=vset.to_json(),
        )

    # 2. seed solutions
    await add_children([root], cfg.n_seed, rnd=0)

    for rnd in range(1, cfg.max_rounds + 1):
        # 3. termination check
        if (winner := solved_node()) is not None:
            return finish(env.eval_vars(winner.state)["answer"], "solved", rnd, winner)

        # 4. parent sampling over expandable nodes
        expandable = [n for n in nodes if not n.is_terminal and n.id != 0]
        if not expandable:
            break
        parents = softmax_sample(rng, expandable, cfg.n_parents, cfg.tau)

        # 5. expansion
        await add_children(parents, cfg.k_children, rnd)

        # 6-7. verifier self-improvement (disabled when evolve_every == 0)
        if cfg.evolve_every > 0 and rnd % cfg.evolve_every == 0:
            for v in await env.evolve_verifiers(nodes[1:], vset):
                vset.add(v)
            rescore_all()
            vset.drop_no_variance(
                [n.scores for n in nodes[1:]], cfg.max_active_verifiers
            )
            rescore_all()
            emit({
                "event": "verifiers", "verifiers": vset.to_json(),
                "values": {
                    n.id: {"value": round(n.value, 4),
                           "scores": {k: round(v, 3) for k, v in n.scores.items()}}
                    for n in nodes[1:]
                },
            })
        emit({"event": "round", "round": rnd})

    if (winner := solved_node()) is not None:
        return finish(env.eval_vars(winner.state)["answer"], "solved", cfg.max_rounds, winner)

    # budget exhausted: best terminal answer, else force one from the best node
    terminals = [n for n in nodes if n.is_terminal]
    if terminals:
        best = max(terminals, key=lambda n: n.value)
        return finish(env.eval_vars(best.state)["answer"], "budget_terminal", cfg.max_rounds, best)
    candidates = [n for n in nodes if n.id != 0]
    if not candidates:
        return finish("", "budget_forced", cfg.max_rounds, None)
    best = max(candidates, key=lambda n: n.value)
    answer = await env.finalize(best)
    return finish(answer, "budget_forced", cfg.max_rounds, best)
