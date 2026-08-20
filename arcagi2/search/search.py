"""Verifier-guided best-first search for ARC-AGI-2.

Searches ONE test input of a task: nodes are candidate output grids for that
input. Verifiers (executable hypotheses about the transformation,
train-pair-validated) score every node; parents are sampled via softmax over
V(n) - deg_coef * children; expansion shows the parent's candidate plus its
failing verifier statements and asks for a revision. Every evolve_every
rounds the verifier set grows from the frontier. After max_rounds, the top-2
distinct candidates by V become this test input's two attempts.

Multi-test-input tasks run one independent search per input (run_search.py
loops over pairs).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import llm
from arcagi2.baselines.common import build_prompt, parse_grid
from arcagi2.tasks import Task
from verifiers import (Verifier, VerifierSet, evolve_prompt, retry_message,
                       seed_prompt)


@dataclass
class SearchConfig:
    n_seed: int = 3
    n_parents: int = 1
    k_children: int = 1
    max_rounds: int = 10
    evolve_every: int = 2  # 0 disables
    tau: float = 0.2
    deg_coef: float = 0.3
    feedback: bool = True
    random_parents: bool = False  # ablation: sample parents uniformly, ignoring V
    expand_temperature: float = 0.8
    max_feedback_items: int = 8
    n_inspirations: int = 2  # extra tried candidates shown on expansion; 0 disables
    seed_verifiers: str = "5-8"    # count range shown in the seed prompt
    evolve_verifiers: str = "1-3"  # count range shown in the evolve prompt


@dataclass
class Node:
    id: int
    parent: int | None
    candidate: Any  # output grid, or None if the response had no grid
    response: str
    value: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    children: int = 0


@dataclass
class SearchResult:
    attempts: list[Any]  # up to 2 distinct candidate grids, best first
    n_nodes: int
    rounds: int
    reason: str


class ArcSearch:
    def __init__(self, task: Task, pair_index: int = 0, *,
                 model: str, provider: str, reasoning_effort: str | None,
                 cfg: SearchConfig, log: Callable[[dict], None] = lambda ev: None):
        self.task = task
        self.pair_index = pair_index
        self.test_input = task.test[pair_index]["input"]
        self.model, self.provider = model, provider
        self.reasoning_effort = reasoning_effort
        self.cfg = cfg
        self.log = log
        self.vset = VerifierSet(task.train)
        self.nodes: list[Node] = []
        self.llm_calls = 0
        self.parse_failures = 0
        self.rng = random.Random(0)

    # ---------------------------------------------------------------- prompts

    def _inspirations(self, parent: Node) -> list[Node]:
        """Up to n_inspirations distinct tried candidates: the highest-V node
        whose grid differs from the parent's, then uniform-random others."""
        distinct: list[Node] = []
        for n in sorted(self.nodes, key=lambda n: -n.value):
            if (n.candidate is None or n.candidate == parent.candidate
                    or any(n.candidate == c.candidate for c in distinct)):
                continue
            distinct.append(n)
        chosen = distinct[:1]
        rest = distinct[1:]
        while rest and len(chosen) < self.cfg.n_inspirations:
            chosen.append(rest.pop(self.rng.randrange(len(rest))))
        return chosen

    def _satisfied(self, node: Node) -> tuple[int, int]:
        total = len(self.vset.verifiers)
        return sum(1 for v in self.vset.verifiers
                   if node.scores.get(v.name, 0.0) >= 1.0), total

    def guidance(self, parent: Node) -> str:
        """Feedback block appended after the baseline prompt on expansion."""
        if self.cfg.feedback and self.vset.verifiers:
            sat, total = self._satisfied(parent)
            lines = [f"--Parent Candidate--\nV = {parent.value:.2f} ({sat}/{total} verifiers satisfied):\n{json.dumps(parent.candidate)}"]
            checks = sorted(((v, parent.scores.get(v.name, 0.0))
                             for v in self.vset.verifiers), key=lambda x: x[1])
            failing = [(v, s) for v, s in checks if s < 1.0]
            if failing:
                lines.append("\nVerifiers are hypotheses about the transformation, each verified to hold on every training example. Failing verifiers:")
                for v, s in failing[: self.cfg.max_feedback_items]:
                    lines.append(f"- [{s:.2f}] {v.statement}")
            else:
                lines.append("\nVerifiers are hypotheses about the transformation, each verified to hold on every training example. All verifiers are satisfied.")
            inspirations = (self._inspirations(parent)
                            if self.cfg.n_inspirations else [])
            if inspirations:
                lines.append("\n--Other Tried Candidates--")
                for n in inspirations:
                    sat, total = self._satisfied(n)
                    lines.append(f"V = {n.value:.2f} ({sat}/{total} verifiers satisfied):\n{json.dumps(n.candidate)}\n")
            lines.append("--End of Candidates--")
            if failing:
                lines.append("\nRevise the parent candidate: keep what the satisfied verifiers confirm, fix what the failing verifiers flag, and re-derive any part you now believe was wrong. Give your final output grid at the end of your response.")
            else:
                lines.append("\nEvery current verifier is satisfied, but the verifiers may not capture the full rule. Re-derive the transformation from the training examples; if you conclude the parent candidate is right, return it unchanged, otherwise return your corrected version. Give your final output grid at the end of your response.")
        else:
            lines = [f"--Parent Candidate--\n{json.dumps(parent.candidate)}",
                     "\n--End of Candidates--",
                     "\nProduce an improved output grid. Give your final output grid at the end of your response."]
        return "\n".join(lines)

    def frontier_desc(self, k: int = 2) -> str:
        """Top-k distinct-candidate nodes by V plus the lowest-V distinct
        candidate for contrast, each with per-verifier scores."""
        distinct: list[Node] = []
        for n in sorted(self.nodes, key=lambda n: -n.value):
            if n.candidate is None or any(n.candidate == c.candidate for c in distinct):
                continue
            distinct.append(n)
        chosen = distinct[:k]
        if len(distinct) > k:
            chosen.append(distinct[-1])  # lowest-V contrast candidate
        parts = []
        for i, n in enumerate(chosen):
            scores = "\n".join(f"- [{n.scores.get(v.name, 0.0):.2f}] {v.statement}"
                               for v in self.vset.verifiers)
            label = "lowest-scoring candidate, for contrast" if i == k else f"Candidate {i}"
            parts.append(f"--{label} (V={n.value:.2f})--\n"
                         f"{json.dumps(n.candidate)}\n{scores}")
        return "\n\n".join(parts) if parts else "(no scored candidates yet)"

    # ---------------------------------------------------------------- steps

    async def _gen_verifiers(self, prompt: str, kind: str) -> list[Verifier]:
        """One generation call plus at most one conversational retry that
        shows the model its own response and why proposals were rejected."""
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        admitted_total: list[Verifier] = []
        for attempt in (0, 1):
            try:
                text = await llm.generate(
                    messages, model=self.model, provider=self.provider,
                    json_mode=True, reasoning_effort=self.reasoning_effort)
                self.llm_calls += 1
            except Exception as e:
                self.log({"event": "verifier_gen_error", "error": repr(e)})
                break
            problems: list[str] = []
            cands: list[Verifier] = []
            try:
                items = json.loads(llm._strip_code_fence(text or "")).get("verifiers", [])
            except (json.JSONDecodeError, AttributeError, TypeError) as e:
                items = []
                problems.append(f"the response was not a valid JSON object: {e}")
            for v in items if isinstance(items, list) else []:
                if isinstance(v, dict) and v.get("name") and v.get("statement") and v.get("code"):
                    cands.append(Verifier(name=str(v["name"]), statement=str(v["statement"]),
                                          code=str(v["code"]), kind=kind))
                else:
                    problems.append(f"entry {v.get('name', '?') if isinstance(v, dict) else '?'}: "
                                    "missing required fields (name/statement/code)")
            admitted, rejections = await self.vset.admit(cands)
            admitted_total.extend(admitted)
            problems.extend(f"{r['name']}: {r['reason']}" for r in rejections)
            self.log({"event": "verifiers", "kind": kind, "attempt": attempt,
                      "proposed": len(cands), "admitted": len(admitted),
                      "problems": problems, "verifiers": self.vset.to_json()})
            if not problems or attempt == 1:
                break
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": retry_message(problems)}]
        return admitted_total

    async def _score_node(self, node: Node) -> None:
        scores = await self.vset.score_pairs([[self.test_input, node.candidate]])
        node.scores = scores[0]
        node.value = self.vset.aggregate(node.scores)

    async def _expand(self, parent: Node | None, round_no: int) -> Node:
        prompt = build_prompt(self.task, self.pair_index)
        if parent is not None:
            # replace the official trailer with the feedback block
            trailer = "Your response:"
            assert prompt.endswith(trailer)
            prompt = prompt[: -len(trailer)].rstrip() + "\n\n" + self.guidance(parent)
        try:
            text = await llm.generate(
                prompt, model=self.model, provider=self.provider,
                temperature=self.cfg.expand_temperature,
                reasoning_effort=self.reasoning_effort)
            self.llm_calls += 1
        except Exception as e:
            text = ""
            self.log({"event": "expand_error", "error": repr(e)})
        candidate = parse_grid(text or "")
        if candidate is None:
            self.parse_failures += 1
        node = Node(id=len(self.nodes), parent=parent.id if parent else None,
                    candidate=candidate, response=text or "")
        if candidate is not None:
            await self._score_node(node)
        self.nodes.append(node)
        if parent is not None:
            parent.children += 1
        self.log({"event": "node", "id": node.id, "parent": node.parent,
                  "round": round_no, "value": node.value, "scores": node.scores,
                  "candidate": node.candidate, "prompt": prompt,
                  "output": (text or "")[-8000:]})
        return node

    def _sample_parents(self) -> list[Node]:
        pool = [n for n in self.nodes if n.candidate is not None]
        if not pool:
            return []
        logits = [(n.value - self.cfg.deg_coef * n.children) / self.cfg.tau
                  for n in pool]
        mx = max(logits)
        weights = [math.exp(l - mx) for l in logits]
        picks = []
        for _ in range(min(self.cfg.n_parents, len(pool))):
            picks.append(self.rng.choices(pool, weights=weights, k=1)[0])
        return picks

    async def _rescore_all(self) -> None:
        for n in self.nodes:
            if n.candidate is not None:
                await self._score_node(n)
        self.log({"event": "rescore",
                  "values": {n.id: n.value for n in self.nodes}})

    # ---------------------------------------------------------------- run

    async def run(self) -> SearchResult:
        await self._gen_verifiers(
            seed_prompt(self.task.train, [self.test_input],
                        self.cfg.seed_verifiers), "seed")

        import asyncio
        await asyncio.gather(*(self._expand(None, 0)
                               for _ in range(self.cfg.n_seed)))

        for round_no in range(1, self.cfg.max_rounds + 1):
            parents = self._sample_parents()
            await asyncio.gather(*(self._expand(p, round_no)
                                   for p in parents
                                   for _ in range(self.cfg.k_children)))
            if self.cfg.evolve_every and round_no % self.cfg.evolve_every == 0:
                added = await self._gen_verifiers(
                    evolve_prompt(self.task.train, [self.test_input], self.vset,
                                  self.frontier_desc(), self.cfg.evolve_verifiers),
                    "evolved")
                if added:
                    await self._rescore_all()
            self.log({"event": "round", "round": round_no})

        attempts: list[Any] = []
        for n in sorted(self.nodes, key=lambda n: -n.value):
            if n.candidate is not None and n.candidate not in attempts:
                attempts.append(n.candidate)
            if len(attempts) == 2:
                break
        self.log({"event": "done", "reason": "budget",
                  "rounds": self.cfg.max_rounds, "n_nodes": len(self.nodes),
                  "attempts": attempts})
        return SearchResult(attempts=attempts, n_nodes=len(self.nodes),
                            rounds=self.cfg.max_rounds, reason="budget")
