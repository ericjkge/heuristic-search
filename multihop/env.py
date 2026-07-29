"""MuSiQue environment: agentic (think, search, information) partial states
over full-wiki BM25 retrieval (Elasticsearch, wiki-18 dump).

Terminology follows Search-R1: think / search / information / answer.
A state is a tuple of steps. Non-terminal steps are
    {"think": str, "search": str, "information": [{"title": ..., "text": ...}]}
and a terminal step is
    {"think": str, "answer": str}
Expansion = sample one more step from the LLM; retrieval and verifier scoring
are LLM-free.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import llm
from embeddings import get_embedder
from search import Node
from verifiers import VERIFIER_RULES, Verifier, VerifierSet

from .es_retriever import ESRetriever

Step = dict[str, Any]
State = tuple[Step, ...]

SYSTEM_PROMPT = """\
You answer multi-hop questions by searching Wikipedia.
Strictly follow this protocol on every turn:
1. Reason inside <think>...</think>.
2. Then take EXACTLY ONE action:
   - search for one fact: <search>your query</search>
   - or give the final answer: <answer>...</answer>
After each search you will receive results inside <information>...</information>.
Rules: one action per turn; each query should target a single fact (one entity
or relation), not the whole question; the final answer must be a short span
(typically 1-5 words), not a sentence.

The search engine is BM25, which ranks passages by keyword overlap rather than
meaning: a few distinctive keywords (names or titles plus a relation word like
"founded" or "cast") work better than full questions, and if results come back
off-topic, change or add keywords rather than repeating the same query.

Example workflow for the question "When was the institute that owned The
Collegian founded?":

<think>I first need to identify the institute that owns The Collegian, then
find when that institute was founded.</think>
<search>The Collegian student newspaper university</search>

<information>
Doc 1 (Title: The Collegian (Houston Baptist University)) The Collegian is the
official student publication of Houston Baptist University in Houston, Texas. ...
</information>

<think>The Collegian is owned by Houston Baptist University. Second hop: when
was Houston Baptist University founded?</think>
<search>Houston Baptist University founded</search>

<information>
Doc 1 (Title: Houston Baptist University) Houston Baptist University is a
private Baptist university founded in 1960 ...
</information>

<think>Houston Baptist University was founded in 1960. That answers the
question.</think>
<answer>1960</answer>"""

ANSWER_NOW = (
    "You have no searches left. Based on the information gathered so far, "
    "respond now with <think>...</think> followed by <answer>...</answer>."
)

MAX_DOC_CHARS = 1200  # per-document truncation inside <information> blocks

_THINK = re.compile(r"<think>(.*?)</think>", re.S)
_SEARCH = re.compile(r"<search>(.*?)</search>", re.S)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)


def render_messages(msgs: list[dict[str, str]]) -> str:
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in msgs)


class MusiqueEnv:
    def __init__(
        self,
        example: dict[str, Any],
        retriever: ESRetriever,
        model: str = llm.DEFAULT_MODEL,
        provider: str = llm.DEFAULT_PROVIDER,
        retrieval_k: int = 3,
        max_searches: int = 5,
        expand_temperature: float = 0.9,
    ):
        self.example = example
        self.question: str = example["question"]
        self.retriever = retriever
        self.model = model
        self.provider = provider
        self.retrieval_k = retrieval_k
        self.max_searches = max_searches
        self.expand_temperature = expand_temperature
        self.embedder = get_embedder()
        self.parse_failures = 0

    # ------------------------------------------------------------ state basics
    def initial_state(self) -> State:
        return ()

    def eval_vars(self, state: State) -> dict[str, Any]:
        titles: list[str] = []
        texts: list[str] = []
        seen: set[str] = set()
        for step in state:
            for doc in step.get("information", []):
                if doc["title"] not in seen:
                    seen.add(doc["title"])
                    titles.append(doc["title"])
                    texts.append(doc["text"])
        answer = next((s["answer"] for s in state if "answer" in s), "")
        return {
            "question": self.question,
            "n_steps": len(state),
            "thinks": [s.get("think", "") for s in state],
            "searches": [s["search"] for s in state if "search" in s],
            "titles": titles,
            "texts": texts,
            "all_text": " ".join(t + " " + x for t, x in zip(titles, texts)),
            "answer": answer,
            "is_terminal": bool(answer),
        }

    def state_to_json(self, state: State) -> Any:
        out = []
        for step in state:
            slim = {"think": step.get("think", "")[:300]}
            if "search" in step:
                slim["search"] = step["search"]
                slim["information_titles"] = [d["title"] for d in step.get("information", [])]
            if "answer" in step:
                slim["answer"] = step["answer"]
            out.append(slim)
        return out

    # ------------------------------------------------------------- LLM: expand
    def _messages(self, state: State) -> list[dict[str, str]]:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {self.question}"},
        ]
        n_searches = 0
        for step in state:
            if "search" in step:
                n_searches += 1
                msgs.append({
                    "role": "assistant",
                    "content": f"<think>{step['think']}</think>\n<search>{step['search']}</search>",
                })
                docs = "\n\n".join(
                    f"Doc {i+1} (Title: {d['title']}) {d['text'][:MAX_DOC_CHARS]}"
                    for i, d in enumerate(step.get("information", []))
                ) or "(no results)"
                msgs.append({"role": "user", "content": f"<information>\n{docs}\n</information>"})
        if n_searches >= self.max_searches:
            msgs.append({"role": "user", "content": ANSWER_NOW})
        return msgs

    async def _parse_step(self, text: str) -> Step | None:
        think = _THINK.search(text or "")
        search = _SEARCH.search(text or "")
        answer = _ANSWER.search(text or "")
        think_text = think.group(1).strip() if think else ""
        # if both actions appear, take whichever comes first
        if answer and (not search or answer.start() < search.start()):
            return {"think": think_text, "answer": answer.group(1).strip()}
        if search:
            query = search.group(1).strip()
            docs = await asyncio.to_thread(self.retriever.search, query, self.retrieval_k)
            return {"think": think_text, "search": query, "information": docs}
        return None

    async def expand(self, node: Node, k: int) -> list[tuple[State, bool, dict]]:
        msgs = self._messages(node.state)
        prompt_text = render_messages(msgs)
        completions = await llm.generate_many(
            [msgs] * k,
            model=self.model,
            provider=self.provider,
            temperature=self.expand_temperature,
            max_tokens=500,
            progress=False,
            return_exceptions=True,
        )
        out: list[tuple[State, bool, dict]] = []
        for text in completions:
            if isinstance(text, Exception):
                continue
            step = await self._parse_step(text)
            if step is None:
                self.parse_failures += 1
                continue
            meta = {
                "prompt": prompt_text,
                "output": text,
                "information": step.get("information", []),
            }
            out.append((node.state + (step,), "answer" in step, meta))
        return out

    # -------------------------------------------------------------- verifiers
    def builtin_verifiers(self) -> list[Verifier]:
        return []

    async def seed_verifiers(self) -> list[Verifier]:
        prompt = f"""\
You are designing verifiers for a search over candidate solution states for
the problem below. A verifier measures one property that distinguishes a
correct solution — or a state making real progress toward one — from a
plausible-but-wrong or unpromising one.

Problem: {self.question}

Write exactly 5 verifiers. Think about what information a correct solution MUST have
sought, what could make a WRONG candidate look right, and what search
direction would expose the difference. Cover the distinct pieces of
information the problem needs, not rephrasings of the same one.

{VERIFIER_RULES}

Return JSON: {{"verifiers": [{{"name": "snake_case_id", "description": "...",
"statement": "..."}}]}}"""
        return await self._request_verifiers(prompt, kind="seed")

    async def evolve_verifiers(
        self, nodes: list[Node], vset: VerifierSet
    ) -> list[Verifier]:
        ranked = sorted(nodes, key=lambda n: n.value, reverse=True)
        frontier = ranked[:4] + ranked[-2:] if len(ranked) > 6 else ranked
        existing = "\n".join(
            f"- {v.name}: {v.description} | statement: {v.statement}"
            for v in vset.active
        )
        described = "\n\n".join(self._describe_node(n) for n in frontier)
        prompt = f"""\
A search agent is answering this question:

Question: {self.question}

Current verifiers used to rank candidate states:
{existing}

Candidate states from the current search frontier — strongest and weakest
under the current verifier set, with per-verifier scores:

{described}

Propose 1-4 NEW verifiers, each doing one of:
(i) capture a property the strong states share and the weak states lack;
(ii) a property NO current state fully satisfies but a correct complete
solution would.
Only properties that meaningfully separate these candidates — nothing
redundant with the existing verifiers. If the existing set is adequate,
return an empty list.

{VERIFIER_RULES}

Return JSON: {{"verifiers": [{{"name": "snake_case_id", "description": "...",
"statement": "..."}}]}}"""
        return await self._request_verifiers(prompt, kind="evolved")

    def _describe_node(self, node: Node) -> str:
        ev = self.eval_vars(node.state)
        scores = ", ".join(f"{k}={v:.2f}" for k, v in node.scores.items())
        return (
            f"[node {node.id}] V={node.value:.3f} ({scores})\n"
            f"  searches: {ev['searches']}\n"
            f"  information (titles): {ev['titles']}\n"
            f"  answer: {ev['answer'] or '(none yet)'}"
        )

    async def _request_verifiers(self, prompt: str, kind: str) -> list[Verifier]:
        try:
            data = await llm.generate_json(
                prompt, model=self.model, provider=self.provider, temperature=0.7, max_tokens=1200
            )
        except ValueError:
            return []
        out = []
        for item in data.get("verifiers", []) if isinstance(data, dict) else []:
            if not (isinstance(item, dict) and item.get("name") and item.get("statement")):
                continue
            out.append(
                Verifier(
                    name=str(item["name"]),
                    description=str(item.get("description", "")),
                    statement=str(item["statement"]),
                    kind=kind,
                )
            )
        return out

    # ---------------------------------------------------------------- finalize
    async def finalize(self, node: Node) -> str:
        msgs = self._messages(node.state) + [{"role": "user", "content": ANSWER_NOW}]
        text = await llm.generate(msgs, model=self.model, provider=self.provider, temperature=0.0, max_tokens=300)
        m = _ANSWER.search(text or "")
        return m.group(1).strip() if m else (text or "").strip()

    # ---------------------------------------------------- greedy baseline
    async def run_greedy(self, log=None) -> str:
        """No-search-tree baseline: single greedy agentic chain.

        `log`, if given, receives the same node events the tree search emits,
        so greedy runs render in the viewer as a linear chain."""
        state: State = ()

        def emit(step: Step, prompt: str, output: str) -> None:
            if log is None:
                return
            depth = len(state)
            log({
                "event": "node", "id": depth, "parent": depth - 1, "depth": depth,
                "terminal": "answer" in step, "value": 0.0, "scores": {},
                "children": 0, "born_round": depth, "prompt": prompt,
                "output": output, "information": step.get("information", []),
                "state": self.state_to_json(state),
            })

        for _ in range(self.max_searches + 1):
            msgs = self._messages(state)
            text = await llm.generate(
                msgs, model=self.model, provider=self.provider,
                temperature=0.0, max_tokens=500,
            )
            step = await self._parse_step(text)
            if step is None:
                break
            state = state + (step,)
            emit(step, render_messages(msgs), text)
            if "answer" in step:
                return step["answer"]
        answer = await self.finalize(Node(id=-1, parent=None, depth=0, state=state, is_terminal=False))
        state = state + ({"think": "(forced finalize)", "answer": answer},)
        emit(state[-1], "(finalize)", answer)
        return answer
