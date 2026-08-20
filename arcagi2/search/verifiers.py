"""Executable hypothesis verifiers for ARC-AGI-2 search.

A Verifier is a hypothesis about the transformation rule: an English
statement plus a Python predicate  check(inp, out) -> bool | float in [0,1].
Checks are relational — they judge one (input, output) pair at a time with no
access to the training set — and a verifier is admitted only if its check
passes on EVERY training pair. V(node) = mean of check scores on
(test_input, candidate). The set grows during search (add-only): seeds come
from the training pairs; evolution adds finer-grained hypotheses using what
the frontier reveals.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Any

CHECK_TIMEOUT_S = 20.0  # whole sandbox batch
ADMIT_THRESHOLD = 0.99  # min score on every train pair for admission

# Runs in a separate `python -I` process: exec each verifier's code, apply
# check() to each pair, 1s alarm per call so one bad check can't stall the
# batch. Prints one {"scores", "errors", "compile_error"} object per verifier.
_RUNNER = r"""
import json, signal, sys

class _Timeout(Exception): pass
signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(_Timeout()))

payload = json.load(sys.stdin)
results = []
for job in payload["jobs"]:
    fn, compile_error = None, None
    try:
        ns = {}
        exec(job["code"], ns)
        fn = ns.get("check")
        if fn is None:
            compile_error = "no function named 'check' defined"
    except Exception as e:
        compile_error = repr(e)
    scores, errors = [], []
    for inp, out in payload["pairs"]:
        r, err = 0.0, None
        if fn is not None:
            try:
                signal.alarm(2)
                v = fn(inp, out)
                r = 1.0 if v is True else 0.0 if (v is False or v is None) else float(v)
            except _Timeout:
                err = "timed out (2s)"
            except Exception as e:
                err = repr(e)
            finally:
                signal.alarm(0)
        scores.append(max(0.0, min(1.0, r)))
        errors.append(err)
    results.append({"scores": scores, "errors": errors, "compile_error": compile_error})
print(json.dumps(results))
"""


def _zeroed(n_codes: int, n_pairs: int, reason: str) -> list[dict[str, Any]]:
    return [{"scores": [0.0] * n_pairs, "errors": [None] * n_pairs,
             "compile_error": reason} for _ in range(n_codes)]


async def run_checks(codes: list[str], pairs: list[list[Any]]) -> list[dict[str, Any]]:
    """Run every code's check() on every (inp, out) pair in a sandboxed
    subprocess. Returns one dict per code: "scores" (per pair, 0.0 on error),
    "errors" (per-pair exception text or None), "compile_error" (str or None).
    """
    if not codes or not pairs:
        return _zeroed(len(codes), len(pairs), "nothing to run")
    payload = json.dumps({"jobs": [{"code": c} for c in codes], "pairs": pairs})
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-I", "-c", _RUNNER,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL)
    try:
        out, _ = await asyncio.wait_for(
            proc.communicate(payload.encode()), CHECK_TIMEOUT_S)
        return json.loads(out)
    except (asyncio.TimeoutError, json.JSONDecodeError):
        if proc.returncode is None:
            proc.kill()
        return _zeroed(len(codes), len(pairs), "sandbox batch failed or timed out")


@dataclass
class Verifier:
    name: str
    statement: str  # the hypothesis, in one sentence
    code: str       # def check(inp, out): ...
    kind: str = "seed"  # seed | evolved

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "statement": self.statement,
                "code": self.code, "kind": self.kind}


class VerifierSet:
    def __init__(self, train_pairs: list[dict[str, Any]]):
        self.train_pairs = [[p["input"], p["output"]] for p in train_pairs]
        self.verifiers: list[Verifier] = []

    async def admit(self, candidates: list[Verifier]
                    ) -> tuple[list[Verifier], list[dict[str, str]]]:
        """Validate candidates on the training pairs; keep only those whose
        check passes every pair. Returns (admitted, rejections) where each
        rejection is {"name", "reason"} suitable for a retry message."""
        rejections: list[dict[str, str]] = []
        fresh: list[Verifier] = []
        for v in candidates:
            dup = next((x for x in self.verifiers + fresh
                        if x.name == v.name or x.statement == v.statement
                        or x.code == v.code), None)
            if dup is not None:
                rejections.append({"name": v.name,
                                   "reason": f"duplicate of verifier '{dup.name}'"})
            else:
                fresh.append(v)
        results = await run_checks([v.code for v in fresh], self.train_pairs)
        admitted: list[Verifier] = []
        for v, res in zip(fresh, results):
            if res["compile_error"]:
                rejections.append({"name": v.name,
                                   "reason": f"code failed to load: {res['compile_error']}"})
            elif all(s >= ADMIT_THRESHOLD for s in res["scores"]):
                admitted.append(v)
                self.verifiers.append(v)
            else:
                fails = [(i, s, e) for i, (s, e) in
                         enumerate(zip(res["scores"], res["errors"]))
                         if s < ADMIT_THRESHOLD]
                parts = [f"train pair {i}: " + (f"raised {e}" if e else f"returned {s:.2f}")
                         for i, s, e in fails[:3]]
                rejections.append({"name": v.name,
                                   "reason": "does not hold on the training pairs — "
                                             + "; ".join(parts)})
        return admitted, rejections

    async def score_pairs(self, pairs: list[list[Any]]) -> list[dict[str, float]]:
        """Score every verifier on each (input, candidate) pair in one sandbox
        call. Returns one {name: score} dict per pair."""
        results = await run_checks([v.code for v in self.verifiers], pairs)
        return [{v.name: r["scores"][i] for v, r in zip(self.verifiers, results)}
                for i in range(len(pairs))]

    def aggregate(self, scores: dict[str, float]) -> float:
        vals = [scores[v.name] for v in self.verifiers if v.name in scores]
        return sum(vals) / len(vals) if vals else 0.0

    def to_json(self) -> list[dict[str, Any]]:
        return [v.to_json() for v in self.verifiers]


def render_pairs(train_pairs: list[dict[str, Any]]) -> str:
    out = ""
    for i, p in enumerate(train_pairs):
        out += f"--Example {i}--\nINPUT:\n{json.dumps(p['input'])}\n"
        out += f"OUTPUT:\n{json.dumps(p['output'])}\n\n"
    return out.rstrip()


def render_test_inputs(test_inputs: list[Any]) -> str:
    if len(test_inputs) == 1:
        return json.dumps(test_inputs[0])
    return "\n".join(f"--Test Input {i}--\n{json.dumps(g)}"
                     for i, g in enumerate(test_inputs))


RULES = """\
Each verifier encodes ONE hypothesis about the transformation rule, with two fields:
- "statement": the hypothesis in one precise sentence
- "code": a Python function that checks the hypothesis on a single pair:

    def check(inp, out):
        # inp: input grid; out: candidate output grid
        # both list-of-lists of ints 0-9
        return score

Scoring:
- Return a float in [0, 1]: 1.0 = `out` fully satisfies the hypothesis, 0.0 = clearly violates it. Booleans work for all-or-nothing checks.
- Prefer graded scores (fraction of cells/rows/objects satisfying the property) where natural — they give the search a smoother signal.
- Use only the Python standard library (import inside the function if needed).

Admission: each verifier is automatically run on every training pair and kept only if check(train_input, train_output) passes on ALL of them — so state hypotheses precise enough to actually hold.

Cover distinct facets of the rule:
- size: how output dimensions relate to input dimensions
- palette: which colors appear, disappear, or map to others
- structure: objects, counts, positions, symmetries, repetition, per-region or conditional rules ("in the left half...", "cells adjacent to 8s...")

GOOD (all-or-nothing):
    statement: "The output grid is 3 times the input's height and width"
    def check(inp, out):
        return len(out) == 3 * len(inp) and len(out[0]) == 3 * len(inp[0])

GOOD (graded):
    statement: "Nonzero input cells keep their color in place"
    def check(inp, out):
        cells = [(r, c) for r, row in enumerate(inp) for c, v in enumerate(row) if v]
        if not cells:
            return True
        ok = sum(1 for r, c in cells
                 if r < len(out) and c < len(out[0]) and out[r][c] == inp[r][c])
        return ok / len(cells)

BAD:
    def check(inp, out): return True    # vacuous — always passes
    hardcoding an expected output grid copied from a training example"""


def retry_message(problems: list[str]) -> str:
    listing = "\n".join(f"- {p}" for p in problems)
    return f"""\
Some of your verifiers were rejected:

{listing}

Resubmit ONLY replacements for the rejected verifiers, in the same JSON format:
- If the response or an entry was malformed, or the code crashed, fix the format or repair the code.
- If a check scored below 1.0 on a training pair, the hypothesis is false as stated: revise or narrow it until it actually holds, and keep the statement and the code in agreement — or drop it if you no longer believe it.
Do not resubmit verifiers that were already accepted."""


def seed_prompt(train_pairs: list[dict[str, Any]], test_inputs: list[Any],
                n_verifiers: str = "5-8") -> str:
    return f"""\
You are writing verifiers to guide a search over candidate solutions to an ARC-AGI puzzle. Identify the transformation that maps each input grid to its output grid. Candidate outputs for the test input{'s' if len(test_inputs) > 1 else ''} below will be scored against your verifiers.

# TRAINING EXAMPLES
{render_pairs(train_pairs)}

# TEST INPUT{'S' if len(test_inputs) > 1 else ''}
{render_test_inputs(test_inputs)}

# VERIFIER RULES
{RULES}

# TASK
Write {n_verifiers} verifiers, from coarse invariants (size, palette, background) to the finest-grained rules you can state with confidence. Together they should separate correct outputs from plausible-but-wrong ones.

Each verifier must check a DIFFERENT facet of the rule. Never restate a hypothesis another verifier already encodes, even in different words. A wrong belief written three ways triples its damage to the search.

# OUTPUT FORMAT
Return JSON: {{"verifiers": [{{"name": "snake_case_id", "statement": "...", "code": "def check(inp, out):\\n    ..."}}]}}"""


def evolve_prompt(train_pairs: list[dict[str, Any]], test_inputs: list[Any],
                  vset: VerifierSet, frontier_desc: str,
                  n_verifiers: str = "1-3") -> str:
    existing = "\n".join(f'- {v.name}: "{v.statement}"' for v in vset.verifiers)
    return f"""\
You are adding verifiers to guide a search over candidate solutions to an ARC-AGI puzzle, using what the search has revealed so far.

# TRAINING EXAMPLES
{render_pairs(train_pairs)}

# TEST INPUT{'S' if len(test_inputs) > 1 else ''}
{render_test_inputs(test_inputs)}

# VERIFIER RULES
{RULES}

# CURRENT VERIFIERS
{existing}

# CURRENT SEARCH FRONTIER
{frontier_desc}

# TASK
Propose {n_verifiers} NEW verifiers that sharpen the search:
- finer-grained sub-rules of hypotheses the frontier already satisfies
- checks that separate frontier candidates which currently score the same
- aspects of the transformation no current verifier covers

A new verifier must be independent of the existing ones: it must be able to pass a candidate that an existing verifier fails, or vice versa. Avoid restating verifiers and creating redundancy. If nothing useful is missing, return an empty list.

# OUTPUT FORMAT
Return JSON: {{"verifiers": [{{"name": "snake_case_id", "statement": "...", "code": "def check(inp, out):\\n    ..."}}]}}"""
