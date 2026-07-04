"""Decompose the root goal into K soft verifiers (one LLM call).

BES-style basic checks only: parse the JSON array, keep items that have both
description and verify_code. A broken verify_code is not gated out -- it just
scores 0.0 at eval time via verifiers.verify_one.
"""

import json
import re

DECOMPOSE_PROMPT = """
## Problem
Pack {n} unit regular hexagons (side 1) into the canonical origin-centered regular
hexagon: flat-top, edge-normals along 30/90/150 degrees, so a point p is inside
iff |p . u| <= s*sqrt(3)/2 for those three unit vectors u. The objective is to
MINIMIZE the outer side s. The best known s for this instance is about {s_target};
a rudimentary aligned-honeycomb baseline gives s ~ {s_ref}.

## Task
Design exactly {k} SOFT VERIFIERS: structural heuristics that tend to score higher
for packings with a smaller s. Each is a single Python expression returning a DENSE
score in [0, 1] -- prefer the form min(1.0, <measured> / <target>); a bool is allowed
(True -> 1, False -> 0) but wastes gradient. These are diagnostic measurements used
to rank packings and report weaknesses; they never prescribe fixes.

Each expression is evaluated with eval() over this namespace:
  centers   (n, 2)    hexagon centers
  angles    (n,)      orientations in radians
  vertices  (n, 6, 2) every hexagon's 6 vertices
  x, y      (n,)      center coordinates
  s         float     outer side length
  n         int       number of hexagons
  np, math  modules, plus standard Python builtins

## Hard requirements for verify_code
- a SINGLE expression: no statements, no ';', no newlines, no def/for/if-statements
  (use comprehensions / inline expressions instead);
- must run WITHOUT error on any feasible packing;
- returns a number in [0, 1] (or a bool);
- must NOT be a function of s alone: two different packings with the SAME s must be
  able to get different scores. BAD: min(1.0, n / s**2) -- that is the raw objective
  restated and adds no information;
- must NOT be saturated: set targets/thresholds beyond what a rudimentary aligned
  honeycomb already achieves, so the score has headroom as packings improve toward
  s ~ {s_target}. BAD: counting touching neighbor pairs against a target the
  baseline honeycomb already exceeds.

## Tip
Pick the {k} verifiers from DIFFERENT categories, e.g.: boundary / container-edge
utilization, wasted corner/wedge area, orientation structure (aligned honeycombs
are rarely optimal -- the best known packings use mixed, nonzero rotations), local
contact geometry, radial mass distribution.

Output ONLY a JSON array of exactly {k} objects, nothing else:
[{{"description": "<short property description>", "verify_code": "<single expression>"}}, ...]
"""

def _extract_json_array(text):
    """Pull a JSON array out of an LLM response (tolerates ``` fences / prose)."""
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    raw = m.group(1) if m else None
    if raw is None:
        m = re.search(r"(\[\s*\{.*\}\s*\])", text, re.DOTALL)
        raw = m.group(1) if m else None
    if raw is None:
        raise ValueError(f"no JSON array found in decomposition response:\n{text[:500]}")
    return json.loads(raw)


def _parse_verifiers(text):
    out = []
    for it in _extract_json_array(text):
        if isinstance(it, dict) and "description" in it and "verify_code" in it:
            out.append({
                "description": str(it["description"]).strip(),
                "verify_code": str(it["verify_code"]).strip(),
            })
    return out


def decompose(llm, n, s_ref, s_target, k):
    """Call the LLM once; return up to K soft verifiers for instance n."""
    prompt = DECOMPOSE_PROMPT.format(n=n, s_ref=round(s_ref, 3), s_target=s_target, k=k)
    text = llm.call(
        [{"role": "user", "content": prompt}],
        tags={"instance": f"n{n}", "condition": "decompose", "phase": "decompose",
              "step": None, "cand_id": None, "parent_id": None},
    )
    verifiers = _parse_verifiers(text)
    if not verifiers:
        raise ValueError("decompose produced no valid verifiers")
    return verifiers[:k]


if __name__ == "__main__":
    fake = """```json
[
  {"description": "packing density", "verify_code": "min(1.0, n / s**2)"},
  {"description": "missing its verify_code"}
]
```"""
    for v in _parse_verifiers(fake):
        print(" -", v["description"], "->", v["verify_code"])
