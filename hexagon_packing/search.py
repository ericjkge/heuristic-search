"""Verifier-guided search for one hexagon-packing instance.

decompose (skipped when w_soft=0) -> seed: revise initial.py x num_seeds ->
hard gate (evaluate.py subprocess) -> score = w_raw*raw01 + w_soft*mean(soft) ->
expand top_k x branching -> repeat num_steps.

Each candidate is a full program. Revisions alternate 50/50 between BES diffs and
full rewrites (OpenEvolve: constructor problems need whole-algorithm transitions
that don't fit through diff-shaped edits). Feedback is constraint-only:
measurements, never suggested fixes. w_soft=0 is the raw-only ablation.
"""

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import diff
import scoring
import verifiers
from concurrency import run_parallel
from decompose import decompose

_HERE = Path(__file__).parent
INITIAL_PY = _HERE / "initial.py"
EVALUATE_PY = str(_HERE / "evaluate.py")


@dataclass
class Candidate:
    cand_id: str
    parent_id: str | None
    step: int
    source: str
    feasible: bool
    raw_s: float | None
    soft: list = field(default_factory=list)
    score: float | None = None
    reason: str = ""


REVISE_PROMPT = """\
You are improving a Python program that packs {n} unit regular hexagons (side 1) into
the canonical origin-centered regular hexagon: flat-top, so a point p is inside iff
|p . u| <= s*sqrt(3)/2 for the unit vectors u at 30/90/150 degrees. MINIMIZE the outer
side s. The program defines pack(n) -> (centers, angles, s); keep that signature, keep
the container convention, and keep every hexagon non-overlapping and inside the container.
Each hexagon's position and rotation are free: hexagons may be individually rotated and
placed off-lattice, and the best known packings for problems like this often use mixed, nonzero
rotations rather than a perfectly aligned honeycomb.

Current program:
```python
{source}
```

Measurements of the current program's packing:
{stats}

Propose one or more edits that reduce s. You may add optimization inside pack() (e.g.
scipy.optimize); the program is run with a {timeout}-second limit. Prefer editing inside
the EVOLVE-BLOCK. Present each edit as a SEARCH/REPLACE block in EXACTLY this format,
where the SEARCH text matches the current program verbatim:
<<<<<<< SEARCH
<lines to find>
=======
<replacement lines>
>>>>>>> REPLACE

Output only diff blocks, no prose.
"""

REWRITE_PROMPT = """\
You are improving a Python program that packs {n} unit regular hexagons (side 1) into
the canonical origin-centered regular hexagon: flat-top, so a point p is inside iff
|p . u| <= s*sqrt(3)/2 for the unit vectors u at 30/90/150 degrees. MINIMIZE the outer
side s. The program defines pack(n) -> (centers, angles, s); keep that signature, keep
the container convention, and keep every hexagon non-overlapping and inside the container.
Each hexagon's position and rotation are free: hexagons may be individually rotated and
placed off-lattice, and the best known packings for problems like this often use mixed, nonzero
rotations rather than a perfectly aligned honeycomb.

Current program:
```python
{source}
```

Measurements of the current program's packing:
{stats}

Rewrite the program to reduce s. You may restructure it completely -- change the
construction strategy, add numerical optimization (e.g. scipy.optimize), anything --
as long as pack(n) keeps its signature and returns a valid packing. The program is
run with a {timeout}-second limit.

Output the COMPLETE new program in a single ```python code block, nothing else.
"""


def _extract_program(text):
    """Pull the rewritten program out of a ```python fence. Raises if absent/invalid."""
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if not blocks:
        raise ValueError("no ```python code block found in response")
    code = max(blocks, key=len).strip() + "\n"
    if "def pack(" not in code:
        raise ValueError("code block does not define pack(n)")
    return code


def _format_stats(cand, verifier_list, use_soft, s_target):
    lines = [f"- outer side s = {cand.raw_s:.4f} (lower is better; best known ~ {s_target})"]
    if use_soft and verifier_list:
        lines.append("- structural measurements (each in [0,1], higher tends to mean smaller s):")
        for v, sc in zip(verifier_list, cand.soft):
            lines.append(f"    - {v['description']}: {sc:.3f}")
    return "\n".join(lines)


def _gate(source, n, cand_id, work_dir, timeout):
    """Write source, run evaluate.py as a subprocess, return (metrics, cand_dir)."""
    cand_dir = work_dir / cand_id
    cand_dir.mkdir(parents=True, exist_ok=True)
    prog = cand_dir / "source.py"
    prog.write_text(source)
    try:
        subprocess.run(
            [sys.executable, EVALUATE_PY, str(prog), str(n), str(cand_dir)],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        metrics = {"feasible": False, "raw_s": None, "reason": f"timeout>{timeout}s", "n": n}
        (cand_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return metrics, cand_dir
    mfile = cand_dir / "metrics.json"
    if not mfile.exists():
        return {"feasible": False, "raw_s": None, "reason": "evaluate.py wrote no metrics", "n": n}, cand_dir
    return json.loads(mfile.read_text()), cand_dir


def _make_candidate(cand_id, parent_id, step, source, metrics, cand_dir,
                    verifier_list, s_ref, s_target, w_raw, w_soft):
    if not metrics["feasible"]:
        return Candidate(cand_id, parent_id, step, source, False, None,
                         reason=metrics.get("reason", ""))
    geo = np.load(cand_dir / "geometry.npz")
    ev = verifiers.geometry_vars(geo["centers"], geo["angles"], float(geo["s"]))
    soft = verifiers.score_vector([v["verify_code"] for v in verifier_list], ev) if verifier_list else []
    score = scoring.combined_score(metrics["raw_s"], soft, s_ref, s_target, w_raw, w_soft)
    return Candidate(cand_id, parent_id, step, source, True, metrics["raw_s"], soft, score, "ok")


def _revise(llm, parent, n, cand_id, step, verifier_list, condition, use_soft, s_target,
            max_patch_attempts, timeout, mode):
    """Ask for a revision ("diff" or "rewrite" mode); on a malformed response, continue
    the same conversation with the error and retry (BES max_patch_attempts pattern).
    Raises when exhausted."""
    template = REVISE_PROMPT if mode == "diff" else REWRITE_PROMPT
    prompt = template.format(
        n=n, source=parent.source, timeout=timeout,
        stats=_format_stats(parent, verifier_list, use_soft, s_target),
    )
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(1, max_patch_attempts + 1):
        response = llm.call(
            messages,
            tags={"instance": f"n{n}", "condition": condition, "phase": "revise",
                  "step": step, "cand_id": cand_id, "parent_id": parent.cand_id,
                  "attempt": attempt, "mode": mode},
        )
        try:
            if mode == "diff":
                return diff.apply_diff(parent.source, response)
            return _extract_program(response)
        except ValueError as e:
            error = e
            messages = messages + [
                {"role": "assistant", "content": response},
                {"role": "user",
                 "content": f"The previous edit was not successful. Error: {e}\n\nTry again."},
            ]
    raise ValueError(f"no usable {mode} after {max_patch_attempts} attempts: {error}")


def search(n, llm, *, s_target, s_ref=None, num_seeds=4, num_steps=3, top_k=2,
           branching=2, k_verifiers=3, w_raw=1.0, w_soft=0.3, timeout=300,
           max_patch_attempts=3, work_dir=None, condition=None):
    """Run the search for instance n. Returns (best_candidate, history)."""
    use_soft = w_soft > 0
    condition = condition or ("search_soft" if use_soft else "search_raw")
    work_dir = Path(work_dir) if work_dir else (llm.run_dir / f"n{n}_{condition}")
    root_source = INITIAL_PY.read_text()

    root_metrics, root_dir = _gate(root_source, n, f"n{n}.root", work_dir, timeout)
    if s_ref is None:
        s_ref = root_metrics["raw_s"]

    verifier_list = []
    if use_soft:
        verifier_list = decompose(llm, n, s_ref, s_target, k_verifiers)
        (work_dir / "verifiers.json").write_text(json.dumps(verifier_list, indent=2))

    root = _make_candidate(f"n{n}.root", None, -1, root_source, root_metrics, root_dir,
                           verifier_list, s_ref, s_target, w_raw, w_soft)
    population = [root]
    history = [root]
    print(f"[n={n} {condition}] baseline s={s_ref:.4f}; {len(verifier_list)} verifiers")

    for step in range(num_steps + 1):  # step 0 = seeds; 1..num_steps = expansions
        if step == 0:
            parents = [root] * num_seeds
        else:
            top = sorted((c for c in population if c.feasible),
                         key=lambda c: c.score, reverse=True)[:top_k]
            parents = [p for p in top for _ in range(branching)]

        def thunk(i, parent):
            mode = "diff" if i % 2 == 0 else "rewrite"  # 50/50; one of each per parent
            def run():
                cid = f"n{n}.s{step}.{i}"
                try:
                    child_source = _revise(llm, parent, n, cid, step, verifier_list,
                                           condition, use_soft, s_target,
                                           max_patch_attempts, timeout, mode)
                except ValueError as e:  # revision attempts exhausted -> abandon this child
                    return Candidate(cid, parent.cand_id, step, parent.source, False, None,
                                     reason=str(e))
                metrics, cdir = _gate(child_source, n, cid, work_dir, timeout)
                return _make_candidate(cid, parent.cand_id, step, child_source, metrics,
                                       cdir, verifier_list, s_ref, s_target, w_raw, w_soft)
            return run

        children = run_parallel([thunk(i, p) for i, p in enumerate(parents)])
        population += children
        history += children
        n_feas = sum(c.feasible for c in children)
        best_s = min((c.raw_s for c in population if c.feasible), default=float("nan"))
        print(f"[n={n} {condition}] step {step}: {n_feas}/{len(children)} feasible; best s={best_s:.4f}")

    best = min((c for c in population if c.feasible), key=lambda c: c.raw_s)
    return best, history


if __name__ == "__main__":
    from llm import LLM

    llm = LLM()
    best, history = search(11, llm, s_target=3.931,
                           num_seeds=2, num_steps=1, top_k=1, branching=2)  # tiny smoke
    print(f"\nbest: {best.cand_id}  s={best.raw_s:.4f}  score={best.score:.4f}")
    print(f"artifacts: {llm.run_dir}")
