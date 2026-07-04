"""Budget-matched baselines. The prof's rule: unbounded best-of-N degenerates into
enumeration, so every LLM condition gets the SAME total candidate budget and
results are compared as best-s-vs-calls curves, not just endpoints.

  best_of_n  -- N independent rewrites of the seed; population, no feedback loop
  hillclimb  -- one lineage, always revise the current best; feedback, no population
  multistart -- NO LLM: random configurations + scipy local refinement; the
                "dumb numerics" floor that any LLM condition must beat
"""

import math
import random
from pathlib import Path

import numpy as np

import search as searchmod
from concurrency import run_parallel
from evaluate import validate
from initial import enclosing_side, pentagon_vertices, repair
from search import Candidate, _gate, _make_candidate, _revise, INITIAL_PY


def best_of_n(n, llm, *, s_target, N, s_ref=None, timeout=300,
              max_patch_attempts=3, work_dir=None):
    """N independent single-shot rewrites of initial.py. Returns (best, history)."""
    condition = "best_of_n"
    work_dir = Path(work_dir) if work_dir else (llm.run_dir / f"n{n}_{condition}")
    root_source = INITIAL_PY.read_text()
    root_metrics, root_dir = _gate(root_source, n, f"n{n}.root", work_dir, timeout)
    if not root_metrics["feasible"]:
        raise RuntimeError(f"seed program infeasible: {root_metrics.get('reason')}")
    if s_ref is None:
        s_ref = root_metrics["raw_s"]
    root = _make_candidate(f"n{n}.root", None, -1, root_source, root_metrics, root_dir,
                           s_ref, s_target, 1.0, 0.0)

    def thunk(i):
        def run():
            cid = f"n{n}.bon.{i}"
            try:
                src = _revise(llm, root, n, cid, 0, condition, False, s_target,
                              max_patch_attempts, timeout, "rewrite")
            except ValueError as e:
                return Candidate(cid, root.cand_id, 0, root.source, False, None, reason=str(e))
            metrics, cdir = _gate(src, n, cid, work_dir, timeout)
            return _make_candidate(cid, root.cand_id, 0, src, metrics, cdir,
                                   s_ref, s_target, 1.0, 0.0)
        return run

    history = [root] + run_parallel([thunk(i) for i in range(N)])
    feasible = [c for c in history if c.feasible]
    best = min(feasible, key=lambda c: c.raw_s) if feasible else None
    return best, history


def hillclimb(n, llm, *, s_target, iters, s_ref=None, timeout=300,
              max_patch_attempts=3, work_dir=None):
    """Sequential: always revise the best-so-far; keep the child iff s improves."""
    condition = "hillclimb"
    work_dir = Path(work_dir) if work_dir else (llm.run_dir / f"n{n}_{condition}")
    root_source = INITIAL_PY.read_text()
    root_metrics, root_dir = _gate(root_source, n, f"n{n}.root", work_dir, timeout)
    if not root_metrics["feasible"]:
        raise RuntimeError(f"seed program infeasible: {root_metrics.get('reason')}")
    if s_ref is None:
        s_ref = root_metrics["raw_s"]
    best = _make_candidate(f"n{n}.root", None, -1, root_source, root_metrics, root_dir,
                           s_ref, s_target, 1.0, 0.0)
    history = [best]

    for it in range(iters):
        cid = f"n{n}.hc.{it}"
        mode = "diff" if it % 2 == 0 else "rewrite"
        try:
            src = _revise(llm, best, n, cid, it, condition, False, s_target,
                          max_patch_attempts, timeout, mode)
        except ValueError as e:
            history.append(Candidate(cid, best.cand_id, it, best.source, False, None,
                                     reason=str(e)))
            continue
        metrics, cdir = _gate(src, n, cid, work_dir, timeout)
        child = _make_candidate(cid, best.cand_id, it, src, metrics, cdir,
                                s_ref, s_target, 1.0, 0.0)
        history.append(child)
        if child.feasible and child.raw_s < best.raw_s:
            best = child
    return best, history


def multistart(n, *, restarts=200, seed=0, iters=300):
    """NO LLM: random configurations, dilation repair, then greedy local moves
    (random center nudges + rotations, accept iff honest enclosing side shrinks).
    Returns (best_s, centers, angles) validated by the hard checker."""
    rng = random.Random(seed)
    best_s, best_geo = float("inf"), None

    spread = 1.1 * math.sqrt(n)
    for _ in range(restarts):
        centers = [(rng.uniform(-spread, spread), rng.uniform(-spread, spread))
                   for _ in range(n)]
        angles = [rng.uniform(0, 2 * math.pi / 5) for _ in range(n)]
        centers = repair(centers, angles)
        s = enclosing_side(centers, angles)
        ok, _ = validate(centers, angles, s, n)
        if not ok:
            continue
        for _ in range(iters):  # greedy shrink: nudge one pentagon toward origin
            i = rng.randrange(n)
            cx, cy = centers[i]
            step = 0.05 * rng.random()
            cand = centers.copy()
            cand[i] = (cx - step * math.copysign(1, cx), cy - step * math.copysign(1, cy))
            cand_ang = angles.copy()
            if rng.random() < 0.3:
                cand_ang[i] = (angles[i] + rng.gauss(0, 0.2)) % (2 * math.pi)
            s2 = enclosing_side(cand, cand_ang)
            ok, _ = validate(cand, cand_ang, s2, n)
            if ok and s2 < s:
                centers, angles, s = cand, cand_ang, s2
        if s < best_s:
            best_s, best_geo = s, (centers, angles)
    return best_s, best_geo


if __name__ == "__main__":  # offline smoke for the no-LLM baseline only
    s, _ = multistart(6, restarts=20, iters=100)
    print(f"multistart n=6: s={s:.4f}")
