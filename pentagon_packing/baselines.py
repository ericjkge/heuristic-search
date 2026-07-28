"""Budget-matched baselines. The prof's rule: unbounded best-of-N degenerates into
enumeration, so every LLM condition gets the SAME total candidate budget and
results are compared as best-s-vs-calls curves, not just endpoints.

  best_of_n   -- N independent diff edits of the seed; population, no feedback loop
  self_refine -- one lineage, always revise the current best; feedback, no population

All revisions are diff-only (SEARCH/REPLACE), matching search.py.
"""

from pathlib import Path

from concurrency import run_parallel
from search import Candidate, _gate, _make_candidate, _revise, INITIAL_PY


def best_of_n(n, llm, *, s_target, N, timeout=300,
              max_patch_attempts=3, work_dir=None, condition="best_of_n"):
    """N independent single-shot diff edits of initial.py. Returns (best, history)."""
    work_dir = Path(work_dir) if work_dir else (llm.run_dir / f"n{n}_{condition}")
    root_source = INITIAL_PY.read_text()
    root_metrics, root_dir = _gate(root_source, n, f"n{n}.root", work_dir, timeout)
    if not root_metrics["feasible"]:
        raise RuntimeError(f"seed program infeasible: {root_metrics.get('reason')}")
    root = _make_candidate(f"n{n}.root", None, -1, root_source, root_metrics, root_dir)

    def thunk(i):
        def run():
            cid = f"n{n}.bon.{i}"
            try:
                src = _revise(llm, root, n, cid, 0, condition, False, s_target,
                              max_patch_attempts, timeout)
            except ValueError as e:
                return Candidate(cid, root.cand_id, 0, root.source, False, None, reason=str(e))
            metrics, cdir = _gate(src, n, cid, work_dir, timeout)
            return _make_candidate(cid, root.cand_id, 0, src, metrics, cdir)
        return run

    history = [root] + run_parallel([thunk(i) for i in range(N)])
    feasible = [c for c in history if c.feasible]
    best = min(feasible, key=lambda c: c.raw_s) if feasible else None
    return best, history


def self_refine(n, llm, *, s_target, iters, timeout=300,
                max_patch_attempts=3, work_dir=None, condition="self_refine"):
    """Sequential: always revise the best-so-far; keep the child iff s improves."""
    work_dir = Path(work_dir) if work_dir else (llm.run_dir / f"n{n}_{condition}")
    root_source = INITIAL_PY.read_text()
    root_metrics, root_dir = _gate(root_source, n, f"n{n}.root", work_dir, timeout)
    if not root_metrics["feasible"]:
        raise RuntimeError(f"seed program infeasible: {root_metrics.get('reason')}")
    best = _make_candidate(f"n{n}.root", None, -1, root_source, root_metrics, root_dir)
    history = [best]

    for it in range(iters):
        cid = f"n{n}.sr.{it}"
        try:
            src = _revise(llm, best, n, cid, it, condition, False, s_target,
                          max_patch_attempts, timeout)
        except ValueError as e:
            history.append(Candidate(cid, best.cand_id, it, best.source, False, None,
                                     reason=str(e)))
            continue
        metrics, cdir = _gate(src, n, cid, work_dir, timeout)
        child = _make_candidate(cid, best.cand_id, it, src, metrics, cdir)
        history.append(child)
        if child.feasible and child.raw_s < best.raw_s:
            best = child
    return best, history
