"""Best-of-N baseline: N independent revisions of initial.py, no iteration, no
verifiers. Exactly search() with one seed round -- the compute-matched control
for "does verifier-guided iterative search help?"."""

from search import search


def best_of_n(n, llm, *, s_target, N=16, s_ref=None, timeout=30, work_dir=None):
    return search(
        n, llm, s_target=s_target, s_ref=s_ref,
        num_seeds=N, num_steps=0, w_soft=0.0,
        condition="best_of_n", timeout=timeout, work_dir=work_dir,
    )


if __name__ == "__main__":
    from llm import LLM

    llm = LLM()
    best, history = best_of_n(11, llm, s_target=3.931, N=3)
    print(f"\nbest s={best.raw_s:.4f} ({sum(c.feasible for c in history)}/{len(history)} feasible)")
