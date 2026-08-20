"""Self-Refine baseline (Madaan et al.): initial generation, then iterated
feedback -> refine as ONE growing conversation — every call sees the full
history of attempts and feedback (the paper's history mechanism, preventing
repeated mistakes). Stops early when the feedback declares the candidate
correct. Submits the last two distinct candidates, most recent first.

--steps N = feedback+refine iterations (default 5), so up to 1 + 2N calls
per test pair.

    uv run python arcagi2/baselines/self_refine.py --dev --steps 5
"""

from __future__ import annotations

from common import build_prompt, run_baseline_cli, sample_grid

import llm

FEEDBACK_TURN = "Re-derive the transformation from the training examples and give concrete, actionable feedback on the candidate output grid in your previous response. Do not write a corrected grid yourself. If the candidate is already correct, reply with exactly 'The candidate is correct.'"

REFINE_TURN = "Revise your candidate according to the feedback above. Give your final output grid at the end of your response."


async def solve_pair(task, pair_index, args):
    messages = [{"role": "user", "content": build_prompt(task, pair_index)}]
    chain = []

    r = await sample_grid(messages, args)
    chain.append({"type": "init", **{k: v for k, v in r.items() if k != "text"}})
    candidate = r["answer"]
    candidates = [candidate] if candidate is not None else []
    messages.append({"role": "assistant", "content": r.get("text") or r.get("raw") or ""})

    for _ in range(args.steps):
        if candidate is None:
            break
        messages.append({"role": "user", "content": FEEDBACK_TURN})
        try:
            feedback = await llm.generate(
                messages, model=args.model, provider=args.provider,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort) or ""
        except Exception as e:
            chain.append({"type": "feedback", "error": repr(e)})
            break
        chain.append({"type": "feedback", "text": feedback})
        messages.append({"role": "assistant", "content": feedback})
        if "candidate is correct" in feedback.lower():
            break
        messages.append({"role": "user", "content": REFINE_TURN})
        r = await sample_grid(messages, args)
        chain.append({"type": "refine", **{k: v for k, v in r.items() if k != "text"}})
        messages.append({"role": "assistant", "content": r.get("text") or r.get("raw") or ""})
        if r["answer"] is not None:
            candidate = r["answer"]
            candidates.append(candidate)

    distinct = []
    for c in reversed(candidates):
        if c not in distinct:
            distinct.append(c)
        if len(distinct) == 2:
            break
    return distinct, chain


if __name__ == "__main__":
    run_baseline_cli(
        "self_refine", solve_pair,
        lambda ap: ap.add_argument("--steps", type=int, default=5,
                                   help="feedback+refine iterations after the "
                                        "initial attempt (up to 1+2N calls)"))
