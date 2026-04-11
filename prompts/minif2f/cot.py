"""Chain-of-Thought prompt for the miniF2F generator agent."""


def cot_prompt(
    informal_statement: str,
    formal_statement: str,
    header: str,
) -> str:
    return f"""You are a Lean 4 theorem prover. Produce a tactic proof for the following theorem.

Informal statement:
{informal_statement}

Lean 4 formal statement (your proof follows the `by`):
```lean
{header}

{formal_statement}
```

Let's think step by step. First analyze what the theorem is asking, identify the key mathematical relationships, then determine which Lean 4 / Mathlib tactics are appropriate. Consider:
1. What is the goal type? (equality, inequality, existence, etc.)
2. What hypotheses are available?
3. Which tactics handle this goal type? (norm_num, ring, linarith, omega, simp, etc.)
4. Do we need case splits, induction, or intermediate lemmas?

Do NOT use `sorry` or `admit`.

Output: Wrap your proof body (just the tactics, not the theorem statement) in <OUTPUT> tags:

<OUTPUT>
  simp [h₀, h₁]
  ring
</OUTPUT>"""
