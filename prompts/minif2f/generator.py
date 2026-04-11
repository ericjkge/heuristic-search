"""Prompt templates for the miniF2F generator agent."""


def initial_prompt(
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

Write a complete tactic proof body that follows `by`. Use Mathlib tactics (norm_num, ring, linarith, omega, simp, etc.) as appropriate. Do NOT use `sorry` or `admit`.

Output: Wrap your proof body (just the tactics, not the theorem statement) in <OUTPUT> tags:

<OUTPUT>
  simp [h₀, h₁]
  ring
</OUTPUT>"""


def revision_prompt(
    informal_statement: str,
    formal_statement: str,
    header: str,
    previous_proof: str,
    feedback: str,
) -> str:
    return f"""You are a Lean 4 theorem prover. Fix your previous proof based on verifier feedback.

Informal statement:
{informal_statement}

Lean 4 formal statement (your proof follows the `by`):
```lean
{header}

{formal_statement}
```

Your previous proof body:
```lean
{previous_proof}
```

Verifier feedback:
{feedback}

Write a corrected tactic proof body. Do NOT use `sorry` or `admit`.

Output: Wrap your proof body in <OUTPUT> tags:

<OUTPUT>
  simp [h₀, h₁]
  ring
</OUTPUT>"""
