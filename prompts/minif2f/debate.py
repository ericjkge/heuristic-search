"""Debate (MAD) prompts for the miniF2F task."""


def propose_prompt(
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

Write a complete tactic proof body. Do NOT use `sorry` or `admit`.

Output: Wrap your proof body in <OUTPUT> tags:

<OUTPUT>
  simp [h₀, h₁]
  ring
</OUTPUT>"""


def critique_prompt(
    informal_statement: str,
    formal_statement: str,
    header: str,
    own_proof: str,
    other_proofs: list[str],
) -> str:
    others_text = "\n\n".join(
        f"--- Proof {i+1} ---\n{p}" for i, p in enumerate(other_proofs)
    )
    return f"""You are a Lean 4 theorem prover engaged in a debate. You have proposed a proof and have seen other agents' proofs. Revise your answer if you find errors.

Informal statement:
{informal_statement}

Lean 4 formal statement (your proof follows the `by`):
```lean
{header}

{formal_statement}
```

Your previous proof:
```lean
{own_proof}
```

Other agents' proofs:
{others_text}

Compare all proofs carefully. Check tactic validity, type correctness, and logical soundness. Produce your best revised proof.

Output: Wrap your proof body in <OUTPUT> tags:

<OUTPUT>
  simp [h₀, h₁]
  ring
</OUTPUT>"""
