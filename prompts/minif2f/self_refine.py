"""Self-critique prompt for the miniF2F self-refine baseline."""


def self_critique_prompt(
    informal_statement: str,
    formal_statement: str,
    proof_body: str,
) -> str:
    return f"""You are a Lean 4 proof verifier. Carefully check whether this proof is correct WITHOUT running a compiler.

Informal statement:
{informal_statement}

Formal statement:
```lean
{formal_statement}
```

Proof body:
```lean
{proof_body}
```

Check the following:
1. Each tactic is valid Lean 4 / Mathlib syntax with correct arguments.
2. No `sorry` or `admit` is used.
3. Hypotheses referenced actually exist in the theorem statement.
4. Types are consistent (ℕ vs ℤ vs ℝ, coercions, etc.).
5. The proof strategy logically closes all goals.

Output: If you find errors, list them inside <OUTPUT> tags. If the proof is correct, output empty <OUTPUT></OUTPUT> tags.

<OUTPUT>
`ring` cannot close a goal with natural number subtraction; use `omega` instead
Hypothesis `h₂` does not exist in the theorem statement
</OUTPUT>"""
