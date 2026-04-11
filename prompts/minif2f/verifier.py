"""Prompt templates for the miniF2F verifier agents.

Three orthogonal axes targeting the most common LLM failure modes:
  1. math_soundness  — Is the proof strategy mathematically correct?
  2. lemma_existence — Do all referenced Mathlib lemmas/tactics actually exist?
  3. simp_feasibility — Will simp/simpa/norm_num actually close the claimed subgoals?

LLM-only verifiers: NO compiler errors are passed. Pure verbal gradient reasoning.
"""


def math_soundness_prompt(
    proof_body: str,
    informal_statement: str,
    formal_statement: str,
) -> str:
    return f"""You are a mathematician reviewing a Lean 4 proof for correctness. Ignore Lean syntax entirely — focus only on whether the mathematical reasoning is sound.

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
1. Are all intermediate claims (have statements) mathematically true? Verify each one by hand.
2. If the proof claims a closed-form formula or identity, verify it against the given hypotheses. Substitute concrete values to test.
3. Does the proof strategy actually lead to the conclusion? Trace the logical chain from hypotheses to goal.
4. Are there any steps where the proof "gives up" and hopes a tactic will magically work (e.g., claiming `simpa` will evaluate a complex expression without justification)?
5. Does the proof handle all edge cases (e.g., division by zero, base cases in induction)?

Be especially suspicious of:
- Claimed closed forms that contradict given initial conditions
- Steps that skip non-trivial mathematical reasoning with vague comments
- Proofs that pick a disjunct (e.g., `left` or `Or.inl`) without justifying the choice

Output: If you find mathematical errors, list them inside <OUTPUT> tags with specific details about what is wrong and what the correct reasoning should be. If the math is sound, output empty <OUTPUT></OUTPUT> tags.

Example format:
<OUTPUT>
The proof claims `a n = n` for all n >= 1, but a(2) = 3/7 by hypothesis, not 2. The closed form is incorrect.
Step `have hx : x = 3` is unjustified — from x^2 = 9 and x > 0, we get x = 3, but the proof does not establish x > 0 first.
</OUTPUT>"""


def lemma_existence_prompt(
    proof_body: str,
    informal_statement: str,
    formal_statement: str,
) -> str:
    return f"""You are a Mathlib expert reviewing a Lean 4 proof. Check whether all referenced lemma names, theorem names, and tactic invocations actually exist in Lean 4 / Mathlib.

Formal statement:
```lean
{formal_statement}
```

Proof body:
```lean
{proof_body}
```

Check the following:
1. Every named lemma or theorem referenced (e.g., in `exact`, `apply`, `simp [...]`, `rw [...]`) — does it exist in Mathlib with that exact name?
2. Every tactic used — is it a real Lean 4 or Mathlib tactic with correct syntax?
3. Lemma signatures — if a lemma is applied with specific arguments, are the argument types compatible with the actual lemma?
4. No `sorry` or `admit` anywhere.

Common hallucination patterns to watch for:
- Lemma names that look plausible but don't exist (e.g., `Finset.prod_Icc_succ_top`, `sq_eq_sq_iff_eq_or_eq_neg`, `neg_neg_of_pos`)
- Using Lean 3 syntax in Lean 4 (e.g., `begin...end`, `rw ← ` vs `rw [← ...]`)
- Tactics with wrong arity or argument order
- Referencing hypotheses that don't exist in scope

Only flag names you are confident are wrong. If you are unsure whether a name exists, do not flag it.

Output: If you find non-existent lemmas or tactics, list them inside <OUTPUT> tags. If everything looks valid, output empty <OUTPUT></OUTPUT> tags.

Example format:
<OUTPUT>
`Finset.prod_Icc_succ_top` does not exist in Mathlib. To peel the last element from a Finset.Icc product, use `Finset.prod_Icc_succ_top` is not valid — consider `Finset.Icc_insert_right` with `Finset.prod_insert`.
`mul_inv_rev₀` does not exist. The correct name is `mul_inv_rev`.
</OUTPUT>"""


def simp_feasibility_prompt(
    proof_body: str,
    informal_statement: str,
    formal_statement: str,
) -> str:
    return f"""You are a Lean 4 tactic expert. Review each use of `simp`, `simpa`, `norm_num`, `norm_cast`, `ring`, `linarith`, `nlinarith`, `omega`, and `decide` in the proof below. For each invocation, assess whether it can plausibly close the subgoal.

Formal statement:
```lean
{formal_statement}
```

Proof body:
```lean
{proof_body}
```

For each automation tactic call, check:

1. **`simp` / `simpa`**: These simplify using a fixed set of lemmas. They CANNOT:
   - Evaluate complex symbolic expressions (e.g., sums over Finset, products with 500 terms)
   - Prove non-trivial algebraic identities without appropriate simp lemmas passed as arguments
   - Substitute and simplify unless the rewrite target is in simp normal form
   Flag any `simp`/`simpa` that is expected to do heavy computation or prove non-obvious goals.

2. **`norm_num`**: Evaluates concrete numerical expressions. It CANNOT:
   - Handle symbolic variables
   - Prove divisibility of large numbers without extensions
   - Evaluate expressions involving `Real.sqrt`, `Complex.I`, or transcendental functions

3. **`ring`**: Proves polynomial identities. It CANNOT:
   - Handle division, `Real.sqrt`, or non-polynomial operations
   - Work across different types (e.g., mixing ℕ and ℤ without casts)

4. **`linarith` / `nlinarith`**: Prove linear/nonlinear arithmetic. They CANNOT:
   - Prove goals requiring case splits
   - Handle terms like `Real.sqrt` unless given appropriate auxiliary hypotheses

5. **`omega`**: Proves linear arithmetic over ℤ/ℕ. It CANNOT handle ℝ or ℚ.

6. **`decide`**: Evaluates decidable propositions by computation. It CANNOT handle universally quantified statements or large enumerations efficiently.

Output: If you find infeasible tactic applications, list them inside <OUTPUT> tags explaining why the tactic cannot close that subgoal and suggest an alternative. If all tactic uses look feasible, output empty <OUTPUT></OUTPUT> tags.

Example format:
<OUTPUT>
Line `simpa [h₁]` expects simp to evaluate ∑ k in Finset.Icc 1 12, z^(k^2) for complex z, but simp cannot perform this symbolic computation. This requires explicit calculation or `norm_num` with the right extensions.
Line `ring` is applied to a goal involving `Real.sqrt 2`, which is not a polynomial operation. Use `field_simp` followed by `ring` instead, or provide `Real.sq_sqrt` as a rewrite first.
</OUTPUT>"""
