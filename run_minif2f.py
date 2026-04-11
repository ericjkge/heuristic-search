"""Entry point: Generator-Verifier miniF2F (Lean 4 theorem proving) solver.

Usage: python3 run_minif2f.py [--baseline] [--max_rounds 5] [--dataset data/minif2f/problems_valid.json]
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

from utils.llm import HarvardGPTLLM
from utils.logging import setup_logging, get_logger
from tasks.minif2f import parse_proof, check_proof
from agents.generator import GeneratorAgent
from agents.verifier import VerifierAgent
from agents.debate import DebateAgent
from prompts.minif2f.generator import initial_prompt, revision_prompt
from prompts.minif2f.cot import cot_prompt
from prompts.minif2f.self_refine import self_critique_prompt
from prompts.minif2f.debate import propose_prompt, critique_prompt
from prompts.minif2f.verifier import (
    math_soundness_prompt,
    lemma_existence_prompt,
    simp_feasibility_prompt,
)

logger = get_logger(__name__)


def solve_baseline(
    informal: str,
    formal: str,
    header: str,
    gen_llm: HarvardGPTLLM,
) -> dict:
    """Baseline: single generator call."""
    generator = GeneratorAgent(gen_llm)
    prompt = initial_prompt(informal, formal, header)
    _, raw = generator.generate(prompt)

    return {
        "proof": parse_proof(raw),
        "rounds": 1,
        "total_tokens": generator.total_tokens,
    }


def solve_cot(
    informal: str,
    formal: str,
    header: str,
    gen_llm: HarvardGPTLLM,
) -> dict:
    """CoT baseline: single call with chain-of-thought prompting."""
    generator = GeneratorAgent(gen_llm)
    prompt = cot_prompt(informal, formal, header)
    _, raw = generator.generate(prompt)

    return {
        "proof": parse_proof(raw),
        "rounds": 1,
        "total_tokens": generator.total_tokens,
    }


def solve_self_consistency(
    informal: str,
    formal: str,
    header: str,
    sc_llms: list[HarvardGPTLLM],
) -> dict:
    """Self-Consistency: K=3 independent proofs, return all for compilation check."""
    prompt = initial_prompt(informal, formal, header)

    proofs: list[Optional[str]] = []
    total_tokens = 0
    for llm in sc_llms:
        gen = GeneratorAgent(llm)
        _, raw = gen.generate(prompt)
        total_tokens += gen.total_tokens
        proofs.append(parse_proof(raw))

    return {
        "proofs": proofs,
        "rounds": 1,
        "total_tokens": total_tokens,
    }


def solve_self_refine(
    informal: str,
    formal: str,
    header: str,
    max_rounds: int,
    gen_llm: HarvardGPTLLM,
) -> dict:
    """Self-Refine: generate, self-critique, revise (single LLM)."""
    generator = GeneratorAgent(gen_llm)

    # Initial generation
    prompt = initial_prompt(informal, formal, header)
    _, raw = generator.generate(prompt)
    rounds_used = 1

    for round_num in range(1, max_rounds + 1):
        proof = parse_proof(raw)

        if proof is None:
            feedback = (
                "ERROR: Could not parse your proof. Output tactic proof body "
                "inside <OUTPUT> tags."
            )
        else:
            # Self-critique
            critique_resp = gen_llm.generate(
                self_critique_prompt(informal, formal, proof)
            )
            generator.total_tokens += gen_llm.last_tokens

            matches = re.findall(r"<OUTPUT>(.*?)</OUTPUT>", critique_resp, re.DOTALL)
            feedback = matches[-1].strip() if matches else ""
            if not feedback:
                # Verifier found no issues — stop refining
                break

        # Revise
        if round_num < max_rounds:
            _, raw = generator.generate(
                revision_prompt(informal, formal, header, proof or raw, feedback)
            )
        rounds_used = round_num

    return {
        "proof": parse_proof(raw),
        "rounds": rounds_used,
        "total_tokens": generator.total_tokens,
    }


def solve_debate(
    informal: str,
    formal: str,
    header: str,
    debate_llms: list[HarvardGPTLLM],
    critique_rounds: int = 2,
) -> dict:
    """Debate baseline: 3 agents propose, critique. Return all final proofs."""
    agents = [
        DebateAgent(f"agent_{i}", debate_llms[i], parse_fn=parse_proof)
        for i in range(3)
    ]

    # 1. Propose
    proofs: list[Optional[str]] = []
    raws: list[str] = []
    for agent in agents:
        proof, raw = agent.propose(propose_prompt(informal, formal, header))
        proofs.append(proof)
        raws.append(raw)

    # 2. Critique rounds
    for cr in range(critique_rounds):
        new_proofs: list[Optional[str]] = []
        new_raws: list[str] = []
        for i, agent in enumerate(agents):
            other_raws = [raws[j] for j in range(3) if j != i]
            prompt = critique_prompt(informal, formal, header, raws[i], other_raws)
            proof, raw = agent.critique(prompt)
            new_proofs.append(proof)
            new_raws.append(raw)
        proofs = new_proofs
        raws = new_raws

    total_tokens = sum(a.total_tokens for a in agents)
    rounds_used = 1 + critique_rounds

    return {
        "proofs": proofs,
        "rounds": rounds_used,
        "total_tokens": total_tokens,
    }


def solve_puzzle(
    informal: str,
    formal: str,
    header: str,
    max_rounds: int,
    gen_llm: HarvardGPTLLM,
    ver_llms: list[HarvardGPTLLM],
) -> dict:
    """Run generator-verifier loop on a single theorem (no compilation check)."""
    generator = GeneratorAgent(gen_llm)

    # Build 3 verifiers with closure-based prompt_fn
    verifiers = [
        VerifierAgent(
            "math_soundness",
            ver_llms[0],
            lambda proof, _inf=informal, _frm=formal: math_soundness_prompt(proof, _inf, _frm),
        ),
        VerifierAgent(
            "lemma_existence",
            ver_llms[1],
            lambda proof, _inf=informal, _frm=formal: lemma_existence_prompt(proof, _inf, _frm),
        ),
        VerifierAgent(
            "simp_feasibility",
            ver_llms[2],
            lambda proof, _inf=informal, _frm=formal: simp_feasibility_prompt(proof, _inf, _frm),
        ),
    ]

    # Initial generation
    prompt = initial_prompt(informal, formal, header)
    _, raw = generator.generate(prompt)
    rounds_used = 1

    for round_num in range(1, max_rounds + 1):
        proof = parse_proof(raw)

        if proof is None:
            feedback = (
                "ERROR: Could not parse your proof. Output tactic proof body "
                "inside <OUTPUT> tags."
            )
            _, raw = generator.generate(
                revision_prompt(informal, formal, header, raw, feedback)
            )
            rounds_used = round_num + 1 if round_num < max_rounds else rounds_used
            continue

        # LLM verifiers — NO compiler errors passed
        feedback_parts: list[str] = []
        for verifier in verifiers:
            errors = verifier.verify(proof)
            if errors is not None:
                feedback_parts.append(f"[{verifier.name} verifier]:\n{errors}")

        if not feedback_parts:
            # All verifiers passed — stop loop
            break

        feedback = "\n\n".join(feedback_parts)
        if round_num < max_rounds:
            _, raw = generator.generate(
                revision_prompt(informal, formal, header, proof, feedback)
            )
        rounds_used = round_num

    total_tokens = generator.total_tokens + sum(v.total_tokens for v in verifiers)

    return {
        "proof": parse_proof(raw),
        "rounds": rounds_used,
        "total_tokens": total_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generator-Verifier miniF2F Solver")
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--baseline", action="store_true", help="Single-shot baseline")
    parser.add_argument("--cot", action="store_true", help="Chain-of-thought baseline")
    parser.add_argument("--self_consistency", action="store_true", help="Self-consistency (K=3) baseline")
    parser.add_argument("--self_refine", action="store_true", help="Self-refine baseline")
    parser.add_argument("--debate", action="store_true", help="Debate baseline (3 agents debate)")
    parser.add_argument("--log", action="store_true", help="Enable file logging")
    parser.add_argument("--dataset", type=str, default="data/minif2f/problems_test.json")
    parser.add_argument("--problem_names", type=str, nargs="+", help="Specific problem names to run")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for JSONL results")
    args = parser.parse_args()

    if args.baseline:
        mode = "baseline"
    elif args.cot:
        mode = "cot"
    elif args.self_consistency:
        mode = "self-consistency"
    elif args.self_refine:
        mode = "self-refine"
    elif args.debate:
        mode = "debate"
    else:
        mode = "gen-ver"
    suffix = f"minif2f_{mode}"

    with open(args.dataset) as f:
        dataset = json.load(f)

    if args.problem_names is not None:
        dataset = [e for e in dataset if e["name"] in args.problem_names]

    if args.log:
        pid = args.problem_names[0] if args.problem_names and len(args.problem_names) == 1 else None
        setup_logging(log_dir="logs", suffix=suffix, puzzle_id=pid)

    # JSONL results file
    dataset_stem = Path(args.dataset).stem  # e.g. "problems_valid"
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"minif2f_{mode}_{dataset_stem}.jsonl"

    print(f"Mode: {mode}")
    print(f"Loaded {len(dataset)} problems from {args.dataset}")
    if mode in ("gen-ver", "self-refine"):
        print(f"Max rounds: {args.max_rounds}")
    print()

    gen_llm = HarvardGPTLLM()
    if args.debate:
        debate_llms = [HarvardGPTLLM() for _ in range(3)]
        sc_llms: list[HarvardGPTLLM] = []
        ver_llms: list[HarvardGPTLLM] = []
    elif args.self_consistency:
        debate_llms = []
        sc_llms = [HarvardGPTLLM() for _ in range(3)]
        ver_llms = []
    elif mode == "gen-ver":
        debate_llms = []
        sc_llms = []
        ver_llms = [HarvardGPTLLM() for _ in range(3)]
    else:
        debate_llms = []
        sc_llms = []
        ver_llms = []

    # Phase 1: Run all LLM solvers (no compilation)
    raw_results = []
    start_time = time.time()

    for entry in dataset:
        name = entry["name"]
        informal = entry["informal_statement"]
        formal = entry["formal_statement"]
        header = entry["header"]
        print(f"Problem {name}...", end=" ", flush=True)

        t0 = time.time()
        if args.baseline:
            result = solve_baseline(informal, formal, header, gen_llm)
        elif args.cot:
            result = solve_cot(informal, formal, header, gen_llm)
        elif args.self_consistency:
            result = solve_self_consistency(informal, formal, header, sc_llms)
        elif args.self_refine:
            result = solve_self_refine(informal, formal, header, args.max_rounds, gen_llm)
        elif args.debate:
            result = solve_debate(informal, formal, header, debate_llms)
        else:
            result = solve_puzzle(
                informal, formal, header, args.max_rounds, gen_llm, ver_llms
            )

        result["name"] = name
        result["header"] = header
        result["formal_statement"] = formal
        result["llm_elapsed"] = round(time.time() - t0, 2)
        raw_results.append(result)

        print(f"done in {result['rounds']} rounds ({result['total_tokens']} tokens, {result['llm_elapsed']}s)")

    llm_elapsed = time.time() - start_time
    print(f"\nLLM phase complete: {llm_elapsed:.1f}s")

    # Phase 2: Check all proofs with Lean compiler
    print("\nChecking proofs with Lean compiler...")
    results = []

    for r in raw_results:
        name = r["name"]
        header = r["header"]
        formal = r["formal_statement"]

        # Multi-proof modes (self-consistency, debate): any that compiles wins
        proofs = r.get("proofs", [r.get("proof")])

        solved = False
        for proof in proofs:
            if proof is not None and check_proof(header, formal, proof):
                solved = True
                break

        result = {
            "name": name,
            "solved": solved,
            "rounds": r["rounds"],
            "total_tokens": r["total_tokens"],
            "elapsed": r["llm_elapsed"],
        }
        results.append(result)

        with open(results_file, "a") as rf:
            rf.write(json.dumps(result) + "\n")

        status = "SOLVED" if solved else "FAILED"
        print(f"  {name}: {status}")

    # Summary
    total_elapsed = time.time() - start_time
    solved_count = sum(1 for r in results if r["solved"])
    avg_rounds = sum(r["rounds"] for r in results) / len(results) if results else 0
    total_tokens = sum(r["total_tokens"] for r in results)

    print(f"\n{'='*40}")
    print(f"Solve rate: {solved_count}/{len(results)} ({100*solved_count/len(results):.0f}%)")
    print(f"Avg rounds: {avg_rounds:.1f}")
    print(f"Total tokens: {total_tokens}")
    print(f"Elapsed: {total_elapsed:.1f}s (LLM: {llm_elapsed:.1f}s)")


if __name__ == "__main__":
    main()
