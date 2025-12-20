from dotenv import load_dotenv
import tot_prompts
from models import GeminiLLM, BaseLLM
import re

# Load environment variables
load_dotenv()

class TreeOfThoughts:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def solve(self, problem: str, k: int = 3, b: int = 5, d: int = 3, log_file=None):
        """
        Solves the problem using BFS.

        k: Number of proposed thoughts per state
        b: Branching factor (top states to keep)
        d: Max depth (steps)
        log_file: Open file handle for logging (optional)
        """
        current_states = [""]  # e.g. ["2 + 6 = 8 (left: 2 8 8)\n8 + 8 = 16 (left: 2 16)", "2 + 6 = 8 (left: 2 8 8)\n8 - 2 = 6 (left: 6 8)"]
        total_tokens = 0
        
        for step in range(d):
            print(f"Step {step+1}/{d}, current states: {len(current_states)}")
            
            # 1. Generate k new thoughts per state
            candidates = []
            for state in current_states:
                proposals, tokens = self._propose(problem, state, k)
                total_tokens += tokens
                for proposal in proposals:
                    # Append new thought to existing state
                    new_state = f"{state}\n{proposal}" if state else proposal
                    candidates.append(new_state)

            # 2. Evaluate each candidate state
            scored_candidates = []
            for candidate in candidates:
                score, tokens = self._evaluate(candidate)
                total_tokens += tokens
                scored_candidates.append((score, candidate))

            # 3. Select top "b" states
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            selected = scored_candidates[:b]
            current_states = [state for score, state in selected]
            
            # Log top thoughts
            if log_file:
                log_file.write(f"\n--- Step {step+1} Top Thoughts ---\n")
                for i, (score, state) in enumerate(selected):
                    log_file.write(f"Rank {i+1} (Score: {score}):\n{state}\n{'-'*20}\n")
                    print(f"Rank {i+1} (Score: {score}):\n{state}\n{'-'*20}\n")

            if selected:
                print(f"Top score: {selected[0][0]}")

        print(f"Total tokens used: {total_tokens}")
        return current_states[0] if current_states else "No solution found"

    def _extract_remaining(self, state: str) -> str:
        last_step = state.strip().split('\n')[-1] if state else ""
        match = re.search(r'\(left:\s*([^\)]+)\)', last_step)
        return match.group(1).strip() if match else ""

    def _propose(self, problem: str, state: str, k: int) -> tuple[list[str], int]:
        if state:
            remaining = self._extract_remaining(state)
        else:
            remaining = problem
        
        prompt = tot_prompts.propose_prompt.format(input=remaining)
        response_text, token_count = self.llm.generate(prompt)
        
        proposals = [line.strip() for line in response_text.split('\n') if line.strip()]
        return proposals, token_count

    def _evaluate(self, candidate: str) -> tuple[float, int]:
        remaining = self._extract_remaining(candidate)
        if not remaining:
            return 0.001, 0
        
        prompt = tot_prompts.value_prompt.format(input=remaining)
        response_text, token_count = self.llm.generate(prompt)

        # Map sure/likely/impossible to scores (values from Princeton ToT)
        response_lower = response_text.lower()
        if 'sure' in response_lower:
            return 20, token_count
        elif 'likely' in response_lower:
            return 1, token_count
        else:
            return 0.001, token_count

if __name__ == "__main__":
    problem = "2 2 6 8"
    print(f"Solving: {problem}")
    
    with open("output.txt", "w") as f:
        f.write(f"Problem: {problem}\n")
        llm = GeminiLLM()
        tot = TreeOfThoughts(llm)
        best_path = tot.solve(problem, k=3, b=5, d=3, log_file=f)
        
        f.write("\n=== BEST PATH ===\n")
        f.write(best_path)
    
    print("\n=== BEST PATH ===")
    print(best_path)
