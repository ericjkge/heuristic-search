from dotenv import load_dotenv
import tot_prompts
from models import KimiLLM, BaseLLM
import re

# Load environment variables
load_dotenv()

class TreeOfThoughts:
    def __init__(self, llm: BaseLLM, debug_log=None):
        self.llm = llm
        self.debug_log = debug_log
        self.call_count = 0

    def _log_llm_call(self, call_type: str, prompt: str, response: str):
        """Log LLM call details to debug log."""
        if not self.debug_log:
            return
        self.call_count += 1
        self.debug_log.write(f"\n{'='*60}\n")
        self.debug_log.write(f"[CALL #{self.call_count}] {call_type}\n")
        self.debug_log.write(f"{'='*60}\n")
        self.debug_log.write(f"\n--- USER PROMPT ---\n{prompt}\n")
        self.debug_log.write(f"\n--- RESPONSE ---\n{response}\n")
        self.debug_log.flush()

    def solve(self, problem: str, b: int = 5, d: int = 3, log_file=None):
        """
        Solves the problem using BFS.

        b: Beam width (top states to keep per step)
        d: Max depth (steps)
        log_file: Open file handle for logging (optional)
        """
        current_states = [""]  # e.g. ["2 + 6 = 8 (left: 2 8 8)\n8 + 8 = 16 (left: 2 16)", "2 + 6 = 8 (left: 2 8 8)\n8 - 2 = 6 (left: 6 8)"]
        total_tokens = 0
        
        for step in range(d):
            print(f"Step {step+1}/{d}, current states: {len(current_states)}")
            
            # 1. Generate new thoughts per state
            candidates = []
            for state in current_states:
                proposals, tokens = self._propose(problem, state)
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

    def _propose(self, problem: str, state: str) -> tuple[list[str], int]:
        if state:
            remaining = self._extract_remaining(state)
        else:
            remaining = problem
        
        prompt = tot_prompts.propose_prompt.format(input=remaining)
        response_text, token_count = self.llm.generate(prompt, tot_prompts.system_prompt)
        self._log_llm_call("PROPOSE", prompt, response_text)
        
        proposals = [line.strip() for line in response_text.split('\n') if line.strip()]
        return proposals, token_count

    def _evaluate(self, candidate: str) -> tuple[float, int]:
        remaining = self._extract_remaining(candidate)
        if not remaining:
            return 0.001, 0
        
        prompt = tot_prompts.value_prompt.format(input=remaining)
        response_text, token_count = self.llm.generate(prompt, tot_prompts.system_prompt)
        self._log_llm_call("EVALUATE", prompt, response_text)

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
    
    with open("output.txt", "w") as f, open("log.txt", "w") as debug_log:
        f.write(f"Problem: {problem}\n")
        debug_log.write(f"=== DEBUG LOG for Problem: {problem} ===\n")
        
        llm = KimiLLM()
        tot = TreeOfThoughts(llm, debug_log=debug_log)
        best_path = tot.solve(problem, b=5, d=3, log_file=f)
        
        f.write("\n=== BEST PATH ===\n")
        f.write(best_path)
    
    print("\n=== BEST PATH ===")
    print(best_path)
    print("\nDebug log saved to log.txt")
