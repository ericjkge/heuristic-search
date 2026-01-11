"""
Value oracle implementations for comparing against Stockfish.

5 setups:
- single: single LLM evaluation (baseline)
- vanilla_judge: vanilla agents + judge aggregation
- vanilla_average: vanilla agents + average of values
- feature_judge: feature agents + judge aggregation
- feature_average: feature agents + average of values
"""

from games.chess.config import extract_value, format_state
from games.chess.prompts import value_debate as prompts


class TrackedLLM:
    def __init__(self, llm):
        self.llm = llm
        self.reset_stats()
    
    def reset_stats(self):
        self.total_tokens = 0
        self.total_time = 0.0
        self.call_count = 0
    
    def generate(self, messages, system_prompt=""):
        response = self.llm.generate(messages, system_prompt)
        self.total_tokens += getattr(self.llm, 'last_tokens', 0)
        self.total_time += getattr(self.llm, 'last_elapsed', 0.0)
        self.call_count += 1
        return response
    
    def get_stats(self):
        return {
            "tokens": self.total_tokens,
            "time": self.total_time,
            "calls": self.call_count,
        }


class SingleOracle:
    def __init__(self, llm):
        self.llm = TrackedLLM(llm)
    
    def evaluate(self, board):
        self.llm.reset_stats()
        
        player = board.turn()
        state = format_state(board, player)
        prompt = prompts.single_prompt.format(**state)
        
        response = self.llm.generate([{"role": "user", "content": prompt}])
        value = extract_value(response)
        
        return value, response, self.llm.get_stats()


class BaseOracle:
    n_agents = 3
    n_rounds = 2
    
    def __init__(self, llm):
        self.llm = TrackedLLM(llm)
    
    def _run_debate(self, board, prompt_fn):
        """Run n_agents x n_rounds debate. Returns list of final responses."""
        player = board.turn()
        state = format_state(board, player)
        
        proposals = ["None yet"] * self.n_agents
        
        for round_idx in range(self.n_rounds):
            new_proposals = []
            
            for agent_idx in range(self.n_agents):
                other_proposals = [p for i, p in enumerate(proposals) if i != agent_idx]
                prev_text = "\n".join(other_proposals) if any(p != "None yet" for p in other_proposals) else "None yet"
                
                prompt = prompt_fn(state, agent_idx, prev_text)
                response = self.llm.generate([{"role": "user", "content": prompt}])
                new_proposals.append(response)
            
            proposals = new_proposals
        
        return proposals, state


class VanillaJudgeOracle(BaseOracle):
    def evaluate(self, board):
        self.llm.reset_stats()
        
        def prompt_fn(state, agent_idx, prev):
            return prompts.vanilla_prompt.format(**state, prev_proposals=prev)
        
        proposals, state = self._run_debate(board, prompt_fn)
        
        proposals_text = "\n\n".join(f"[Agent {i+1}]: {p}" for i, p in enumerate(proposals))
        judge_prompt = prompts.judge_prompt.format(**state, proposals=proposals_text)
        judge_response = self.llm.generate([{"role": "user", "content": judge_prompt}])
        
        final_value = extract_value(judge_response)
        detail = f"Proposals:\n{proposals_text}\n\nJudge:\n{judge_response}"
        
        return final_value, detail, self.llm.get_stats()


class VanillaAverageOracle(BaseOracle):
    def evaluate(self, board):
        self.llm.reset_stats()
        
        def prompt_fn(state, agent_idx, prev):
            return prompts.vanilla_prompt.format(**state, prev_proposals=prev)
        
        proposals, _ = self._run_debate(board, prompt_fn)
        
        values = [extract_value(p) for p in proposals]
        final_value = sum(values) / len(values)
        
        proposals_text = "\n\n".join(f"[Agent {i+1}]: {p}" for i, p in enumerate(proposals))
        detail = f"Final round proposals:\n{proposals_text}\n\nValues: {values}\nAverage: {final_value:.3f}"
        return final_value, detail, self.llm.get_stats()


class FeatureJudgeOracle(BaseOracle):
    def evaluate(self, board):
        self.llm.reset_stats()
        
        def prompt_fn(state, agent_idx, prev):
            feature = prompts.FEATURES[agent_idx]
            return prompts.feature_prompt.format(**state, feature=feature, prev_proposals=prev)
        
        proposals, state = self._run_debate(board, prompt_fn)
        
        proposals_text = "\n\n".join(
            f"[{prompts.FEATURES[i][:30]}...]: {p}" for i, p in enumerate(proposals)
        )
        judge_prompt = prompts.judge_prompt.format(**state, proposals=proposals_text)
        judge_response = self.llm.generate([{"role": "user", "content": judge_prompt}])
        
        final_value = extract_value(judge_response)
        detail = f"Proposals:\n{proposals_text}\n\nJudge:\n{judge_response}"
        
        return final_value, detail, self.llm.get_stats()


class FeatureAverageOracle(BaseOracle):
    def evaluate(self, board):
        self.llm.reset_stats()
        
        def prompt_fn(state, agent_idx, prev):
            feature = prompts.FEATURES[agent_idx]
            return prompts.feature_prompt.format(**state, feature=feature, prev_proposals=prev)
        
        proposals, _ = self._run_debate(board, prompt_fn)
        
        values = [extract_value(p) for p in proposals]
        final_value = sum(values) / len(values)
        
        proposals_text = "\n\n".join(
            f"[{prompts.FEATURES[i][:30]}...]: {p}" for i, p in enumerate(proposals)
        )
        detail = f"Final round proposals:\n{proposals_text}\n\nValues: {values}\nAverage: {final_value:.3f}"
        return final_value, detail, self.llm.get_stats()


ORACLES = {
    "single": SingleOracle,
    "vanilla_judge": VanillaJudgeOracle,
    "vanilla_average": VanillaAverageOracle,
    "feature_judge": FeatureJudgeOracle,
    "feature_average": FeatureAverageOracle,
}
