import re
from utils.logging import get_logger

logger = get_logger(__name__)

class MultiAgent:
    def __init__(self, llm, prompts, perspectives):
        self.prompts = prompts
        self.agents = [(llm(), perspective) for perspective in perspectives]
        self.llm = llm()
        
    def extract_move(self, response):
        match = re.search(r"MOVE:\s*([1-7])", response)
        return int(match.group(1)) if match else None

    def debate(self, board, number, num_rounds=1): # For action (need to generalize for policy and value)
        proposals = []
        board_str = str(board)

        for round in range(num_rounds):
            prev_proposals = proposals # Memoryless debate (only previous round's proposals are considered)
            proposals = []

            for llm, perspective in self.agents:
                prompt = self.prompts.action_prompt.format(board=board_str, number=number, other_proposals=prev_proposals)
                response = llm.generate(prompt, system_prompt=perspective)
                move = self.extract_move(response)
                proposals.append((move, response))
                logger.info(f"Round={round} | Move={move} | Response={response} | Perspective={perspective[37:47]}")

        # Use LLM to aggregate proposals for now
        aggregate_prompt = self.prompts.conclusion_prompt.format(board=board_str, proposals=proposals, number=number)
        response = self.llm.generate(aggregate_prompt)
        logger.info(f"Aggregate response: {response}")
        return response

    def choose_move(self, board, number):
        response = self.debate(board, number, 2) # 2 rounds of debate
        move = self.extract_move(response)
        return move