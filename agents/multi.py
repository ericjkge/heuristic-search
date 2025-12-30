from utils.logging import get_logger

logger = get_logger(__name__)

class MultiAgent:
    def __init__(self, llm, prompts, perspectives, game_config):
        self.prompts = prompts
        self.agents = [(llm(), perspective) for perspective in perspectives]
        self.llm = llm()
        self.extract_move = game_config["extract_move"]
        self.format_state = game_config["format_state"]

    def debate(self, board, player, num_rounds=1, prompt_type="action"):
        proposals = []
        state = self.format_state(board, player)

        prompt_template = getattr(self.prompts, f"{prompt_type}_prompt") # Genearlization for action, policy, and value prompts

        for round in range(num_rounds):
            prev_proposals = proposals # Memoryless debate (only previous round's proposals are considered)
            proposals = []

            for llm, perspective in self.agents:
                prompt = prompt_template.format(**state, other_proposals=prev_proposals)
                response = llm.generate(prompt, system_prompt=perspective)
                move = self.extract_move(response)
                proposals.append((move, response))
                logger.info(f"Round={round} | Move={move} | Response={response} | Perspective={perspective[37:47]}")

        # Use LLM to aggregate proposals for now
        aggregate_prompt = self.prompts.conclusion_prompt.format(**state, proposals=proposals)
        response = self.llm.generate(aggregate_prompt)
        logger.info(f"Aggregate response: {response}")
        return response

    def choose_move(self, board, player):
        response = self.debate(board, player, 2, "action") # 2 rounds of debate
        return self.extract_move(response)