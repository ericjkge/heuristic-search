from utils.logging import get_logger

logger = get_logger(__name__)

class MultiAgent:
    def __init__(self, llm, prompts, perspectives, game_config):
        self.llm = llm()
        self.prompts = prompts
        self.perspectives = perspectives
        self.extract_move = game_config["extract_move"]
        self.format_state = game_config["format_state"]

    def _agent_move(self, board, prompt, max_attempts=3):
        messages = [{"role": "user", "content": prompt}]

        # Retries for each perspective agent
        for _ in range(max_attempts):
            response = self.llm.generate(messages)
            move = self.extract_move(response)
            
            if move and (move in board.legal_moves() or move == 'PASS'):
                return response
            
            # Invalid move
            messages.append({"role": "assistant", "content": response})
            if move is None:
                error_msg = "Could not parse your move. Try again using the specified format."
            else:
                error_msg = f"'{move}' is not a legal move. Try again using the specified format."
            messages.append({"role": "user", "content": error_msg})
        
        return None  # Return None if invalid

    def debate(self, board, player, num_rounds=1, prompt_type="action"):
        proposals = []
        state = self.format_state(board, player)
        prompt_template = getattr(self.prompts, f"{prompt_type}_prompt")

        for round_num in range(num_rounds):
            prev_proposals = proposals
            proposals = []

            for perspective in self.perspectives:
                prompt = prompt_template.format(**state, perspective=perspective, proposals=prev_proposals)
                response = self._agent_move(board, prompt)
                proposals.append(response)
                logger.info(f"Round={round_num} | Response={response} | Perspective={perspective[:30]}...")

        return proposals

    def aggregate(self, board, player, proposals, max_attempts=3):
        state = self.format_state(board, player)
        prompt = self.prompts.aggregate_prompt.format(**state, proposals=proposals)
        messages = [{"role": "user", "content": prompt}]
        
        # Retries for final aggregate agent
        for _ in range(max_attempts):
            response = self.llm.generate(messages)
            move = self.extract_move(response)
            
            if move and (move in board.legal_moves() or move == 'PASS'):
                return move
            
            # Invalid move
            messages.append({"role": "assistant", "content": response})
            if move is None:
                error_msg = "Could not parse your move. Try again using the specified format."
            else:
                error_msg = f"'{move}' is not a legal move. Try again using the specified format."
            messages.append({"role": "user", "content": error_msg})
        
        return None

    def choose_move(self, board, player):
        proposals = self.debate(board, player, num_rounds=2, prompt_type="action")
        move = self.aggregate(board, player, proposals)
        return move
