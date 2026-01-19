class SingleAgent:
    def __init__(self, llm, prompts, game_config):
        self.llm = llm()
        self.prompts = prompts
        self.extract_move = game_config["extract_move"]
        self.format_state = game_config["format_state"]

    def choose_move(self, board, player, max_attempts=3):
        state = self.format_state(board, player)
        prompt = self.prompts.action_prompt.format(**state)

        # Build conversation history
        messages = [{"role": "user", "content": prompt}]
        
        for _ in range(max_attempts):
            response = self.llm.generate(messages)
            move = self.extract_move(response)

            # Valid move
            if move and (move in board.legal_moves() or move == 'PASS'):
                return move
            
            # Invalid move (add to history and retry)
            messages.append({"role": "assistant", "content": response})
            if move is None:
                error_msg = "Could not parse your move. Try again using the specified format."
            else:
                error_msg = f"'{move}' is occupied or out-of-bounds. Try again using the specified format."
            messages.append({"role": "user", "content": error_msg})
        
        return None # All retries failed