class SingleAgent:
    def __init__(self, llm, prompts, game_config):
        self.llm = llm()
        self.prompts = prompts
        self.extract_move = game_config["extract_move"]
        self.format_state = game_config["format_state"]
        self.san_to_uci = game_config.get("san_to_uci") # Optional SAN to UCI converter for chess

    def choose_move(self, board, player, max_attempts=3):
        state = self.format_state(board, player)
        prompt = self.prompts.action_prompt.format(**state)

        # Build conversation history
        messages = [{"role": "user", "content": prompt}]

        # Get FEN for SAN conversion (chess)
        fen = state.get("fen")

        for _ in range(max_attempts):
            response = self.llm.generate(messages)
            move_san = self.extract_move(response, fen=fen)

            # Convert SAN to UCI for chess
            move = move_san
            if move_san and self.san_to_uci and fen:
                move = self.san_to_uci(move_san, fen)

            # Valid move (check UCI for chess, regular for other games)
            legal = board.legal_moves_uci() if hasattr(board, 'legal_moves_uci') else board.legal_moves()
            if move and (move in legal or move == 'PASS'):
                return move

            # Invalid move (add to history and retry) - show original SAN in error
            messages.append({"role": "model", "content": response})
            if move_san is None:
                error_msg = "Could not parse your move. Try again using the specified format."
            else:
                error_msg = f"'{move_san}' is not a legal move. Try again."
            messages.append({"role": "user", "content": error_msg})

        return None # All retries failed