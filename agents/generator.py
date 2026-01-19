class GeneratorAgent:
    """Generator agent that proposes moves with conversation history."""

    def __init__(self, llm, initial_prompt, feedback_prompt, extract_move, format_state, name="generator"):
        self.llm = llm
        self.initial_prompt = initial_prompt
        self.feedback_prompt = feedback_prompt
        self.extract_move = extract_move
        self.format_state = format_state
        self.name = name
        self.conversation_history = []

    def reset(self):
        """Clear conversation history for new move generation."""
        self.conversation_history = []

    def propose_move(self, board, player, feedback=None):
        """
        Propose a move. Uses initial prompt on first call, feedback prompt on subsequent calls.

        Args:
            board: Current board state
            player: Current player
            feedback: Optional dict with verifier feedback from previous iteration

        Returns:
            tuple: (move, response_text)
        """
        state = self.format_state(board, player)

        if feedback is None:
            # First iteration: use initial prompt
            prompt = self.initial_prompt.format(**state)
            self.conversation_history = [{"role": "user", "content": prompt}]
        else:
            # Subsequent iterations: use feedback prompt
            prompt = self.feedback_prompt.format(
                previous_move=feedback["move"],
                immediate_loss_score=feedback["immediate_loss"]["score"],
                immediate_loss_reasoning=feedback["immediate_loss"]["reasoning"],
                illegal_move_score=feedback["illegal_move"]["score"],
                illegal_move_reasoning=feedback["illegal_move"]["reasoning"],
                aggressive_score=feedback["aggressive"]["score"],
                aggressive_reasoning=feedback["aggressive"]["reasoning"],
                defensive_score=feedback["defensive"]["score"],
                defensive_reasoning=feedback["defensive"]["reasoning"],
                shape_score=feedback["shape"]["score"],
                shape_reasoning=feedback["shape"]["reasoning"],
                final_score=feedback["final_score"]
            )
            self.conversation_history.append({"role": "user", "content": prompt})

        response = self.llm.generate(self.conversation_history)
        self.conversation_history.append({"role": "assistant", "content": response})

        move = self.extract_move(response)
        return move, response
