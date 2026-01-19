import re


class VerifierAgent:
    """Verifier agent that scores a move on a specific aspect."""

    def __init__(self, llm, prompt_template, name, format_state):
        self.llm = llm
        self.prompt_template = prompt_template
        self.name = name
        self.format_state = format_state

    def evaluate(self, board, player, move):
        """
        Evaluate a move and return score + reasoning.

        Args:
            board: Current board state
            player: Current player
            move: Proposed move to evaluate

        Returns:
            dict: {"score": float, "reasoning": str}
        """
        state = self.format_state(board, player)
        state["move"] = move

        prompt = self.prompt_template.format(**state)
        messages = [{"role": "user", "content": prompt}]

        response = self.llm.generate(messages)
        return self._parse_response(response)

    def _parse_response(self, response):
        """Parse REASONING and SCORE from response."""
        reasoning = ""
        score = 0.0

        # Extract reasoning
        reasoning_match = re.search(r"REASONING:\s*(.+?)(?=SCORE:|$)", response, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Extract score
        score_match = re.search(r"SCORE:\s*([\d.]+)", response, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                # Clamp to valid range
                score = max(0.0, min(1.0, score))
            except ValueError:
                score = 0.0

        return {"score": score, "reasoning": reasoning}
