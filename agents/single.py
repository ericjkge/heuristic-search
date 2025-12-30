class SingleAgent:
    def __init__(self, llm, prompts, game_config):
        self.llm = llm
        self.prompts = prompts
        self.extract_move = game_config["extract_move"]
        self.format_state = game_config["format_state"]

    def choose_move(self, board, player):
        state = self.format_state(board, player)
        prompt = self.prompts.action_prompt.format(**state)
        response = self.llm.generate(prompt)
        return self.extract_move(response)
