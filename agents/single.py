import re

class SingleAgent:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def extract_move(self, response):
        match = re.search(r"MOVE:\s*([1-7])", response)
        return int(match.group(1)) if match else None

    def choose_move(self, board, color):
        prompt = self.prompt.format(color=color, board=str(board))

        response = self.llm.generate(prompt)
        move = self.extract_move(response)

        return move
        

