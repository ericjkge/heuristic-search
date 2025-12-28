import re
import time


class SingleAgent:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def extract_move(self, response):
        match = re.search(r"MOVE:\s*([1-7])", response)
        return int(match.group(1)) if match else None

    def choose_move(self, board, legal_moves, color, logger=None):
        moves_str = " ".join(str(m) for m in legal_moves)
        prompt = self.prompt.PROMPT.format(color=color, board=str(board), moves=moves_str)

        start = time.time()
        response, tokens = self.llm.generate(prompt)
        elapsed = time.time() - start
        move = self.extract_move(response)

        if logger:
            logger.log("Single Agent:")
            logger.log(response.strip())
            logger.log()

        if move in legal_moves:
            if logger:
                logger.log(f"--- STATS: {tokens} tokens, {elapsed:.2f}s, 1 LLM calls ---")
                logger.log(f">>> SINGLE CHOSEN: {move} <<<")
                logger.log()
            return move
        else:
            raise ValueError(f"Single agent chose invalid move: {move}. Legal moves: {legal_moves}")

        

