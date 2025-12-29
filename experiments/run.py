from agents.multi import MultiAgent
from prompts.multi import core, perspectives
from games.connect4.board import Connect4Board
from games.connect4.engine import Connect4Engine
from utils.llm import KimiLLM
from utils.logging import setup_logging

setup_logging(log_dir="logs/connect4")

def run():
    board = Connect4Board()
    engine = Connect4Engine()
    agent = MultiAgent(KimiLLM, core, perspectives.PERSPECTIVES)

    while not board.check_winner():
        move = agent.choose_move(board, "1")
        board.drop(move)
        print(board)
        print(f"Multi-agent chose move {move}")

        engine.move_sequence = board.move_history
        move = engine.get_move()
        board.drop(move)
        print(board)
        print(f"Engine chose move {move}")

    print(board.check_winner())

if __name__ == "__main__":
    run()