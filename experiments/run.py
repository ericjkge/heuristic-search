from agents.single import SingleAgent
from games.connect4.prompts import single
from games.connect4.board import Connect4Board
from games.connect4.engine import Connect4Engine
from games.connect4.config import config as connect4_config
from utils.llm import GeminiLLM
from utils.logging import setup_logging


setup_logging(log_dir="logs/connect4") # Change for different games

def run():
    board = Connect4Board()
    engine = Connect4Engine()
    agent1 = SingleAgent(GeminiLLM, single, connect4_config)
    agent2 = SingleAgent(GeminiLLM, single, connect4_config)

    while not board.check_winner():
        move1 = agent1.choose_move(board, "1")
        board.drop(move1)
        print(board)
        print(f"Agent 1 chose move {move1}")

        move2 = agent2.choose_move(board, "2")
        board.drop(move2)
        print(board)
        print(f"Agent 2 chose move {move2}")

        # engine.move_sequence = board.move_history
        # move = engine.get_move()
        # board.drop(move)
        # print(board)
        # print(f"Engine chose move {move}")

    print(board.check_winner())

if __name__ == "__main__":
    run()