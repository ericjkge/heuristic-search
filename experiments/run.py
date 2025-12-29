from agents import SingleAgent
from prompts.single.action import PROMPT
from games import Connect4Board, Connect4Engine
from utils import KimiLLM, setup_logging

setup_logging(log_dir="logs/connect4")

def run():
    board = Connect4Board()
    llm = KimiLLM()
    engine = Connect4Engine()
    agent = SingleAgent(llm, PROMPT)


    while not board.check_winner():

        move = agent.choose_move(board, "1")
        board.drop(move)
        print(board)
        print(f"Agent chose move {move}")

        engine.move_sequence = board.move_history
        move = engine.get_move()
        board.drop(move)
        print(board)
        print(f"Engine chose move {move}")

    print(board.check_winner())

if __name__ == "__main__":
    run()