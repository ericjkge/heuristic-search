from agents.single import SingleAgent
from prompts.single.action import PROMPT
from games.connect4.board import Connect4Board
from games.connect4.engine import Connect4Engine
from utils.llm import KimiLLM

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