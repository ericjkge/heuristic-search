from dotenv import load_dotenv

from c4_board import Connect4Board
from log_utils import TextLogger, default_log_path
from models import GeminiLLM
from prompts import c4_perspective, c4_role
from agents.multi_agent import MultiAgent

load_dotenv()


def main():
    log_path = default_log_path("pr")
    logger = TextLogger(log_path)
    logger.clear()

    board = Connect4Board()

    # Multi-agent setup (Perspective-based) - Player 1
    perspective_prompts = c4_perspective.PERSPECTIVES
    perspective_roles = ["OFFENSIVE", "DEFENSIVE", "POSITIONAL", "CONCLUSION"]
    multi_perspective = MultiAgent(
        GeminiLLM,
        perspective_prompts,
        c4_perspective.propose_prompt,
        c4_perspective.debate_prompt,
        c4_perspective.conclusion_prompt,
        num_rounds=1,
        roles=perspective_roles,
    )

    # Multi-agent setup (Role-based) - Player 2
    role_prompts = c4_role.ROLE
    role_roles = ["PROPOSER", "CRITIC", "REVISER", "CONCLUSION"]
    multi_role = MultiAgent(
        GeminiLLM,
        role_prompts,
        c4_role.propose_prompt,
        c4_role.debate_prompt,
        c4_role.conclusion_prompt,
        num_rounds=1,
        roles=role_roles,
    )

    print("Starting Connect 4. Multi (Perspective) (1) vs Multi (Role) (2)\n")
    print(board, "\n")
    logger.header("Connect4: Multi (Perspective) (1) vs Multi (Role) (2)", {"log": log_path})

    turn = 0
    while True:
        winner = board.check_winner()
        if winner:
            print(f"Game over: {winner}")
            logger.log()
            logger.log(f"Game over: {winner}")
            break

        if board.current_player == "1":
            turn += 1
            legal = board.legal_moves()
            logger.turn_header(turn, "1", str(board))
            logger.log(f"Legal: {' '.join(str(m) for m in legal)}")
            move = multi_perspective.choose_move(board, legal, "1", logger=logger)
            board.drop(move)
            print(f"Multi (Perspective) plays column: {move}")
        else:
            legal = board.legal_moves()
            logger.log()
            logger.turn_header(turn, "2", str(board))
            logger.log(f"Legal: {' '.join(str(m) for m in legal)}")
            move = multi_role.choose_move(board, legal, "2", logger=logger)
            board.drop(move)
            print(f"Multi (Role) plays column: {move}")

        print(board, "\n")


if __name__ == "__main__":
    main()