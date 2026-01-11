from search.mcts import MCTS
from games.chess.prompts import mcts as prompts
from games.chess.config import config


class MCTSAgent:
    def __init__(self, llm, prompts_module=None, game_config=None, value_oracle=None):
        self.llm = llm()
        self.mcts = MCTS(
            self.llm,
            prompts_module or prompts,
            game_config or config,
            value_oracle=value_oracle
        )
    
    def choose_move(self, board, player=None):
        legal_moves = board.legal_moves()
        if not legal_moves:
            return None
        
        best_move = self.mcts.search(board)
        
        if best_move and best_move in legal_moves:
            return best_move
        
        raise RuntimeError(
            f"MCTS returned invalid move '{best_move}'. "
            f"Legal moves: {legal_moves[:10]}..."
        )
    
    def clear_cache(self):
        self.mcts._policy_cache.clear()
