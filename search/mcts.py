import copy
import math


# === MCTS HYPERPARAMETERS ===
TAU = 0.5           # Temperature for rank-to-score conversion
TOP_K = 5           # Number of top moves from LLM
C_PUCT = 1.5        # PUCT exploration constant
NUM_SIMULATIONS = 25  # MCTS iterations per move


class MCTSNode:
    def __init__(self, state, parent=None, action_from_parent=None):
        self.state = state # FEN board position
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = {} # action : MCTSNode
        self.N = {} # action: visit counts
        self.W = {} # action: total value
        self.P = {} # action: prior probability
    
    def Q(self, action):
        n = self.N.get(action, 0)
        if n == 0:
            return 0.0
        return self.W.get(action, 0.0) / n
    
    def total_visits(self):
        return sum(self.N.values())
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def is_root(self):
        return self.parent is None


class MCTS:
    def __init__(self, llm, prompts, game_config, value_oracle=None):
        self.llm = llm
        self.prompts = prompts
        self.format_state = game_config["format_state"]
        self.extract_ranked_moves = game_config["extract_ranked_moves"]
        self.extract_value = game_config["extract_value"]
        self.value_oracle = value_oracle
        self._policy_cache = {} # Cache for policy priors to avoid redundant LLM calls
    
    def search(self, board):
        root = MCTSNode(state=board.to_positions())
        
        for _ in range(NUM_SIMULATIONS):
            sim_board = copy.deepcopy(board) # Tree persists but new sim_board used for each simulation
            leaf, sim_board = self._select(root, sim_board)
            value = self._evaluate(sim_board)
            self._backpropagate(leaf, value)
        
        if not root.N:
            raise RuntimeError(
                f"MCTS failed: no visits recorded at root. "
                f"State: {root.state}, Legal moves: {board.legal_moves()[:10]}..."
            )
        # Select move with highest visit count        
        return max(root.N.keys(), key=lambda a: root.N[a])
    
    def _select(self, node, board):
        current = node
        
        while not current.is_leaf():
            # Select action with highest PUCT value
            action = self._select_action(current)
            
            # Move to child
            board.push(action)
            current = current.children[action]
        
        # Expand if not terminal
        if board.winner() is None:
            self._expand(current, board)
            
            # If we just expanded, select one child to evaluate
            if current.P:  # Has policy prior
                action = self._select_action(current)
                board.push(action)
                current = current.children[action]
        
        return current, board
    
    def _select_action(self, node):
        total_n = node.total_visits()
        sqrt_total = math.sqrt(total_n) if total_n > 0 else 1.0
        
        best_action = None
        best_value = float('-inf')
        
        # Find highest PUCT
        for action in node.P.keys():
            q = node.Q(action)
            p = node.P[action]
            n = node.N.get(action, 0)
            
            # PUCT formula: Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            puct = q + C_PUCT * p * sqrt_total / (1 + n)
            
            if puct > best_value:
                best_value = puct
                best_action = action
        
        return best_action
    
    def _expand(self, node, board):
        # Check cache first
        if node.state in self._policy_cache:
            node.P = self._policy_cache[node.state].copy()
        else:
            # Get policy prior from LLM
            player = board.turn()
            legal_moves = board.legal_moves()
            
            if not legal_moves:
                return
            
            state = self.format_state(board, player)
            prompt = self.prompts.policy_prompt.format(**state)
            
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.generate(messages)
            
            ranked_moves = self.extract_ranked_moves(response, legal_moves)
            
            # Convert ranks to policy prior using softmax
            node.P = self._ranks_to_prior(ranked_moves, legal_moves)
            
            # Cache the policy
            self._policy_cache[node.state] = node.P.copy()
        
        # Initialize edge statistics and create child nodes
        for action in node.P.keys():
            node.N[action] = 0
            node.W[action] = 0.0
            
            child_board = copy.deepcopy(board)
            child_board.push(action)
            child = MCTSNode(
                state=child_board.to_positions(),
                parent=node,
                action_from_parent=action
            )
            node.children[action] = child
    
    def _ranks_to_prior(self, ranked_moves, legal_moves):
        valid_ranked = [m for m in ranked_moves if m in legal_moves][:TOP_K]
        
        if not valid_ranked:
            valid_ranked = legal_moves[:TOP_K]
        
        scores = {}
        for i, move in enumerate(valid_ranked, start=1):
            scores[move] = math.exp(-TAU * i) 
        
        total = sum(scores.values())
        return {move: score / total for move, score in scores.items()} # Softmax
    
    def _evaluate(self, board):
        winner = board.winner()
        if winner is not None:
            if winner == 0:  # Draw
                return 0.0
            current_player = board.turn()
            return 1.0 if winner != current_player else -1.0
        
        if self.value_oracle:
            value, _, _ = self.value_oracle.evaluate(board)
        else:
            player = board.turn()
            state = self.format_state(board, player)
            prompt = self.prompts.value_prompt.format(**state)
            
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.generate(messages)
            
            value = self.extract_value(response)
        
        return -value # Value is from current player's perspective, so negate it for the caller (opponent's perspective)
    
    def _backpropagate(self, node, value):
        current = node
        v = value
        
        while current.parent is not None:
            action = current.action_from_parent
            parent = current.parent
            
            # Update parent's statistics for this action
            parent.N[action] = parent.N.get(action, 0) + 1
            parent.W[action] = parent.W.get(action, 0.0) + v
            
            # Flip value for next level up (opponent's perspective)
            v = -v
            current = parent
