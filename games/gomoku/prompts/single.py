action_prompt = """You are an expert Gomoku player.

Game Rules:
- Gomoku is played on a {size} x {size} grid.
- Two players take turns placing stones on empty intersections.
- The first player to get exactly 5 stones in a row (horizontally, vertically, or diagonally) wins.
- You cannot place a stone on an already occupied position.

Board Coordinate System:
- The board uses 1-indexed coordinates.
- The top-left corner is position (1,1).
- The first number is the row, the second is the column.
- Positions are written as "row,column" (e.g., "8,8" means row 8, column 8).

Current Game State:
- You are Player {player}
- Your positions: {player_positions}
- Opponent positions: {opponent_positions}

Your Task:
1. Analyze the current board state.
2. Identify any winning opportunities or threats to block.
3. Choose the best position to place your stone.

Format your response as:
ANALYSIS: <your reasoning in MAX 2 sentences>
MOVE: <row>,<column>

Example response:
ANALYSIS: I can create a threat by extending my line at row 8. This forces the opponent to defend.
MOVE: 8,10"""
