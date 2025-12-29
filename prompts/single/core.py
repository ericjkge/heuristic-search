action_prompt = """
You are an expert Connect 4 player.

You are given:
- Current board: 

{board}

- You are playing as: {number} 

Game rules:
- The board is a 7x6 grid (rows and columns are both 1-indexed).
- Empty cells are represented by "0".
- Player 1 is represented by "1".
- Player 2 is represented by "2".
- A move consists of choosing a column where a disc will fall to the lowest available row.
- You win by connecting four of your discs horizontally, vertically, or diagonally.
- Columns that are full are illegal moves.

Example: Illegal move (column 4 is full, so next move CANNOT be 4)

0 0 0 1 0 0 0
0 0 0 1 0 0 0
0 0 0 2 0 0 0
0 0 0 1 0 0 0
0 0 0 2 0 0 0
2 0 0 1 0 0 2

Analyze and propose the best column to play. Format your response as:
ANALYSIS: <your reasoning in MAX 1 sentence>
MOVE: <column number 1-7>"""