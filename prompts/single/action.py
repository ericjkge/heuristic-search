PROMPT = """
You are an expert Connect 4 player.

You are given:
- Current board: 

{board}

- You are playing as: {color} 

Game rules:
- The board is a 7x6 grid (rows and columns are both 1-indexed).
- Empty cells are represented by "0".
- White discs are represented by "1".
- Black discs are represented by "2".
- A move consists of choosing a column where a disc will fall to the lowest available row.
- You win by connecting four of your discs horizontally, vertically, or diagonally.
- Do NOT suggest illegal moves (e.g. columns that are full).

Analyze and propose the best column to play. Format your response as:
ANALYSIS: <your reasoning in MAX 1 sentence>
MOVE: <your chosen move in column number 1-7>"""