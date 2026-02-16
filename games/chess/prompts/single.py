action_prompt = """
You are player {player} in a chess game.

FEN: {fen}
PGN: {pgn}
Legal moves: {legal_moves}

IMPORTANT: You must choose a move from the legal moves list above.

ANALYSIS: <1-2 sentences>
MOVE: <move from legal moves list>
"""

# NOTE: Few-shot prompting does NOT work better for generation! Few-shot examples tend to bias game-play (e.g. tactical positions with captures -> aggressive moves)

# action_fewshot = """
# FEN: r1b1k2r/pp1pnpbp/2n1p1p1/q1p5/4PP2/2NPBN2/PPP1Q1PP/R3KB1R b KQkq - 7 9                                                                                                                                                                                                                                       
# PGN: 1. Nc3 c5 2. d3 g6 3. e4 Bg7 4. f4 Nc6 5. Nf3 e6 6. Be3 Nge7 7. Qe2 Qb6 8. Na4 Qa5+ 9. Nc3                                                                         
# Legal moves: Bd4, Be5, Bf6, Bf8, Bh6, Bxc3+, Kd8, Kf8, Nb4, Nb8, Nd4, Nd5, Nd8, Ne5, Nf5, Ng8, O-O, Qa3, Qa4, Qa6, Qb4, Qb5, Qb6, Qc7, Qd8, Qxa2, Qxc3+, Rb8, Rf8, Rg8, a6, b5, b6, c4, d5, d6, e5, f5, f6, g5, h5, h6   

# ANALYSIS: Bxc3+ is best because it's a forcing check that removes the key c3-knight and, after the almost-forced bxc3, lets Black play Qxc3+ with tempo to win further material (often continuing to Qxa1).
# MOVE: Bxc3+

# FEN: rn3rk1/1p3p2/p2p2p1/2pN2q1/2P1P2p/3B3P/PP3P2/R2QR1K1 w - - 1 18
# PGN: 1. d4 c5 2. c4 d6 3. Bf4 g6 4. Nf3 Bh6 5. Bxh6 Nxh6 6. Nc3 Nf5 7. d5 O-O 8. e4 Nd4 9. Bd3 Bd7 10. O-O a6 11. Ng5 h5 12. Re1 e6 13. dxe6 Nxe6 14. Nxe6 Bxe6 15. Nd5 h4 16. h3 Bxh3 17. gxh3 Qg5+                                                                                                                                                                          
# Legal moves: Kf1, Kh1, Kh2, Qg4  

# ANALYSIS: Kh1 is best because it's the only move that tucks the king off the g-file and sidesteps immediate queen checks (…Qg2# ideas), while keeping White's queen free to meet …Qxd2 with Qxd2 and leaving Black with no forcing continuation.
# MOVE: Kh1

# FEN: {fen}
# PGN: {pgn}
# Legal moves: {legal_moves}

# ANALYSIS:
# MOVE:
# """