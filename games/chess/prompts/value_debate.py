# Chess value evaluation prompts

FEATURES = [
    "material balance: piece count, pending captures, trades",
    "king safety: castling, pawn shield, attacking pieces nearby",
    "piece activity and coordination: centralization, mobility, outposts",
]


# === SINGLE PROMPT (baseline) ===

single_prompt = """Position (FEN): {fen}
Move history: {pgn}

Evaluate the current position for {player} on scale -1.0 (losing) to +1.0 (winning).

VALUE: <-1.0 to +1.0>"""


# === VANILLA AGENT PROMPT ===

vanilla_prompt = """Position (FEN): {fen}
Move history: {pgn}

Prior evaluations from other agents:
{prev_proposals}

Using the proposals from other agents, evaluate the current position for {player} on scale -1.0 (losing) to +1.0 (winning).

ANALYSIS: <1 sentence>
VALUE: <-1.0 to +1.0>"""


# === FEATURE AGENT PROMPT ===

feature_prompt = """Position (FEN): {fen}
Move history: {pgn}

Your specialty: {feature}

Prior evaluations from other agents:
{prev_proposals}

Using the proposals from other agents and your speciality perspective, evaluate the current position for {player} on scale -1.0 (losing) to +1.0 (winning).

ANALYSIS: <1 sentence>
VALUE: <-1.0 to +1.0>"""


# === JUDGE PROMPT ===

judge_prompt = """Position (FEN): {fen}
Move history: {pgn}

Agent evaluations:
{proposals}

Synthesize these evaluations for {player}.

VERDICT: <1 sentence>
VALUE: <-1.0 to +1.0>"""
