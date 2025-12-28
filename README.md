# Perspective-Based Multi-Agent Reasoning

A multi-agent debate framework where LLM agents with different seeded perspectives collaborate to make decisions.

## Quick Start

### Chess
```bash
# 1. Set API key in .env
echo "KIMI_API_KEY=your_key_here" > .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download Stockfish binary and move to project root
# Get from https://stockfishchess.org/download/
# macOS example:
mv ~/Downloads/stockfish-macos-m1-apple-silicon ./

# 4. Run chess (agents vs Stockfish)
python main.py
```

### Connect 4
```bash
# 1. Install Connect 4 engine
git clone https://github.com/RenaudGaudron/connect-4-game-engine.git
cd connect-4-game-engine && pip install . && cd ..

# 2. Run Connect 4 (agents vs engine)
python main_connect4.py
```

**Configuration**: `NUM_ROUNDS` (debate rounds), `ENGINE_PATH` (chess), engine depth in code (Connect 4)

## Files

| File | Description |
|------|-------------|
| `main.py` / `main_connect4.py` | Multi-agent game loops |
| `baseline.py` | Single-LLM baseline for Connect 4 |
| `prompts.py` / `prompts_connect4.py` | Perspective definitions and prompt templates |
| `models.py` | LLM wrapper classes (Kimi, Gemini) |
| `log*.txt` | Generated logs of all LLM calls |