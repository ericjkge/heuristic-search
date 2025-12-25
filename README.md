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
# 1. Build the Connect 4 solver
git clone https://github.com/PascalPons/connect4.git
cd connect4 && make

# 2. Move solver binary to project root
cp c4solver ../ && cd ..

# 3. Download opening book
wget https://github.com/PascalPons/connect4/releases/download/book/7x6.book
mv 7x6.book ./

# 4. Run Connect 4 (agents vs solver)
python main_connect4.py
```

**Configuration**: `NUM_ROUNDS` (debate rounds), `ENGINE_PATH`/`SOLVER_PATH` (engine binary)

## Files

| File | Description |
|------|-------------|
| `main.py` / `main_connect4.py` | Main game loops |
| `prompts.py` / `prompts_connect4.py` | Perspective definitions and prompt templates |
| `models.py` | LLM wrapper classes (Kimi, Gemini) |
| `log.txt` / `log_connect4.txt` | Generated logs of all LLM calls |
| `baseline/` | Tree-of-Thought baseline implementation |