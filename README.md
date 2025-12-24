# Perspective-Based Multi-Agent Reasoning

A multi-agent debate framework where LLM agents with different seeded perspectives collaborate to make decisions.

## Quick Start

```bash
# 1. Set API key in .env
echo "KIMI_API_KEY=your_key_here" > .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the game (agents vs Stockfish)
python main.py
```

**Configuration** (in `main.py`): `NUM_ROUNDS` (debate rounds), `ENGINE_PATH` (Stockfish binary)

## Files

| File | Description |
|------|-------------|
| `main.py` | Main game loop with debate and aggregation |
| `models.py` | LLM wrapper classes (Kimi, Gemini) |
| `prompts.py` | Perspective definitions and prompt templates |
| `log.txt` | Generated logs of all LLM calls |
| `baseline/` | Tree-of-Thought baseline implementation |

## Output

All LLM calls are logged to `log.txt` with:
- Agent perspective
- Full prompt sent
- Full response received
- Extracted move
