#!/usr/bin/env bash
# Benchmark miniF2F: launch problems in parallel, aggregate JSONL results.
#
# Usage: bash bench_minif2f.sh <mode> [max_rounds] [split]
#   mode: baseline, cot, self-consistency, self-refine, debate, gen-ver
#   split: valid (default), test
#
# Examples:
#   bash bench_minif2f.sh baseline
#   bash bench_minif2f.sh gen-ver 5 test
#   N=20 bash bench_minif2f.sh cot
#
# Environment variables:
#   N           - number of problems (default: all)
#   PROCS       - max parallel processes (default: 5, Lean uses 2-4 GB each)
#   RESULTS_DIR - results directory (default: results)

set -euo pipefail

MODE="${1:?Usage: bash bench_minif2f.sh <mode> [max_rounds] [split]}"
MAX_ROUNDS="${2:-5}"
SPLIT="${3:-test}"

N="${N:-0}"
PROCS="${PROCS:-10}"
RESULTS_DIR="${RESULTS_DIR:-results}"
DATA_DIR="data/minif2f"
DATASET="$DATA_DIR/problems_${SPLIT}.json"

if [ ! -f "$DATASET" ]; then
    echo "Dataset not found: $DATASET" >&2
    echo "Run: python3 gen_minif2f.py" >&2
    exit 1
fi

# Map mode name to CLI flag
mode_flag() {
    case "$1" in
        baseline)          echo "--baseline" ;;
        cot)               echo "--cot" ;;
        self-consistency)  echo "--self_consistency" ;;
        self-refine)       echo "--self_refine" ;;
        debate)            echo "--debate" ;;
        gen-ver)           echo "" ;;
        *) echo "Unknown mode: $1" >&2; exit 1 ;;
    esac
}

FLAG=$(mode_flag "$MODE")

# Clean old results
mkdir -p "$RESULTS_DIR"
STEM="problems_${SPLIT}"
rm -f "$RESULTS_DIR/minif2f_${MODE}_${STEM}.jsonl"

# Extract problem names
if [ "$N" -gt 0 ]; then
    NAMES=$(python3 -c "
import json
data = json.load(open('$DATASET'))[:$N]
print('\n'.join(e['name'] for e in data))
")
else
    NAMES=$(python3 -c "
import json
data = json.load(open('$DATASET'))
print('\n'.join(e['name'] for e in data))
")
fi

TOTAL=$(echo "$NAMES" | wc -l | tr -d ' ')

echo "Mode: $MODE | Max rounds: $MAX_ROUNDS | Split: $SPLIT | N: $TOTAL | Procs: $PROCS"
echo "Dataset: $DATASET"
echo

echo "Launching $TOTAL problem jobs..."
START_TIME=$(date +%s)

# Launch all jobs via xargs
echo "$NAMES" | xargs -P "$PROCS" -I {} bash -c '
    python3 run_minif2f.py '"$FLAG"' \
        --max_rounds '"$MAX_ROUNDS"' \
        --dataset '"$DATASET"' \
        --problem_names "{}" \
        --results_dir '"$RESULTS_DIR"' \
        --log 2>&1 | tail -1
'

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo
echo "All jobs complete. Elapsed: ${ELAPSED}s"
echo

# Aggregate results
python3 -c "
import json
from pathlib import Path

results_file = Path('$RESULTS_DIR') / 'minif2f_${MODE}_${STEM}.jsonl'
if not results_file.exists():
    print('No results found.')
    exit()

results = []
for line in results_file.read_text().strip().split('\n'):
    if line:
        results.append(json.loads(line))

if not results:
    print('No results found.')
    exit()

solved = sum(1 for r in results if r['solved'])
total = len(results)
avg_tok = sum(r['total_tokens'] for r in results) / total
avg_rounds = sum(r['rounds'] for r in results) / total
total_elapsed = sum(r.get('elapsed', 0) for r in results)

print(f'Solved: {solved}/{total} ({100*solved/total:.0f}%)')
print(f'Avg tokens: {avg_tok:.0f}')
print(f'Avg rounds: {avg_rounds:.1f}')
print(f'Total elapsed: {total_elapsed:.0f}s')
print(f'Avg time/problem: {total_elapsed/total:.1f}s')
"
