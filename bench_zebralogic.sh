#!/usr/bin/env bash
# Benchmark ZebraLogic: launch puzzles in parallel, aggregate JSONL results.
#
# Usage: bash bench_zebralogic.sh <mode> [max_rounds] [sizes...]
#   mode: baseline, cot, self-consistency, self-refine, debate, gen-ver
#
# Examples:
#   bash bench_zebralogic.sh baseline
#   bash bench_zebralogic.sh cot 5 5x4 6x3
#
# Environment variables:
#   N          - puzzles per size (default: 5)
#   PROCS      - max parallel processes (default: 10)
#   RESULTS_DIR - results directory (default: results)

set -euo pipefail

MODE="${1:?Usage: bash bench_zebralogic.sh <mode> [max_rounds] [sizes...]}"
MAX_ROUNDS="${2:-5}"
shift 2 2>/dev/null || shift 1 2>/dev/null || true

N="${N:-5}"
PROCS="${PROCS:-10}"
RESULTS_DIR="${RESULTS_DIR:-results}"
MODEL="${MODEL:-gpt-none}"
DATA_DIR="data/zebralogic"

# Remaining args are size filters (e.g. 5x4 6x3), empty = all
SIZES=("$@")

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

# Collect dataset files
DATASET_FILES=()
for f in "$DATA_DIR"/puzzles_*.json; do
    size=$(basename "$f" .json | sed 's/puzzles_//')
    if [ ${#SIZES[@]} -eq 0 ] || printf '%s\n' "${SIZES[@]}" | grep -qx "$size"; then
        DATASET_FILES+=("$f")
    fi
done

if [ ${#DATASET_FILES[@]} -eq 0 ]; then
    echo "No dataset files found." >&2
    exit 1
fi

# Clean old results for these datasets
mkdir -p "$RESULTS_DIR"
for f in "${DATASET_FILES[@]}"; do
    stem=$(basename "$f" .json)
    rm -f "$RESULTS_DIR/zebralogic_${MODE}_${MODEL}_${stem}.jsonl"
done

echo "Mode: $MODE | Model: $MODEL | Max rounds: $MAX_ROUNDS | N: $N | Procs: $PROCS"
echo "Datasets: ${DATASET_FILES[*]}"
echo

# Build list of (dataset, puzzle_id) jobs
JOBS=()
for f in "${DATASET_FILES[@]}"; do
    # Extract puzzle IDs (strings like "lgp-test-2x2-33")
    ids=$(python3 -c "
import json
data = json.load(open('$f'))[:$N]
print('\n'.join(str(e['id']) for e in data))
")
    while IFS= read -r pid; do
        JOBS+=("$f $pid")
    done <<< "$ids"
done

echo "Launching ${#JOBS[@]} puzzle jobs..."
START_TIME=$(date +%s)

# Launch all jobs via xargs (-n 2 avoids macOS -I size limit)
printf '%s\n' "${JOBS[@]}" | tr ' ' '\n' | xargs -P "$PROCS" -n 2 bash -c '
    python3 run_zebralogic.py '"$FLAG"' \
        --max_rounds '"$MAX_ROUNDS"' \
        --model '"$MODEL"' \
        --dataset "$0" \
        --puzzle_ids "$1" \
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

results_dir = Path('$RESULTS_DIR')
mode = '$MODE'
model = '$MODEL'
dataset_files = sorted(Path('$DATA_DIR').glob('puzzles_*.json'))

sizes_filter = set('${SIZES[*]:-}'.split()) if '${SIZES[*]:-}' else set()

print(f\"{'Size':<8} {'Solved':<12} {'Avg Tok':<10} {'Avg s/rnd':<10}\")
print('-' * 40)

grand_results = []

for df in dataset_files:
    size = df.stem.replace('puzzles_', '')
    if sizes_filter and size not in sizes_filter:
        continue

    jsonl = results_dir / f'zebralogic_{mode}_{model}_{df.stem}.jsonl'
    if not jsonl.exists():
        continue

    results = []
    for line in jsonl.read_text().strip().split('\n'):
        if line:
            results.append(json.loads(line))

    if not results:
        continue

    solved = sum(1 for r in results if r['solved'])
    avg_tok = sum(r['total_tokens'] for r in results) / len(results)
    total_rounds = sum(r['rounds'] for r in results)
    total_elapsed = sum(r.get('elapsed', 0) for r in results)
    avg_s_per_round = total_elapsed / total_rounds if total_rounds else 0

    print(f'{size:<8} {solved}/{len(results):<10} {avg_tok:<10.0f} {avg_s_per_round:<10.1f}')
    grand_results.extend(results)

if grand_results:
    total = len(grand_results)
    total_solved = sum(1 for r in grand_results if r['solved'])
    total_tokens = sum(r['total_tokens'] for r in grand_results)
    total_rounds = sum(r['rounds'] for r in grand_results)
    total_elapsed = sum(r.get('elapsed', 0) for r in grand_results)
    avg_s = total_elapsed / total_rounds if total_rounds else 0
    print(f\"{'='*40}\")
    print(f'Overall: {total_solved}/{total} ({100*total_solved/total:.0f}%)')
    print(f'Total tokens: {total_tokens}')
    print(f'Avg s/round: {avg_s:.1f}')
"
