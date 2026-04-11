#!/usr/bin/env bash
# run_sweep.sh — Run a bench mode across multiple mask rates with auto-retry on zero-token failures.
#
# Usage:
#   bash run_sweep.sh <task> <mode> <max_rounds> <mask_rates...>
#
# Env vars (with defaults):
#   MODEL=gpt-none
#   N=100
#   PROCS=20
#   MAX_RETRIES=5   # max retry rounds per mask rate
#
# Examples:
#   bash run_sweep.sh sudoku baseline 1 0.3 0.4 0.5 0.6 0.7
#   MODEL=qwen-none PROCS=50 bash run_sweep.sh sudoku baseline 1 0.3 0.4 0.5
#   MODEL=gpt-none PROCS=10 bash run_sweep.sh sudoku gen-ver 5 0.6 0.7

set -euo pipefail

TASK="${1:?Usage: run_sweep.sh <task> <mode> <max_rounds> <mask_rates...>}"
MODE="${2:?}"
MAX_ROUNDS="${3:?}"
shift 3
MASK_RATES=("$@")

MODEL="${MODEL:-gpt-none}"
N="${N:-100}"
PROCS="${PROCS:-20}"
MAX_RETRIES="${MAX_RETRIES:-5}"

RESULTS_DIR="results"
TRACES_DIR="traces"

# Returns space-separated list of zero-token puzzle IDs for a given trace dir
find_zero_token_ids() {
    local trace_dir="$1"
    python3 - "$trace_dir" <<'EOF'
import sys, json, glob
trace_dir = sys.argv[1]
files = sorted(glob.glob(f"{trace_dir}/p*.jsonl"))
zero = []
for f in files:
    pid = f.split("/p")[-1].replace(".jsonl", "")
    rows = [json.loads(l) for l in open(f)]
    if sum(r.get("tokens_in", 0) + r.get("tokens_out", 0) for r in rows) == 0:
        zero.append(pid)
print(" ".join(zero))
EOF
}

# Remove rows for given IDs from results JSONL
remove_result_rows() {
    local results_file="$1"
    shift
    local ids=("$@")
    python3 - "$results_file" "${ids[@]}" <<'EOF'
import sys, json
path = sys.argv[1]
bad = set(int(x) for x in sys.argv[2:])
rows = [json.loads(l) for l in open(path)]
kept = [r for r in rows if r.get("id") not in bad]
with open(path, "w") as f:
    for r in kept:
        f.write(json.dumps(r) + "\n")
print(f"Removed {len(rows)-len(kept)} rows, {len(kept)} remaining")
EOF
}

# Run a single puzzle via run_{task}.py (no mode flag for gen-ver, --baseline etc for others)
run_puzzle() {
    local task="$1" mode="$2" max_rounds="$3" dataset="$4" pid="$5"
    local flag=""
    case "$mode" in
        baseline)         flag="--baseline" ;;
        cot)              flag="--cot" ;;
        self-consistency) flag="--self_consistency" ;;
        self-refine)      flag="--self_refine" ;;
        debate)           flag="--debate" ;;
        gen-ver)          flag="" ;;
    esac
    python3 "run_${task}.py" $flag \
        --max_rounds "$max_rounds" \
        --model "$MODEL" \
        --puzzle_ids "$pid" \
        --dataset "$dataset" \
        --log 2>&1 | tail -1
}
export -f run_puzzle
export MODEL

for mask in "${MASK_RATES[@]}"; do
    dataset="data/${TASK}/puzzles_${mask}.json"
    stem="puzzles_${mask}"
    trace_dir="${TRACES_DIR}/${TASK}_${MODE}_${MODEL}_${stem}"
    results_file="${RESULTS_DIR}/${TASK}_${MODE}_${MODEL}_${stem}.jsonl"

    echo "========================================"
    echo "Running: task=$TASK mode=$MODE model=$MODEL mask=$mask n=$N procs=$PROCS"
    echo "========================================"

    # Initial bench run (cleans results automatically)
    N=$N PROCS=$PROCS MODEL=$MODEL bash "bench_${TASK}.sh" "$MODE" "$MAX_ROUNDS" "$mask"

    # Retry loop
    for attempt in $(seq 1 "$MAX_RETRIES"); do
        zero_ids=$(find_zero_token_ids "$trace_dir")
        if [ -z "$zero_ids" ]; then
            echo "No zero-token failures. Done with mask=$mask."
            break
        fi

        count=$(echo "$zero_ids" | wc -w | tr -d ' ')
        echo "Attempt $attempt: $count zero-token failures: $zero_ids — retrying..."

        # Clean trace files
        for pid in $zero_ids; do
            rm -f "${trace_dir}/p${pid}.jsonl"
        done

        # Remove result rows
        remove_result_rows "$results_file" $zero_ids

        # Rerun in parallel (capped at PROCS)
        echo "$zero_ids" | tr ' ' '\n' | \
            xargs -P "$PROCS" -I {} bash -c 'run_puzzle "$@"' _ \
                "$TASK" "$MODE" "$MAX_ROUNDS" "$dataset" "{}"

        if [ "$attempt" -eq "$MAX_RETRIES" ]; then
            echo "WARNING: still have zero-token failures after $MAX_RETRIES attempts for mask=$mask"
        fi
    done

    # Final tally
    python3 - "$results_file" "$N" <<'EOF'
import sys, json
path, n = sys.argv[1], int(sys.argv[2])
rows = [json.loads(l) for l in open(path)]
solved = sum(1 for r in rows if r.get("solved"))
avg_tok = sum(r.get("total_tokens", 0) for r in rows) / len(rows) if rows else 0
print(f"FINAL: {solved}/{len(rows)} solved ({100*solved//len(rows) if rows else 0}%) | avg_tokens={avg_tok:.0f}")
EOF
    echo ""
done

echo "========================================"
echo "Sweep complete."
echo "========================================"
