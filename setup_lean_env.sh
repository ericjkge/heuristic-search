#!/usr/bin/env bash
# One-time Lean 4 workspace setup with Mathlib.
#
# Prerequisites: elan + lean installed (~/.elan/bin/)
# Usage: bash setup_lean_env.sh
#
# Creates lean_workspace/ with Mathlib dependency for proof checking via:
#   lake env lean <file.lean>

set -euo pipefail

WORKSPACE="lean_workspace"

if [ -d "$WORKSPACE" ]; then
    echo "Workspace '$WORKSPACE' already exists. Delete it first to re-create."
    exit 1
fi

export PATH="$HOME/.elan/bin:$PATH"
if ! command -v lake &> /dev/null; then
    echo "Error: 'lake' not found. Install elan first: https://github.com/leanprover/elan"
    exit 1
fi

echo "=== Creating Lean workspace ==="
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"
lake init . math

echo "=== Downloading pre-built Mathlib oleans ==="
lake exe cache get

echo "=== Building workspace ==="
lake build

echo
echo "=== Setup complete ==="
echo "Workspace: $(pwd)"
