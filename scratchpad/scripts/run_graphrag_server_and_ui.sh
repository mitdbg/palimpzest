#!/usr/bin/env zsh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

exec "$PY" "$REPO_ROOT/scratchpad/scripts/run_graphrag_server_and_ui.py"
