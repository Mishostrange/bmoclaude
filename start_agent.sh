#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

# Add the project root to PYTHONPATH so that `import core`, `import audio`, etc. resolve correctly
export PYTHONPATH="$BASE_DIR:${PYTHONPATH:-}"

source venv/bin/activate
exec python agent.py
