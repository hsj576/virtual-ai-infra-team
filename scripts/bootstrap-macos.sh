#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="$ROOT/.venv"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  print -u2 "This v0.1 developer preview requires an Apple Silicon Mac."
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required")
print("Python", sys.version.split()[0])
PY

MEMORY_BYTES="$(sysctl -n hw.memsize)"
MEMORY_GB="$((MEMORY_BYTES / 1024 / 1024 / 1024))"
if (( MEMORY_GB < 32 )); then
  print -u2 "Warning: ${MEMORY_GB} GB unified memory detected; the 27B reference path recommends at least 32 GB."
fi

FREE_KB="$(df -Pk "$ROOT" | tail -1 | tr -s ' ' | cut -d ' ' -f4)"
FREE_GB="$((FREE_KB / 1024 / 1024))"
if (( FREE_GB < 25 )); then
  print -u2 "At least 25 GB free disk is recommended; ${FREE_GB} GB detected."
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/pip" install -e "$ROOT"
mkdir -p "$ROOT/.infra-team/config"
cp "$ROOT/configs/autonomy-local.yaml" "$ROOT/.infra-team/config/autonomy-local.yaml"

print ""
print "Installation complete."
print "Next: $ROOT/scripts/start-local-ai.sh"
print "The first model download is about 15 GB. No launchd task was installed."
