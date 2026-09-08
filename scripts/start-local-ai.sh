#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLI="$ROOT/.venv/bin/infra-team"
DASHBOARD_DIR="$ROOT/.infra-team/dashboard"
DASHBOARD_PID="$DASHBOARD_DIR/dashboard.pid"
DASHBOARD_LOG="$DASHBOARD_DIR/dashboard.log"

if [[ ! -x "$CLI" ]]; then
  print -u2 "Run scripts/bootstrap-macos.sh first."
  exit 1
fi

if "$CLI" --root "$ROOT" serve status --name default --json >/dev/null 2>&1; then
  print "Managed model service is already healthy."
else
  "$CLI" --root "$ROOT" serve start --name default
fi
mkdir -p "$DASHBOARD_DIR"

if lsof -nP -iTCP:9000 -sTCP:LISTEN >/dev/null 2>&1; then
  print -u2 "Port 9000 is already in use. The model service is running, but Dashboard was not started."
  exit 1
fi

nohup "$CLI" --root "$ROOT" dashboard --service-name default --host 127.0.0.1 --port 9000 >"$DASHBOARD_LOG" 2>&1 &
DASHBOARD_PROCESS=$!
print "$DASHBOARD_PROCESS" > "$DASHBOARD_PID"

print ""
print "Local AI is starting:"
print "  API:       http://127.0.0.1:8000/v1"
print "  Dashboard: http://127.0.0.1:9000"
print "  Log:       $DASHBOARD_LOG"
print ""
print "No Watch or launchd task was installed automatically."
