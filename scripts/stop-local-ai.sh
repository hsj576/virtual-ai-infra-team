#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLI="$ROOT/.venv/bin/infra-team"
DASHBOARD_PID="$ROOT/.infra-team/dashboard/dashboard.pid"

if [[ -f "$DASHBOARD_PID" ]]; then
  PID="$(tr -dc '0-9' < "$DASHBOARD_PID")"
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    COMMAND="$(ps -p "$PID" -o command= 2>/dev/null || true)"
    if [[ "$COMMAND" == *"infra_team.cli"*"dashboard"*"--root $ROOT"* ]]; then
      kill "$PID"
    else
      print -u2 "Refusing to stop PID $PID because it is not this repository's Dashboard."
    fi
  fi
  rm -f "$DASHBOARD_PID"
fi

if [[ -x "$CLI" ]]; then
  "$CLI" --root "$ROOT" serve stop --name default || true
fi

print "Dashboard stop requested and managed model service stopped."
print "Verify ports with: lsof -nP -iTCP:8000 -sTCP:LISTEN; lsof -nP -iTCP:9000 -sTCP:LISTEN"
