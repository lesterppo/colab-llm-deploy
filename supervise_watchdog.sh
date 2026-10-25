#!/bin/bash
# Watchdog for supervise.py (used by the supervisor-watchdog cron).
# Idempotent: if a supervisor for this model is already running, do
# nothing. If not, restart it detached so it survives the cron run
# itself. Never starts a second copy: pgrep matches the full command
# line, including --model, so a supervisor for another model is not
# mistaken for this one.
#
# Usage: supervise_watchdog.sh [model] [gpu]
#   Defaults: model=nemotron-3.5-bolt-e4b-q4, gpu=T4
set -u
MODEL="${1:-nemotron-3.5-bolt-e4b-q4}"
GPU="${2:-T4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/supervisor.log"
PATTERN="supervise.py --model $MODEL"

if pgrep -f "$PATTERN" >/dev/null 2>&1; then
  echo "supervisor already running for $MODEL; nothing to do."
  exit 0
fi
echo "supervisor NOT running for $MODEL; restarting..."
cd "$HERE" || exit 1
# setsid + nohup: detached from the cron agent's process group.
setsid nohup python3 "$HERE/supervise.py" --model "$MODEL" --gpu "$GPU" \
  --check-interval 300 --max-restarts 3 >>"$LOG" 2>&1 < /dev/null &
sleep 5
if pgrep -f "$PATTERN" >/dev/null 2>&1; then
  echo "supervisor restarted for $MODEL."
  exit 0
else
  echo "ERROR: supervisor failed to restart for $MODEL; see $LOG" >&2
  exit 1
fi
