#!/usr/bin/env bash
# Stop a backgrounded vLLM server (serve.sh) on :8000. Ctrl-C already stops a foreground one.
set -uo pipefail
PORT="${VLLM_PORT:-8000}"

pids_on_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti :"$port" 2>/dev/null
  elif command -v fuser >/dev/null 2>&1; then
    fuser "$port"/tcp 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$'
  else
    echo "Neither lsof nor fuser found — cannot look up the process on :$port." >&2
  fi
}

pids="$(pids_on_port "$PORT" | sort -u)"
if [ -n "$pids" ]; then
  echo "$pids" | xargs kill 2>/dev/null && echo "Stopped vLLM server on :$PORT"
else
  echo "Nothing running on :$PORT."
fi
