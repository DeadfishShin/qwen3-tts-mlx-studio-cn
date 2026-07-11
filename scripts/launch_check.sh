#!/bin/bash
# Start the app on a scratch port, wait for the UI to answer, then shut down.
set -e
cd "$(dirname "$0")/.."
.venv/bin/python app.py --port 7897 &
PID=$!
trap 'kill $PID 2>/dev/null' EXIT
for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:7897/ >/dev/null 2>&1; then
    echo "LAUNCH OK"
    exit 0
  fi
  sleep 1
done
echo "LAUNCH FAILED"
exit 1
