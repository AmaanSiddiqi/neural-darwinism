#!/usr/bin/env bash
# Starts the colony server persistently in a tmux session.
# Safe to run multiple times — kills existing session first.
#
# Usage:
#   ./start.sh          # start (or restart) the server
#   tmux attach -t colony   # reattach to see live logs
#   tmux kill-session -t colony   # stop the server

SESSION="colony"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Kill existing session if running
tmux kill-session -t "$SESSION" 2>/dev/null

tmux new-session -d -s "$SESSION" -x 220 -y 50
tmux send-keys -t "$SESSION" "cd '$DIR' && source .venv/bin/activate && python3 serve.py" Enter

echo "Colony server starting in tmux session '$SESSION'."
echo ""
echo "  View logs:  tmux attach -t $SESSION"
echo "  Detach:     Ctrl+B then D"
echo "  Stop:       tmux kill-session -t $SESSION"
echo "  URL:        http://localhost:8000"
