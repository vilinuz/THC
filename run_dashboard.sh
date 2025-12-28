#!/bin/bash
# Start the Crypto Sentiment Dashboard

# Ensure we are in the project root
cd "$(dirname "$0")"

echo "Starting Crypto Sentiment Dashboard..."
echo "Access the dashboard at: http://localhost:8000/static/index.html"

# Run via main.py in dashboard mode using local virtual environment
# Check for venv, create if missing
if [ ! -f ".venv/bin/python3" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Run
./.venv/bin/python3 main.py --mode dashboard
