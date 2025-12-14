#!/bin/bash
# Start the Crypto Sentiment Dashboard

# Ensure we are in the project root
cd "$(dirname "$0")"

echo "Starting Crypto Sentiment Dashboard..."
echo "Access the dashboard at: http://localhost:8000/static/index.html"

# Run via main.py in dashboard mode using local virtual environment
if [ -f ".venv/bin/python3" ]; then
    ./.venv/bin/python3 main.py --mode dashboard
else
    echo "Virtual environment not found. Installing dependencies..."
    python3 -m venv .venv
    ./.venv/bin/pip install -r requirements.txt
    ./.venv/bin/python3 main.py --mode dashboard
fi
