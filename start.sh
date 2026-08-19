#!/bin/bash

# Llama Studio Startup Script
# This script activates the Python virtual environment and starts the application

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🦙 Starting Llama Studio..."
echo ""

# Check if venv exists
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "❌ Virtual environment not found at $SCRIPT_DIR/venv"
    echo ""
    echo "To set up the environment, run:"
    echo "  python3 -m venv $SCRIPT_DIR/venv"
    echo "  source $SCRIPT_DIR/venv/bin/activate"
    echo "  pip install -r $SCRIPT_DIR/requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source "$SCRIPT_DIR/venv/bin/activate"

# Change to backend directory
cd "$SCRIPT_DIR/backend"

# Get configuration
if [ -f "$SCRIPT_DIR/config/app.json" ]; then
    PORT=$(grep -o '"webui_port"[[:space:]]*:[[:space:]]*[0-9]*' "$SCRIPT_DIR/config/app.json" | grep -o '[0-9]*')
    PORT=${PORT:-7999}
    echo "⚙️  WebUI Port: $PORT"
else
    # First run: prompt user for port
    echo "⚠️  First run detected. No config file found."
    echo ""
    read -p "Enter WebUI port (default: 7999): " USER_PORT
    PORT=${USER_PORT:-7999}

    # Validate port is a number
    if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
        echo "❌ Invalid port. Using default 7999"
        PORT=7999
    fi

    echo "✓ Port set to $PORT (you can change this later in the Settings modal)"
fi

echo ""
echo "🚀 Starting FastAPI server..."
echo "   URL: http://localhost:${PORT:-7999}"
echo "   Press Ctrl+C to stop"
echo ""

# Check for verbose flag
VERBOSE_FLAG=""
if [[ "$1" == "-v" ]] || [[ "$1" == "--verbose" ]]; then
    VERBOSE_FLAG="--verbose"
    echo "🔍 Verbose logging enabled"
    echo ""
fi

# Start uvicorn (pass VERBOSE_FLAG as environment variable, not as argument).
# Only set VERBOSE when it's actually on: an empty VERBOSE is still an exported
# variable, and anything we spawn inherits it. vLLM reads its own VERBOSE as
# an int, so an inherited empty value crashes it.
if [ -n "$VERBOSE_FLAG" ]; then
    VERBOSE=$VERBOSE_FLAG python -m uvicorn main:app --reload --host 0.0.0.0 --port "${PORT:-7999}"
else
    python -m uvicorn main:app --reload --host 0.0.0.0 --port "${PORT:-7999}"
fi
