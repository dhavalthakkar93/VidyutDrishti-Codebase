#!/bin/bash
# VidyutDrishti — Start all laptop services for demo
# Run: bash start_demo.sh

echo "=================================="
echo "  VidyutDrishti — Starting Demo"
echo "=================================="

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 3
else
    echo "Ollama: already running"
fi

# Check if model is available
if ! ollama list | grep -q "qwen2.5:7b"; then
    echo "Pulling Qwen 2.5 7B model..."
    ollama pull qwen2.5:7b
fi
echo "LLM: ready"

# Kill any existing streamlit
pkill -f "streamlit run" 2>/dev/null

# Start dashboard
echo "Starting dashboard..."
cd "$(dirname "$0")/dashboard"
streamlit run app.py --server.headless true --server.port 8501 &

sleep 3
echo ""
echo "=================================="
echo "  Dashboard: http://localhost:8501"
echo "  Pi API:    http://192.168.29.61:5000"
echo "=================================="
echo ""
echo "Press Ctrl+C to stop"
wait
