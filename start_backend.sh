#!/bin/bash
echo "Starting FastAPI Backend..."
echo ""
echo "Make sure you have activated the virtual environment first:"
echo "  source .venv/bin/activate"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
uvicorn app.api:app --reload --port 8000
