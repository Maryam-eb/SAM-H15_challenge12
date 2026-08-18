#!/usr/bin/env bash
cd "$(dirname "$0")"
echo "Starting VisionVerse AI on http://localhost:8000"
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
