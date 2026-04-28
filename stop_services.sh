#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$PROJECT_DIR"

echo "=== Stopping local API services only ==="

pkill -f "uvicorn api_encoder:app" || true
pkill -f "uvicorn api_ehr_rag:app" || true
pkill -f "uvicorn api_literature_rag:app" || true

sleep 3

echo "=== Remaining uvicorn processes ==="
ps -ef | grep uvicorn | grep -v grep || true

echo ""
echo "NOTE:"
echo "OpenWebUI on port 8080 is intentionally left running."
echo "Docker services are intentionally left running:"
echo "  - Qdrant"
echo "  - vLLM/Qwen"
echo ""
echo "To also stop Docker services manually, run:"
echo "  docker compose down"
echo ""
echo "=== Local API shutdown complete ==="
