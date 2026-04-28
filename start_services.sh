#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="ehr_rag"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-qwen-ehr}"
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/llm_weights/Qwen3.6-35B-A3B}"
ENCODER_API_URL="${ENCODER_API_URL:-http://localhost:8092}"
LLM_WEIGHTS_DIR="${LLM_WEIGHTS_DIR:-$HOME/LLM_Weights}"
BIOMEDCLIP_CKPT_DIR="${BIOMEDCLIP_CKPT_DIR:-$LLM_WEIGHTS_DIR/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224}"
BIOMEDBERT_TOKENIZER_PATH="${BIOMEDBERT_TOKENIZER_PATH:-$LLM_WEIGHTS_DIR/BiomedNLP-BiomedBERT-base-uncased-abstract}"

cd "$PROJECT_DIR"
mkdir -p logs

echo "=== Activating conda env: $CONDA_ENV ==="
source "$CONDA_SH"
conda activate "$CONDA_ENV"

echo "=== Confirming Python environment ==="
echo "Python: $(which python)"
python - <<'PY'
import sys
print("Executable:", sys.executable)
PY

echo "=== Starting Docker services: Qdrant + vLLM ==="
export LLM_WEIGHTS_DIR VLLM_MODEL_NAME VLLM_MODEL_PATH
docker compose up -d qdrant vllm_qwen

echo "=== Checking Qdrant ==="
curl -fsS "$QDRANT_URL" >/dev/null
echo "Qdrant OK"

echo "=== Checking vLLM ==="
curl -fsS "${VLLM_BASE_URL%/}/models" >/dev/null
echo "vLLM OK"

echo "=== Stopping old API services if present ==="
pkill -f "uvicorn api_encoder:app" || true
pkill -f "uvicorn api_ehr_rag:app" || true
pkill -f "uvicorn api_literature_rag:app" || true
sleep 2

echo "=== Starting Encoder API on 8092 ==="
nohup env PYTHONPATH=. uvicorn api_encoder:app --host 0.0.0.0 --port 8092 --workers 1 \
  > logs/api_encoder.log 2>&1 &

sleep 8
if curl -fsS "${ENCODER_API_URL%/}/health" >/dev/null; then
  echo "Encoder API OK"
else
  echo "Encoder API FAILED. Last logs:"
  tail -n 100 logs/api_encoder.log
  exit 1
fi

echo "=== Starting EHR RAG API on 8090 ==="
nohup env \
  PYTHONPATH=. \
  VLLM_BASE_URL="$VLLM_BASE_URL" \
  VLLM_MODEL_NAME="$VLLM_MODEL_NAME" \
  QDRANT_URL="$QDRANT_URL" \
  ENCODER_API_URL="$ENCODER_API_URL" \
  BIOMEDCLIP_CKPT_DIR="$BIOMEDCLIP_CKPT_DIR" \
  BIOMEDBERT_TOKENIZER_PATH="$BIOMEDBERT_TOKENIZER_PATH" \
  uvicorn api_ehr_rag:app --host 0.0.0.0 --port 8090 --workers 1 \
  > logs/api_ehr_rag.log 2>&1 &

sleep 8
if curl -fsS http://localhost:8090/health >/dev/null; then
  echo "EHR API OK"
else
  echo "EHR API FAILED. Last logs:"
  tail -n 100 logs/api_ehr_rag.log
  exit 1
fi

echo "=== Starting Literature Approval RAG API on 8093 ==="
nohup env \
  PYTHONPATH=. \
  VLLM_BASE_URL="$VLLM_BASE_URL" \
  VLLM_MODEL_NAME="$VLLM_MODEL_NAME" \
  ENCODER_API_URL="$ENCODER_API_URL" \
  BIOMEDCLIP_CKPT_DIR="$BIOMEDCLIP_CKPT_DIR" \
  BIOMEDBERT_TOKENIZER_PATH="$BIOMEDBERT_TOKENIZER_PATH" \
  uvicorn api_literature_rag:app --host 0.0.0.0 --port 8093 --workers 1 \
  > logs/api_literature_rag.log 2>&1 &

sleep 8
if curl -fsS http://localhost:8093/health >/dev/null; then
  echo "Literature API OK"
else
  echo "Literature API FAILED. Last logs:"
  tail -n 100 logs/api_literature_rag.log
  exit 1
fi

echo "=== All services started ==="
echo "Logs:"
echo "  logs/api_encoder.log"
echo "  logs/api_ehr_rag.log"
echo "  logs/api_literature_rag.log"
echo ""
echo "Ports:"
echo "  Qdrant:         http://localhost:6333"
echo "  vLLM:           http://localhost:8000/v1/models"
echo "  EHR API:        http://localhost:8090"
echo "  Encoder API:    http://localhost:8092"
echo "  Literature API: http://localhost:8093"
