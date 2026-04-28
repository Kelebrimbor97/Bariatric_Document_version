#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="ehr_rag"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-qwen-ehr}"
ENCODER_API_URL="${ENCODER_API_URL:-http://localhost:8092}"
LLM_WEIGHTS_DIR="${LLM_WEIGHTS_DIR:-$HOME/LLM_Weights}"
BIOMEDCLIP_CKPT_DIR="${BIOMEDCLIP_CKPT_DIR:-$LLM_WEIGHTS_DIR/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224}"
BIOMEDBERT_TOKENIZER_PATH="${BIOMEDBERT_TOKENIZER_PATH:-$LLM_WEIGHTS_DIR/BiomedNLP-BiomedBERT-base-uncased-abstract}"

cd "$PROJECT_DIR"
mkdir -p logs

echo "=== Activating conda env: $CONDA_ENV ==="
source "$CONDA_SH"
conda activate "$CONDA_ENV"

echo "=== Stopping local API services only ==="
pkill -f "uvicorn api_encoder:app" || true
pkill -f "uvicorn api_ehr_rag:app" || true
pkill -f "uvicorn api_literature_rag:app" || true
sleep 3

echo "=== Confirm remaining uvicorn processes ==="
ps -ef | grep uvicorn | grep -v grep || true

echo "=== Starting Encoder API on 8092 ==="
nohup env PYTHONPATH=. uvicorn api_encoder:app --host 0.0.0.0 --port 8092 --workers 1 \
  > logs/api_encoder.log 2>&1 &

sleep 8
curl -fsS "${ENCODER_API_URL%/}/health" >/dev/null
echo "Encoder API OK"

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
curl -fsS http://localhost:8090/health >/dev/null
echo "EHR API OK"

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
curl -fsS http://localhost:8093/health >/dev/null
echo "Literature API OK"

echo "=== Local API restart complete ==="
echo "Docker services were not touched."
echo ""
echo "Health checks:"
echo "  Encoder:    http://localhost:8092/health"
echo "  EHR API:    http://localhost:8090/health"
echo "  Literature: http://localhost:8093/health"
