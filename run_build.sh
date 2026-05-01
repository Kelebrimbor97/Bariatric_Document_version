# NOTE:
# This rebuilds corpus JSONL and Qdrant embeddings.
# Run this only when the source PDFs or indexing schema changed.
# This is not needed for normal API startup.
#
# Optional:
#   CLEAN_BUILD=1 ./run_build.sh
# removes generated JSONL/checkpoints/errors inside PROCESSED_DIR before rebuilding.

#!/usr/bin/env bash
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ehr_rag
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

export COLLECTION_NAME="${COLLECTION_NAME:-ehr_chunks_test}"
echo "Qdrant collection: $COLLECTION_NAME"
echo "Processed dir: ${PROCESSED_DIR:-$PROJECT_DIR/Data/processed}"

mkdir -p logs

BUILD_ARGS=()
if [[ "${CLEAN_BUILD:-0}" == "1" || "${CLEAN_BUILD:-false}" == "true" ]]; then
  BUILD_ARGS+=("--clean")
  echo "Clean build: enabled"
else
  echo "Clean build: disabled"
fi

echo "=== BUILD START $(date) ==="
PYTHONPATH=. python scripts/build_ehr_corpus.py "${BUILD_ARGS[@]}" | tee logs/build_ehr_corpus.out

echo "=== INDEX START $(date) ==="
PYTHONPATH=. python scripts/index_qdrant_medcpt.py | tee logs/index_qdrant_medcpt.out

echo "=== DONE $(date) ==="
