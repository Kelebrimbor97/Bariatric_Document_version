# NOTE:
# This rebuilds corpus JSONL and Qdrant embeddings.
# Run this only when the source PDFs or indexing schema changed.
# This is not needed for normal API startup.

#!/usr/bin/env bash
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ehr_rag
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

export COLLECTION_NAME="${COLLECTION_NAME:-ehr_chunks_test}"
echo "Qdrant collection: $COLLECTION_NAME"

mkdir -p logs

echo "=== BUILD START $(date) ==="
PYTHONPATH=. python scripts/build_ehr_corpus.py | tee logs/build_ehr_corpus.out

echo "=== INDEX START $(date) ==="
PYTHONPATH=. python scripts/index_qdrant_medcpt.py | tee logs/index_qdrant_medcpt.out

echo "=== DONE $(date) ==="
