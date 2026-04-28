from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import hashlib
from tqdm import tqdm

from src.config import (
    PROCESSED_DIR,
    QDRANT_URL,
    COLLECTION_NAME,
    MEDCPT_ARTICLE_MODEL,
)
from src.medcpt_embed import MedCPTEncoder
from src.qdrant_store import get_client, ensure_collection, upsert_points


CHECKPOINT_FILE = PROCESSED_DIR / "index_qdrant_medcpt.ckpt"


def load_checkpoint() -> int:
    if CHECKPOINT_FILE.exists():
        txt = CHECKPOINT_FILE.read_text(encoding="utf-8").strip()
        if txt:
            return int(txt)
    return 0


def save_checkpoint(idx: int) -> None:
    CHECKPOINT_FILE.write_text(str(idx), encoding="utf-8")


def stable_point_id(chunk_id: str) -> int:
    return int(hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[:16], 16)


def main():
    chunks_file = PROCESSED_DIR / "chunks.jsonl"
    if not chunks_file.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_file}")

    records = []
    with chunks_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("No records found in chunks.jsonl")
        return

    encoder = MedCPTEncoder(MEDCPT_ARTICLE_MODEL, device="cuda:1")
    client = get_client(QDRANT_URL)

    sample_vec = encoder.encode(["test embedding"])
    vector_size = int(sample_vec.shape[1])
    ensure_collection(client, COLLECTION_NAME, vector_size)

    batch_size = 32
    start_idx = load_checkpoint()

    if start_idx >= len(records):
        print(
            f"Checkpoint already at end ({start_idx}/{len(records)}). "
            f"Nothing to do."
        )
        return

    print(f"Loaded {len(records)} chunk records.")
    print(f"Resuming from index {start_idx}.")
    print(f"Checkpoint file: {CHECKPOINT_FILE}")

    for start in tqdm(range(start_idx, len(records), batch_size), desc="Indexing"):
        batch = records[start:start + batch_size]
        texts = [r["chunk_text"] for r in batch]
        vecs = encoder.encode(texts)

        ids = [stable_point_id(r["chunk_id"]) for r in batch]

        payloads = []
        for r in batch:
            payloads.append(
                {
                    "patient_id": r["patient_id"],
                    "actual_patient_id": r.get("actual_patient_id"),
                    "patient_folder_name": r.get("patient_folder_name"),
                    "patient_name_raw": r.get("patient_name_raw"),
                    "patient_name_normalized": r.get("patient_name_normalized"),
                    "patient_name_confidence": r.get("patient_name_confidence"),
                    "patient_name_reason": r.get("patient_name_reason"),
                    "relative_path": r["relative_path"],
                    "path_parts": r["path_parts"],
                    "file_name": r["file_name"],
                    "path_tags": r["path_tags"],
                    "page_num": r["page_num"],
                    "chunk_id": r["chunk_id"],
                    "chunk_text": r["chunk_text"],
                }
            )

        upsert_points(client, COLLECTION_NAME, ids, vecs, payloads)
        save_checkpoint(start + len(batch))

    print("Indexing complete.")
    print(f"Final checkpoint: {len(records)}")


if __name__ == "__main__":
    main()