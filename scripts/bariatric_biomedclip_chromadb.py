#!/usr/bin/env python3
"""
Build a ChromaDB vector DB from clinical notes using BioMedCLIP (OpenCLIP).

✅ Supports:
- Persistent ChromaDB at --persist_dir
- Collection name via --collection
- Token-aware chunking (chunk_tokens / overlap_tokens / min_chunk_tokens)
- Metadata includes: record_id, chunk_id, MRN (+ optional date/type)
- Safe column handling: respects your explicit --record_id_col/--mrn_col/--notes_col;
  if a provided column doesn't exist, it will fall back to guessing and FAIL EARLY if still missing.
- Uses GPU/CPU device via --device (e.g. cuda:0)

This script DOES NOT depend on NotesIndexer.upsert_record() signature (no more idx NameError).
It directly uses:
- BiomedCLIPTextEmbedder
- chunk_text_for_context_length
from your scripts/utils/big_chungus.py
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings

# IMPORTANT: this import assumes you run this from project root like:
#   python scripts/bariatric_biomedclip_chromadb.py ...
# and that "scripts/utils" is on PYTHONPATH.
#
# If you run from elsewhere and this import fails, run:
#   export PYTHONPATH=$PWD/scripts:$PYTHONPATH
#
from scripts.utils.big_chungus import (
    BiomedCLIPTextEmbedder,
    ChunkConfig,
    chunk_text_for_context_length,
)


def _read_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No header found in CSV: {csv_path}")
        return [c.strip() for c in reader.fieldnames]


def _guess_column(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    lower_to_actual = {c.lower(): c for c in fieldnames}
    for cand in candidates:
        c = lower_to_actual.get(cand.lower())
        if c:
            return c
    return None


def resolve_columns(
    csv_path: Path,
    record_id_col: str,
    mrn_col: str,
    notes_col: str,
) -> Tuple[str, str, str]:
    fieldnames = _read_header(csv_path)
    cols_set = set(fieldnames)

    def ensure_or_guess(name: str, guesses: List[str], label: str) -> str:
        if name and name in cols_set:
            return name
        guessed = _guess_column(fieldnames, guesses)
        if guessed:
            if name and name not in cols_set:
                print(f"[WARN] --{label} '{name}' not found in CSV; using '{guessed}' instead.")
            return guessed
        return name  # will be validated below

    rec = ensure_or_guess(
        record_id_col,
        ["record_id", "recordid", "note_id", "noteid", "encounter_id", "visit_id", "id"],
        "record_id_col",
    )
    mrn = ensure_or_guess(
        mrn_col,
        ["mrn", "MRN", "patient_id", "patientid", "subject_id", "subjectid"],
        "mrn_col",
    )
    notes = ensure_or_guess(
        notes_col,
        ["note_text", "text", "notes", "note", "clinical_notes", "content", "document"],
        "notes_col",
    )

    missing = [c for c in [rec, mrn, notes] if c not in cols_set]
    if missing:
        raise ValueError(
            f"Missing required columns in CSV: {missing}\n"
            f"Available columns: {fieldnames}\n"
            f"Pass --record_id_col / --mrn_col / --notes_col correctly."
        )

    return rec, mrn, notes


def main() -> None:
    ap = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parent.parent

    # IO
    ap.add_argument("--csv", type=Path, default=Path("./Data/MBS_clinical_notes.csv"))
    ap.add_argument("--persist_dir", type=Path, default=Path("./Data/chroma_dbs/chroma_notes_biomedclip"))
    ap.add_argument("--collection", type=str, default="notes_chunks_mrn")

    # CSV columns
    ap.add_argument("--record_id_col", type=str, default="record_id")
    ap.add_argument("--mrn_col", type=str, default="mrn")
    ap.add_argument("--notes_col", type=str, default="note_text")
    ap.add_argument("--date_col", type=str, default="", help="Optional date column to store in metadata as note_date")
    ap.add_argument("--note_type_col", type=str, default="", help="Optional note type column to store in metadata as note_type")

    # Chunking
    ap.add_argument("--chunk_tokens", type=int, default=200)
    ap.add_argument("--overlap_tokens", type=int, default=40)
    ap.add_argument("--min_chunk_tokens", type=int, default=60)

    # Embedder
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument(
        "--ckpt_dir",
        type=Path,
        default=Path(
            os.getenv(
                "BIOMEDCLIP_CKPT_DIR",
                str(
                    project_root.parent
                    / "LLM_Weights"
                    / "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                ),
            )
        ),
    )
    ap.add_argument("--model_name", type=str, default="biomedclip_local")

    # Loop control
    ap.add_argument("--start_row", type=int, default=0)
    ap.add_argument("--max_rows", type=int, default=0, help="0 means no limit")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--dry_run", action="store_true")

    # Behavior
    ap.add_argument("--skip_existing", action="store_true", help="Skip record_id if it already exists in the collection")

    args = ap.parse_args()

    csv_path: Path = args.csv
    persist_dir: Path = args.persist_dir

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    persist_dir.mkdir(parents=True, exist_ok=True)

    # Resolve columns safely
    record_id_col, mrn_col, notes_col = resolve_columns(
        csv_path=csv_path,
        record_id_col=args.record_id_col,
        mrn_col=args.mrn_col,
        notes_col=args.notes_col,
    )

    # Validate optional columns
    fieldnames = _read_header(csv_path)
    cols_set = set(fieldnames)
    if args.date_col and args.date_col not in cols_set:
        print(f"[WARN] --date_col '{args.date_col}' not found in CSV. Ignoring.")
        args.date_col = ""
    if args.note_type_col and args.note_type_col not in cols_set:
        print(f"[WARN] --note_type_col '{args.note_type_col}' not found in CSV. Ignoring.")
        args.note_type_col = ""

    print(f"[INFO] CSV: {csv_path}")
    print(f"[INFO] Using record_id_col='{record_id_col}', mrn_col='{mrn_col}', notes_col='{notes_col}'")
    print(f"[INFO] Chroma persist_dir: {persist_dir}")
    print(f"[INFO] Collection: {args.collection}")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] BiomedCLIP ckpt_dir: {args.ckpt_dir}")
    print(f"[INFO] dry_run={args.dry_run} skip_existing={args.skip_existing}")

    # Chroma client + collection
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(args.collection)

    # Embedder
    embedder = BiomedCLIPTextEmbedder(
        device=args.device,
        ckpt_dir=str(args.ckpt_dir),
        model_name=args.model_name,
    )

    cfg = ChunkConfig(
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        min_chunk_tokens=args.min_chunk_tokens,
    )

    # Counters
    processed = 0
    skipped_empty_notes = 0
    skipped_missing_record_id = 0
    skipped_existing = 0
    total_chunks_written = 0

    # Helper: does a record already exist?
    def record_exists(rec_id: str) -> bool:
        res = collection.get(where={"record_id": rec_id}, limit=1, include=[])
        return bool(res.get("ids"))

    # Main loop
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        # prevent csv field size errors on very large notes
        try:
            csv.field_size_limit(1024 * 1024 * 50)  # 50MB per field
        except Exception:
            pass

        reader = csv.DictReader(f)

        for row_i, row in enumerate(reader):
            if row_i < args.start_row:
                continue
            if args.max_rows and processed >= args.max_rows:
                break

            rec_id = (row.get(record_id_col) or "").strip()
            if not rec_id:
                skipped_missing_record_id += 1
                continue

            notes = (row.get(notes_col) or "").strip()
            if not notes:
                skipped_empty_notes += 1
                continue

            if args.skip_existing and record_exists(rec_id):
                skipped_existing += 1
                if skipped_existing % 1000 == 0:
                    print(f"[INFO] skipped_existing={skipped_existing}")
                continue

            mrn = (row.get(mrn_col) or "").strip()

            # Chunk the note
            chunks = chunk_text_for_context_length(notes, embedder.hf_tokenizer, cfg)
            if not chunks:
                skipped_empty_notes += 1
                continue

            # Prepare ids + metadata (MRN included!)
            ids = [f"{rec_id}::c{i:05d}" for i in range(len(chunks))]

            base_meta: Dict[str, object] = {"record_id": rec_id, "chunk_id": None, "MRN": mrn}

            if args.date_col:
                base_meta["note_date"] = (row.get(args.date_col) or "").strip()
            if args.note_type_col:
                base_meta["note_type"] = (row.get(args.note_type_col) or "").strip()

            metadatas = []
            for i in range(len(chunks)):
                md = dict(base_meta)
                md["chunk_id"] = i
                metadatas.append(md)

            if args.dry_run:
                processed += 1
                if processed % args.log_every == 0:
                    tok_est = len(embedder.hf_tokenizer.encode(notes, add_special_tokens=False))
                    print(f"[DRY] processed={processed} rec_id={rec_id} chunks={len(chunks)} note_tokens≈{tok_est}")
                continue

            # Embed + upsert
            embeddings = embedder.embed_texts(chunks)
            collection.upsert(
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
                embeddings=embeddings,
            )

            processed += 1
            total_chunks_written += len(chunks)

            if processed % args.log_every == 0:
                print(
                    f"[INFO] processed={processed} total_chunks={total_chunks_written} "
                    f"last_rec_id={rec_id} last_chunks={len(chunks)}"
                )

    print("\n[DONE]")
    print(f"processed_rows={processed}")
    print(f"skipped_missing_record_id={skipped_missing_record_id}")
    print(f"skipped_empty_notes={skipped_empty_notes}")
    print(f"skipped_existing={skipped_existing}")
    print(f"total_chunks_written={total_chunks_written}")
    print(f"chroma_db={persist_dir}")
    print(f"collection={args.collection}")

    # Quick sanity peek (if collection non-empty)
    try:
        peek = collection.peek(1)
        if peek and peek.get("metadatas"):
            print(f"[SANITY] metadatas[0] = {peek['metadatas'][0]}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
