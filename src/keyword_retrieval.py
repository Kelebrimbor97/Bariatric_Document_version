from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.config import PROCESSED_DIR


TOKEN_RE = re.compile(r"[A-Za-z0-9_./+-]+")


@dataclass
class KeywordHit:
    score: float
    record: dict[str, Any]


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def load_chunk_records(chunks_path: Path | None = None) -> list[dict[str, Any]]:
    path = chunks_path or (PROCESSED_DIR / "chunks.jsonl")
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


class KeywordRetriever:
    """Lightweight BM25 retriever over processed chunk JSONL records."""

    def __init__(self, records: list[dict[str, Any]]):
        self.records = records
        tokenized = [self._record_tokens(r) for r in records]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def _record_tokens(self, record: dict[str, Any]) -> list[str]:
        fields = [
            record.get("chunk_text") or "",
            record.get("document_type") or "",
            record.get("section_title") or "",
            record.get("relative_path") or "",
            record.get("file_name") or "",
        ]
        return tokenize("\n".join(str(x) for x in fields if x is not None))

    def search(
        self,
        query: str,
        patient_id: str | None = None,
        document_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[KeywordHit]:
        if not self.records or self.bm25 is None:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        allowed_doc_types = set(document_types or [])
        scores = self.bm25.get_scores(query_tokens)

        hits: list[KeywordHit] = []
        for score, record in zip(scores, self.records):
            if score <= 0:
                continue

            if patient_id and not record_matches_patient(record, patient_id):
                continue

            if allowed_doc_types:
                record_doc_type = record.get("document_type") or "unknown"
                if record_doc_type not in allowed_doc_types:
                    continue

            hits.append(KeywordHit(score=float(score), record=record))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]


def record_matches_patient(record: dict[str, Any], patient_id: str) -> bool:
    if not patient_id:
        return True

    patient_id = patient_id.strip()
    return patient_id in {
        str(record.get("patient_id") or ""),
        str(record.get("actual_patient_id") or ""),
        str(record.get("patient_folder_name") or ""),
    }


@lru_cache(maxsize=1)
def get_keyword_retriever() -> KeywordRetriever:
    return KeywordRetriever(load_chunk_records())
