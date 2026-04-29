from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.ehr_rag_service import answer_question
from src.config import PROCESSED_DIR


app = FastAPI(title="EHR RAG API", version="0.3.0")


class AskRequest(BaseModel):
    patient_id: str | None = None
    question: str
    structured: bool = False


class SourceItem(BaseModel):
    relative_path: str | None = None
    page_num: int | None = None
    chunk_id: str | None = None
    document_type: str | None = None
    section_title: str | None = None
    rerank_score: float | None = None
    retrieval_source: str | None = None


class RetrievalPlanItem(BaseModel):
    primary_query: str | None = None
    subqueries: list[str] = Field(default_factory=list)
    target_document_types: list[str] = Field(default_factory=list)
    rationale: str | None = None


class AskResponse(BaseModel):
    patient_id: str | None = None
    answer: str
    structured_answer: dict[str, Any] | None = None
    sources: list[SourceItem] = Field(default_factory=list)
    retrieval_plan: RetrievalPlanItem | dict[str, Any] | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/patients")
def list_patients():
    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        return {"patients": []}

    seen = {}
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pid = obj.get("patient_id")
            if pid and pid not in seen:
                seen[pid] = {
                    "patient_id": pid,
                    "actual_patient_id": obj.get("actual_patient_id"),
                    "patient_folder_name": obj.get("patient_folder_name"),
                }

    return {"patients": list(seen.values())}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    result = answer_question(req.patient_id, req.question, structured=req.structured)

    if isinstance(result, dict):
        return AskResponse(
            patient_id=req.patient_id,
            answer=result.get("answer", ""),
            structured_answer=result.get("structured_answer"),
            sources=result.get("sources", []),
            retrieval_plan=result.get("retrieval_plan"),
        )

    return AskResponse(
        patient_id=req.patient_id,
        answer=str(result),
        structured_answer=None,
        sources=[],
        retrieval_plan=None,
    )
