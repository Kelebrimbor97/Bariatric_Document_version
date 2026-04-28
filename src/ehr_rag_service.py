import os
from functools import lru_cache

import torch
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import (
    QDRANT_URL,
    COLLECTION_NAME,
    MEDCPT_QUERY_MODEL,
    MEDCPT_RERANK_MODEL,
    VLLM_BASE_URL,
    VLLM_MODEL_NAME,
)
from src.medcpt_embed import MedCPTEncoder
from src.encoder_client import embed_query_texts


class MedCPTReranker:
    def __init__(self, model_name: str, device: str = None):
        default_device = os.getenv("RERANK_DEVICE")
        self.device = device or default_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=True,
        ).to(self.device).eval()

    @torch.no_grad()
    def score(self, query: str, docs: list[str], max_length: int = 512) -> list[float]:
        if not docs:
            return []

        batch = self.tokenizer(
            [query] * len(docs),
            docs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**batch).logits
        if logits.ndim == 2 and logits.shape[1] == 1:
            scores = logits[:, 0]
        elif logits.ndim == 2:
            scores = logits[:, -1]
        else:
            scores = logits.view(-1)

        return scores.detach().float().cpu().tolist()


@lru_cache(maxsize=1)
def get_query_encoder():
    print("[INIT] Loading MedCPT query encoder once...")
    return MedCPTEncoder(MEDCPT_QUERY_MODEL, device=os.getenv("QUERY_EMBED_DEVICE"))


@lru_cache(maxsize=1)
def get_reranker():
    print("[INIT] Loading MedCPT reranker once...")
    return MedCPTReranker(MEDCPT_RERANK_MODEL, device=os.getenv("RERANK_DEVICE"))


@lru_cache(maxsize=1)
def get_qdrant_client():
    print("[INIT] Creating Qdrant client once...")
    return QdrantClient(url=QDRANT_URL)


@lru_cache(maxsize=1)
def get_llm_client():
    print("[INIT] Creating OpenAI/vLLM client once...")
    return OpenAI(base_url=VLLM_BASE_URL, api_key="dummy")


def resolve_patient_identifier(patient_identifier: str) -> tuple[str, str]:
    """
    Returns:
      (field_name, field_value)

    Supports either:
    - actual patient ID from folder name, e.g. 021494762
    - full folder name, e.g. Test 1 - 021494762
    - legacy hashed patient_id, e.g. patient_708d3d65ee7b4cc9
    """
    patient_identifier = patient_identifier.strip()

    if patient_identifier.startswith("patient_"):
        return "patient_id", patient_identifier

    if " - " in patient_identifier:
        return "patient_folder_name", patient_identifier

    return "actual_patient_id", patient_identifier


def answer_question(patient_id: str | None, question: str):
    
    reranker = get_reranker()
    client = get_qdrant_client()
    llm = get_llm_client()

    qvec = embed_query_texts([question])[0]

    query_filter = None
    if patient_id:
        filter_key, filter_value = resolve_patient_identifier(patient_id)
        query_filter = Filter(
            must=[
                FieldCondition(
                    key=filter_key,
                    match=MatchValue(value=filter_value)
                )
            ]
        )

    initial_hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,
        query_filter=query_filter,
        limit=50 if patient_id else 100,
    ).points

    if not initial_hits:
        scope = f"patient identifier: {patient_id}" if patient_id else "the full corpus"
        return {
            "answer": f"No matching chunks found for {scope}.",
            "sources": [],
        }

    docs = [h.payload.get("chunk_text", "") for h in initial_hits]
    rerank_scores = reranker.score(question, docs)

    reranked = sorted(
        zip(initial_hits, rerank_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    top_hits = reranked[:8]

    evidence_blocks = []
    sources = []

    for i, (hit, rr_score) in enumerate(top_hits, start=1):
        p = hit.payload
        evidence_blocks.append(
            f"[{i}] rerank_score={rr_score:.4f} "
            f"path={p.get('relative_path')} "
            f"page={p.get('page_num')}\n"
            f"{p.get('chunk_text')}"
        )
        sources.append(
            {
                "relative_path": p.get("relative_path"),
                "page_num": p.get("page_num"),
                "chunk_id": p.get("chunk_id"),
            }
        )

    prompt = f"""
You are answering questions about one patient only.

Rules:
- Use only the evidence below.
- If the answer is uncertain, say so.
- Cite evidence using the bracket numbers.
- Do not invent facts.

Question:
{question}

Evidence:
{chr(10).join(evidence_blocks)}
"""

    resp = llm.chat.completions.create(
        model=VLLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    return {
        "answer": resp.choices[0].message.content,
        "sources": sources,
    }
