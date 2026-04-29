import os
from functools import lru_cache
from types import SimpleNamespace

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
from src.keyword_retrieval import get_keyword_retriever
from src.retrieval_planner import build_retrieval_plan
from src.structured_answering import build_structured_answer_prompt, extract_json_object


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
    - full patient folder name, e.g. Test 1 - 021494762
    - legacy hashed patient_id, e.g. patient_708d3d65ee7b4cc9
    """
    patient_identifier = patient_identifier.strip()

    if patient_identifier.startswith("patient_"):
        return "patient_id", patient_identifier

    if " - " in patient_identifier:
        return "patient_folder_name", patient_identifier

    return "actual_patient_id", patient_identifier


def build_filter(patient_id: str | None = None, document_type: str | None = None) -> Filter | None:
    must = []

    if patient_id:
        filter_key, filter_value = resolve_patient_identifier(patient_id)
        must.append(
            FieldCondition(
                key=filter_key,
                match=MatchValue(value=filter_value),
            )
        )

    if document_type:
        must.append(
            FieldCondition(
                key="document_type",
                match=MatchValue(value=document_type),
            )
        )

    if not must:
        return None

    return Filter(must=must)


def query_qdrant(client: QdrantClient, query_text: str, patient_id: str | None, document_type: str | None, limit: int):
    qvec = embed_query_texts([query_text])[0]
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,
        query_filter=build_filter(patient_id=patient_id, document_type=document_type),
        limit=limit,
    ).points


def collect_planned_hits(client: QdrantClient, patient_id: str | None, question: str, limit_per_query: int = 12):
    """Run CLI-RAG-style planned retrieval, with broad fallback for older indexes."""
    plan = build_retrieval_plan(question)
    hits_by_chunk_id = {}

    # Targeted global/local-ish search: subquery x document_type.
    for subquery in plan.subqueries:
        for document_type in plan.target_document_types:
            try:
                hits = query_qdrant(
                    client=client,
                    query_text=subquery,
                    patient_id=patient_id,
                    document_type=document_type,
                    limit=limit_per_query,
                )
            except Exception:
                hits = []

            for hit in hits:
                chunk_id = hit.payload.get("chunk_id")
                if chunk_id and chunk_id not in hits_by_chunk_id:
                    hits_by_chunk_id[chunk_id] = hit

    # Keyword/BM25 retrieval improves recall for exact clinical terms that dense
    # retrieval may miss, e.g. B12, ferritin, PTH, RYGB, thiamine.
    # Keep this conservative: use the same planner subqueries and target document
    # types, then let the MedCPT reranker decide what survives.
    try:
        keyword_retriever = get_keyword_retriever()
        keyword_queries = list(dict.fromkeys([plan.primary_query, *plan.subqueries]))

        for subquery in keyword_queries:
            keyword_hits = keyword_retriever.search(
                query=subquery,
                patient_id=patient_id,
                document_types=plan.target_document_types,
                limit=limit_per_query,
            )

            for keyword_hit in keyword_hits:
                payload = dict(keyword_hit.record)
                chunk_id = payload.get("chunk_id")
                if chunk_id and chunk_id not in hits_by_chunk_id:
                    hits_by_chunk_id[chunk_id] = SimpleNamespace(payload=payload)

    except Exception as exc:
        print(f"[WARN] Keyword retrieval skipped: {exc}")

    # Broad fallback is important for pre-existing indexes that do not yet have
    # document_type payloads, and for questions not covered by deterministic rules.
    for subquery in plan.subqueries[:3]:
        hits = query_qdrant(
            client=client,
            query_text=subquery,
            patient_id=patient_id,
            document_type=None,
            limit=50 if patient_id else 100,
        )
        for hit in hits:
            chunk_id = hit.payload.get("chunk_id")
            if chunk_id and chunk_id not in hits_by_chunk_id:
                hits_by_chunk_id[chunk_id] = hit

    return list(hits_by_chunk_id.values()), plan


def answer_question(patient_id: str | None, question: str, structured: bool = False):
    reranker = get_reranker()
    client = get_qdrant_client()
    llm = get_llm_client()

    initial_hits, plan = collect_planned_hits(client, patient_id, question)

    if not initial_hits:
        scope = f"patient identifier: {patient_id}" if patient_id else "the full corpus"
        return {
            "answer": f"No matching chunks found for {scope}.",
            "structured_answer": None,
            "sources": [],
            "retrieval_plan": plan.to_dict(),
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
            f"document_type={p.get('document_type', 'unknown')} "
            f"section={p.get('section_title')} "
            f"path={p.get('relative_path')} "
            f"page={p.get('page_num')}\n"
            f"{p.get('chunk_text')}"
        )
        sources.append(
            {
                "relative_path": p.get("relative_path"),
                "page_num": p.get("page_num"),
                "chunk_id": p.get("chunk_id"),
                "document_type": p.get("document_type"),
                "section_title": p.get("section_title"),
                "rerank_score": rr_score,
            }
        )

    plan_dict = plan.to_dict()

    if structured:
        prompt = build_structured_answer_prompt(
            question=question,
            evidence_blocks=evidence_blocks,
            retrieval_plan=plan_dict,
        )
        resp = llm.chat.completions.create(
            model=VLLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw_answer = resp.choices[0].message.content or ""
        structured_answer = extract_json_object(raw_answer)
        return {
            "answer": structured_answer.get("concise_answer", raw_answer) if structured_answer else raw_answer,
            "structured_answer": structured_answer,
            "sources": sources,
            "retrieval_plan": plan_dict,
        }

    prompt = f"""
You are answering questions about one patient or one local EHR corpus.

Rules:
- Use only the evidence below.
- If the answer is uncertain, say so.
- Cite evidence using the bracket numbers.
- Do not invent facts.
- If relevant evidence is absent, explicitly say it was not found in the retrieved evidence.

Retrieval plan:
{plan_dict}

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
        "structured_answer": None,
        "sources": sources,
        "retrieval_plan": plan_dict,
    }
