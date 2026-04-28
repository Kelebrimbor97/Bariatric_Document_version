from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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


def main():
    patient_id = input("Patient ID: ").strip()
    question = input("Question: ").strip()

    query_encoder = MedCPTEncoder(
        MEDCPT_QUERY_MODEL,
        device=os.getenv("QUERY_EMBED_DEVICE", os.getenv("EMBED_DEVICE")),
    )
    reranker = MedCPTReranker(MEDCPT_RERANK_MODEL, device=os.getenv("RERANK_DEVICE"))

    qvec = query_encoder.encode([question])[0]
    client = QdrantClient(url=QDRANT_URL)

    initial_hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec.tolist(),
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="patient_id",
                    match=MatchValue(value=patient_id)
                )
            ]
        ),
        limit=24,
    ).points

    if not initial_hits:
        print("\nNo matching chunks found for that patient.")
        return

    docs = [h.payload.get("chunk_text", "") for h in initial_hits]
    rerank_scores = reranker.score(question, docs)

    reranked = sorted(
        zip(initial_hits, rerank_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    top_hits = reranked[:8]

    evidence_blocks = []
    print("\n=== RETRIEVED EVIDENCE (RERANKED) ===\n")
    for i, (hit, rr_score) in enumerate(top_hits, start=1):
        p = hit.payload
        block = (
            f"[{i}] rerank_score={rr_score:.4f} "
            f"path={p.get('relative_path')} "
            f"page={p.get('page_num')}\n"
            f"{p.get('chunk_text')}"
        )
        evidence_blocks.append(block)
        print(block[:1600])
        print("\n" + "=" * 100 + "\n")

    prompt = f"""
You are answering questions about one patient only.

Rules:
- Use only the evidence below.
- If the answer is uncertain, say so.
- Cite evidence using the bracket numbers.
- Do not invent facts.
- Prefer direct evidence from the most relevant chunks.
- If multiple chunks describe the same document, summarize them together.

Question:
{question}

Evidence:
{chr(10).join(evidence_blocks)}
"""

    llm = OpenAI(base_url=VLLM_BASE_URL, api_key="dummy")
    resp = llm.chat.completions.create(
        model=VLLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    print("\n=== ANSWER ===\n")
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
