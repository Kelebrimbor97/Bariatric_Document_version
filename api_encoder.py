from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

from src.medcpt_embed import MedCPTEncoder
from src.config import MEDCPT_QUERY_MODEL, MEDCPT_ARTICLE_MODEL

app = FastAPI(title="MedCPT Encoder API", version="0.1.0")


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    model_type: Literal["query", "article"]


print("[INIT] Loading MedCPT query encoder on cuda:1 ...")
QUERY_ENCODER = MedCPTEncoder(MEDCPT_QUERY_MODEL, device="cuda:1")

print("[INIT] Loading MedCPT article encoder on cuda:1 ...")
ARTICLE_ENCODER = MedCPTEncoder(MEDCPT_ARTICLE_MODEL, device="cuda:1")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embed/query", response_model=EmbedResponse)
def embed_query(req: EmbedRequest):
    vecs = QUERY_ENCODER.encode(req.texts)
    return EmbedResponse(
        vectors=vecs.tolist(),
        model_type="query",
    )


@app.post("/embed/article", response_model=EmbedResponse)
def embed_article(req: EmbedRequest):
    vecs = ARTICLE_ENCODER.encode(req.texts)
    return EmbedResponse(
        vectors=vecs.tolist(),
        model_type="article",
    )