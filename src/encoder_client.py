import requests
from src.config import ENCODER_API_URL


def embed_query_texts(texts: list[str]) -> list[list[float]]:
    resp = requests.post(
        f"{ENCODER_API_URL}/embed/query",
        json={"texts": texts},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["vectors"]


def embed_article_texts(texts: list[str]) -> list[list[float]]:
    resp = requests.post(
        f"{ENCODER_API_URL}/embed/article",
        json={"texts": texts},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["vectors"]