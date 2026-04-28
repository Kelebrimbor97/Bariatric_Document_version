from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def get_client(url: str):
    return QdrantClient(url=url)

def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int):
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

def upsert_points(client: QdrantClient, collection_name: str, ids, vectors, payloads):
    points = [
        PointStruct(id=i, vector=v.tolist(), payload=p)
        for i, v, p in zip(ids, vectors, payloads)
    ]
    client.upsert(collection_name=collection_name, points=points)