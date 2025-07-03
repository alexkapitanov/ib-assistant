import os
from typing import List, Dict, Any, Tuple
import qdrant_client
import boto3

qdrant = None
s3 = None

COLLECTION_NAME = "documents"


def init_clients() -> Tuple[qdrant_client.QdrantClient, boto3.client]:
    global qdrant, s3
    qdrant = qdrant_client.QdrantClient("qdrant", port=6333)
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
        aws_access_key_id=os.getenv("MINIO_ROOT_USER", "minio"),
        aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD", "minio123"),
    )
    return qdrant, s3


def upsert_vector(id: str, vector: List[float], payload: Dict[str, Any]):
    """
    Добавляет или обновляет вектор в коллекции Qdrant.
    """
    global qdrant
    if qdrant is None:
        raise RuntimeError("Qdrant client is not initialized")
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            {
                "id": id,
                "vector": vector,
                "payload": payload,
            }
        ],
    )


def search_vectors(query_vector: List[float], top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Поиск ближайших векторов по косинусной близости.
    """
    global qdrant
    if qdrant is None:
        raise RuntimeError("Qdrant client is not initialized")
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        score_threshold=None,
    )
    return [
        {"id": hit.id, "score": hit.score, "payload": hit.payload}
        for hit in hits
    ]

# Инициализация клиентов при импорте модуля
init_clients()
