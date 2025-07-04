from qdrant_client import QdrantClient, models

_QDRANT = None
COLLECTION = "docs"
DIM = 1536          # ← или 3072, если вы используете text-embedding-3-large

def get_client() -> QdrantClient:
    global _QDRANT
    if _QDRANT is None:
        _QDRANT = QdrantClient(host="qdrant", port=6333)
        ensure_collection(_QDRANT)
    return _QDRANT


def ensure_collection(client: QdrantClient):
    "Создаёт коллекцию с нужным размером, если она отсутствует или имеет другой dim"
    if client.collection_exists(COLLECTION):
        info = client.get_collection(COLLECTION)
        if info.config.params.vectors.size == DIM:
            return
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=DIM,
            distance=models.Distance.COSINE,
        ),
    )
