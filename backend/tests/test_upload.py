from qdrant_client import QdrantClient

def test_qdrant_points_after_upload():
    client = QdrantClient(host="qdrant", port=6333)
    count = client.count(collection_name="docs", exact=True).count
    assert count > 0, f"Qdrant collection 'docs' is empty (count={count}) after upload"
