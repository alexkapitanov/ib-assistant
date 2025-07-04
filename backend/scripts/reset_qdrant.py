from qdrant_client import QdrantClient

client = QdrantClient(host="qdrant", port=6333)
if client.collection_exists("docs"):
    client.delete_collection("docs")
print("🗑  collection 'docs' removed")
