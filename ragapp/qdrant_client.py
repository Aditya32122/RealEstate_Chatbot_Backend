from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import os

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = "realestate"
EMBEDDING_DIM = 768  # Dimension for Gemini text-embedding-004

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

def initialize_qdrant_collection():
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"Qdrant collection '{COLLECTION_NAME}' created.")
    else:
        print(f"Qdrant collection '{COLLECTION_NAME}' already exists.")

def add_vector(id: str, vector: list[float], metadata: dict):
    point = PointStruct(id=id, vector=vector, payload=metadata)
    client.upsert(collection_name=COLLECTION_NAME, points=[point])

def search_vectors(query_vector: list[float], top_k: int = 5):
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points