from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)

COLLECTION_NAME = "meme_collection_v1"


def setup_meme_collection():
    if client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "vector_image": VectorParams(
                size=512,
                distance=Distance.COSINE
            ),
            "vector_text": VectorParams(
                size=512,
                distance=Distance.COSINE
            )
        }
    )
    print(f"Collection '{COLLECTION_NAME}' has successfully created!")


if __name__ == "__main__":
    setup_meme_collection()
