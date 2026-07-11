from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)
COLLECTION_NAME = "meme_collection_v1"


def test_search(query_vector_image, query_vector_text, top_k=5):

    image_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=("vector_image", query_vector_image),
        limit=top_k
    )

    # 2. Поиск по текстовому смыслу/OCR (E5)
    text_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=("vector_text", query_vector_text),
        limit=top_k
    )

    return image_results, text_results
