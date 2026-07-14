from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, BinaryQuantization, BinaryQuantizationConfig

client = QdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)
COLLECTION_NAME_V2 = "meme_collection_v2"


def setup_optimized_collection():
    if client.collection_exists(collection_name=COLLECTION_NAME_V2):
        client.delete_collection(collection_name=COLLECTION_NAME_V2)

    client.create_collection(
        collection_name=COLLECTION_NAME_V2,
        vectors_config={
            "vector_image": VectorParams(size=256, distance=Distance.COSINE, on_disk=True),
            "vector_text": VectorParams(size=256, distance=Distance.COSINE, on_disk=True)
        },
        quantization_config=BinaryQuantization(
            binary=BinaryQuantizationConfig(always_ram=True)
        )
    )

    client.create_payload_index(
        collection_name=COLLECTION_NAME_V2,
        field_name="ocr_text",
        field_schema="text"
    )

    client.create_payload_index(
        collection_name=COLLECTION_NAME_V2,
        field_name="file_name",
        field_schema="keyword"
    )

    print(
        f" '{COLLECTION_NAME_V2}' (256xI8 + BQ + Payload Indexes) has been successfully created!")


if __name__ == "__main__":
    setup_optimized_collection()
