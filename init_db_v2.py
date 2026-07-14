from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, BinaryQuantization, BinaryQuantizationConfig

client = QdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)
COLLECTION_NAME_V2 = "meme_collection_v2"


def setup_optimized_collection():
    if client.collection_exists(collection_name=COLLECTION_NAME_V2):
        print(
            f"Коллекция '{COLLECTION_NAME_V2}' уже существует. Удаляем для чистоты эксперимента...")
        client.delete_collection(collection_name=COLLECTION_NAME_V2)

    # 1. Создаем саму коллекцию с бинарным квантованием и on_disk логикой
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
    print(
        f"Коллекция '{COLLECTION_NAME_V2}' создана. Начинаем индексацию Payload...")

    # 2. ВНЕДРЕНИЕ PAYLOAD-ИНДЕКСОВ ЧЕРЕЗ СТРОКОВЫЕ ТИПЫ (без импортов):
    # Полнотекстовый индекс для OCR-текста
    client.create_payload_index(
        collection_name=COLLECTION_NAME_V2,
        field_name="ocr_text",
        field_schema="text"  # Используем стандартную строку 'text'
    )

    # Ключевой индекс для имен файлов
    client.create_payload_index(
        collection_name=COLLECTION_NAME_V2,
        field_name="file_name",
        field_schema="keyword"  # Используем стандартную строку 'keyword'
    )

    print(
        f"Супер-оптимизированная коллекция '{COLLECTION_NAME_V2}' (256xI8 + BQ + Payload Indexes) успешно создана!")


if __name__ == "__main__":
    setup_optimized_collection()
