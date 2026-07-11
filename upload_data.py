import json
import time
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

client = QdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)
COLLECTION_NAME = "meme_collection_v1"


def upload_jsonl_in_batches(jsonl_path, batch_size=1000):
    points = []
    batch_counter = 0
    start_time = time.time()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            point = PointStruct(
                id=data["id"],
                vector={
                    "vector_image": data["image_embedding"],
                    "vector_text": data["text_embedding"]
                },
                payload={
                    "ocr_text": data["ocr_text"],
                    "file_name": data["file_name"]
                }
            )
            points.append(point)

            if len(points) == batch_size:
                client.upsert(collection_name=COLLECTION_NAME,
                              points=points, wait=False)
                batch_counter += 1
                if batch_counter % 10 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"Loaded: {batch_counter} ({batch_counter * batch_size}). Seconds: {elapsed:.1f}")
                points = []

        if points:
            client.upsert(collection_name=COLLECTION_NAME,
                          points=points, wait=False)

    print(
        f"Loading is complete! Number of object in database: {batch_counter * batch_size + len(points)}")


if __name__ == "__main__":
    pass
