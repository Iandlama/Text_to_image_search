import time
import uuid
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


client = QdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)
COLLECTION_NAME = "meme_collection_v2"

NAMESPACE_MEMES = uuid.UUID('12345678-1234-5678-1234-567812345678')


def upload_parquet_dataset(parquet_path, batch_size=1000):
    print(f"Reading dataset from {parquet_path}...")
    start_time = time.time()

    df = pd.read_parquet(parquet_path, engine="pyarrow")
    total_rows = len(df)
    print(f"Total records found: {total_rows}")

    points = []
    batch_counter = 0

    for idx, row in df.iterrows():
        img_emb = list(row["image_embedding"])
        text_emb = list(row["text_embedding"])

        raw_id_str = str(row["id"])
        generated_uuid = str(uuid.uuid5(NAMESPACE_MEMES, raw_id_str))

        point = PointStruct(
            id=generated_uuid,
            vector={
                "vector_image": img_emb,
                "vector_text": text_emb
            },
            payload={
                "ocr_text": str(row["ocr_text"]),
                "file_name": str(row["file_name"]),
                "original_hash_id": raw_id_str
            }
        )
        points.append(point)

        if len(points) == batch_size:
            client.upsert(collection_name=COLLECTION_NAME,
                          points=points, wait=False)
            batch_counter += 1

            if batch_counter % 10 == 0:
                elapsed = time.time() - start_time
                processed = batch_counter * batch_size
                print(
                    f"Uploaded: {processed}/{total_rows} rows. Time elapsed: {elapsed:.1f}s")

            points = []

    if points:
        client.upsert(collection_name=COLLECTION_NAME,
                      points=points, wait=False)
        print(f"Uploaded final batch of {len(points)} rows.")

    total_time = time.time() - start_time
    print(
        f"Successfully ingested all {total_rows} elements into Qdrant in {total_time:.1f}s!")


if __name__ == "__main__":
    PARQUET_FILE = "embeddings_v2.parquet"
    upload_parquet_dataset(PARQUET_FILE)
