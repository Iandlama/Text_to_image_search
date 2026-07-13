# DLS_project

Text-to-image search over memes. Each meme is an image (`images/{id}.webp`)
paired with textual signals for retrieval in `metadata.jsonl`.

## Dataset Structure (`metadata.jsonl`)

One JSON object per line, one line per meme:

* **`id`** — MD5 hash of the image, also used as its filename (`images/{id}.webp`).
* **`title`** — title of the post the meme comes from.
* **`image_desc`** — what is depicted (visual signal, matches visual queries).
* **`meaning`** — the meaning/joke of the meme (semantic signal, matches intent queries).

Example:

```json
{"id": "c343ad68...", "title": "And that's a fact", "image_desc": "Two dogs carry a white flag ...", "meaning": "Meme poster is saying that searching Google ..."}
```

## Scripts

* `loader/meme_loader.py` — downloads the dataset from Hugging Face and packs images + `metadata.jsonl` into a zip.
* `loader/meme_utils.py` — `parse_caption()`: parses the raw memecap prompt into `title` / `image_desc` / `meaning` fields.
* `loader/migrate_metadata.py` — rebuilds `metadata.jsonl` inside an already-downloaded archive to the new schema (no re-download).




## Vector Database Setup & Ingestion (Qdrant)

The vector storage, named-vector schema configuration, and high-throughput ingestion pipeline are handled via **Qdrant** (Task 3). The database runs locally inside a Docker container and communicates using the high-performance gRPC protocol (`port 6334`) to safely scale up to the mandatory 500,000+ objects constraint [].

### Schema Configuration (Iteration 1: Baseline)
The collection `meme_collection_v1` is configured to support **Multi-Vector Search**, indexing two separate dense vector fields per node into an interconnected coordinate space:

*   **`vector_image`**: Size `512`, `Distance.COSINE` (Visual embeddings from `clip-jina-v2`).
*   **`vector_text`**: Size `512`, `Distance.COSINE` (OCR text embeddings from `clip-jina-v2`).

### Payload & ID Translation
Qdrant does not natively accept raw string hashes as structural point IDs []. To bridge Task 1's MD5 hashes with Qdrant requirements, the ingestion pipeline dynamically converts string IDs into deterministic **UUID v5** formats []. 
The corresponding payload includes:
*   `ocr_text`: Text extracted from the image.
*   `file_name`: Path to the local asset (e.g., `<hash>.webp`).
*   `original_hash_id`: The raw MD5 string hash preserved for easy asset cross-referencing.

### Database Scripts

*   `init_db.py` — Establishes connection to the local instance via gRPC and handles the declarative creation of the collection schema [].
*   `upload_data.py` — Ingests `.parquet` files exported by Task 2. Features an internal look-ahead mechanism that pushes data in batches of 1,000 points with `wait=False`, preventing RAM bottlenecks [].
*   `search_test.py` — A query execution blueprint demonstrating how to conduct parallel multi-vector search queries (used by Task 4 FastAPI backend) [].

### Execution Guide (Windows PowerShell)

1. **Spin up Qdrant** (Ensure Docker Desktop is active):
   ```powershell
   docker run -p 6333:6333 -p 6334:6334 -v C:\qdrant_storage:/qdrant/storage:z qdrant/qdrant
   ```
   *Dashboard can be monitored at `http://localhost:6333/dashboard`* []

2. **Initialize Environment & Schema**:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   .venv\Scripts\Activate.ps1
   pip install qdrant-client pandas pyarrow
   python init_db.py
   ```

3. **Populate Database**:
   Place the `.parquet` dataset in the root folder, check the filename configuration inside `upload_data.py`, and run:
   ```powershell
   python upload_data.py
   ```

