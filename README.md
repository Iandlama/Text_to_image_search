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

## Meme Embedding & Benchmark

### `meme_embed/` - Embedding library

A small, importable package that turns a meme (**image** + **OCR text**) into vectors for semantic search.  
It provides a unified interface for multiple multimodal models (CLIP, SigLIP, jina-clip, VLM) - all image↔text, never text‑only.

**Key features:**
- One‑liner: `from meme_embed import load_embedder; emb = load_embedder("clip-jina-v2", embedding_dim=256)`
- Core method: `image_vec, text_vec = emb.encode(image, ocr_text)` - separate vectors in the same space.
- Batched encoding: `img_mat, txt_mat = emb.batch_encode(images, ocrs)`
- Search‑time query: `query_vec = emb.encode_query(text="...")` (or image, or both late‑fused).
- Supports PIL, `np.ndarray`, file paths, URLs, base64 as image inputs.
- Lazy loading, automatic device/dtype, Matryoshka truncation, L2 normalisation.

> See **`examples/meme_embed/example.py`** for a complete quickstart.

---

### `benchmark/` - Evaluation suite

Runs retrieval benchmarks on a meme dataset (the one described above) to compare different embedding models.

**Dataset ground truth:** each meme’s query field (e.g., `meaning`) must retrieve the meme itself - identity retrieval.

**Tests (selectable):**
- `text2image` - text query → image embeddings only (the classic image‑only test).
- `text2ocr` - text query → OCR‑text embeddings.
- `text2fused` - text query → fused (image+OCR) embeddings.
- `space_consistency` - diagnostic: image↔OCR ranking overlap and alignment (guards against mixing spaces).
- `jpeg_robustness` - simulates “bad‑quality” memes (repeated JPEG recompression + downscale), reports content retention, identity recall, downstream recall, and artifact clustering.

**Usage:**
```bash
cd benchmark
python benchmark.py --config benchmark_models.yaml
python benchmark.py --only text2image,jpeg_robustness --models jina-clip-v2
python benchmark.py --export-config jina-clip-v2 --out prod_model.yaml
```

**Outputs:** tables (CSV, Markdown) and charts in `bench_results/`.

**OCR:** place a custom `my_ocr.py` (e.g., Tesseract wrapper) in the benchmark folder - results are cached in `bench_cache/ocr.json`.  
If missing, OCR channels degrade gracefully.

#### Benchmark results
> Based on the benchmark results, **jina‑clip‑v2** was selected as the primary embedder – it delivered the best overall balance of retrieval accuracy and robustness, and manual inspections confirmed it consistently produced the most semantically relevant matches.
=======

## Vector Database Setup & Ingestion (Qdrant)

The vector storage, named-vector schema configuration, and high-throughput ingestion pipeline are handled via **Qdrant**. The database runs locally inside a Docker container and communicates using the high-performance gRPC protocol (`port 6334`) to safely scale objects constraint.

### Schema Configurations

#### Iteration 1: Baseline
The collection `meme_collection_v1` is configured to support **Multi-Vector Search**, indexing two separate dense vector fields per node into an interconnected coordinate space:
*   **`vector_image`**: Size `512`, `Distance.COSINE` (Visual embeddings from `clip-jina-v2`).
*   **`vector_text`**: Size `512`, `Distance.COSINE` (OCR text embeddings from `clip-jina-v2`).

#### Iteration 2: Hardware-Optimized & Indexed
The collection `meme_collection_v2` introduces structural database-level optimizations to tackle resource trade-offs and latency overhead:
*   **`vector_image` & `vector_text`**: Size `256`, `Distance.COSINE` (Saves 50% storage via Matryoshka Truncation).
*   **On-Disk Storage (`on_disk=True`)**: Raw `int8` embeddings are offloaded directly to SSD storage to minimize the system's baseline RAM footprint.
*   **Binary Quantization (BQ)**: Compresses vector dimensions to 1-bit inside RAM (`always_ram=True`). Distances are evaluated using native hardware-level `Popcount` CPU bitwise operations, delivering a 32x index memory compression.
*   **Payload Field Indexing**: Registers an explicit structural `TEXT` index on `ocr_text` and a `KEYWORD` index on `file_name` to enable instant multi-field pre-filtering directly during HNSW graph traversal.

### Payload & ID Translation
Qdrant does not natively accept raw string hashes as structural point IDs. To bridge Task 1's MD5 hashes with Qdrant constraints, the pipeline dynamically converts string IDs into deterministic **UUID v5** formats. 
The stored payload includes:
*   `ocr_text`: Text extracted from the image.
*   `file_name`: Path to the local asset (e.g., `<hash>.webp`).
*   `original_hash_id`: The raw MD5 string hash preserved for asset cross-referencing.

### Database Scripts

*   `init_db.py` / `init_db_v2.py` — Establish connections via gRPC and handle the declarative creation of the vector collections and payload indexes.
*   `upload_data.py` / `upload_data_v2.py` — Ingest `.parquet` files exported by Task 2. Features batching chunks of 1,000 points with `wait=False` to prevent I/O bottlenecks.
*   `search_test.py` — A query execution blueprint demonstrating how to conduct parallel multi-vector search queries for Task 4 FastAPI backend.

### Execution Guide (Windows PowerShell)

1. **Spin up Qdrant** (Ensure Docker Desktop is active):
   ```powershell
   docker run -p 6333:6333 -p 6334:6334 -v C:\qdrant_storage:/qdrant/storage:z qdrant/qdrant
   ```
   *Dashboard can be monitored at `http://localhost:6333/dashboard`* 

2. **Initialize Environment & Schema**:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   .venv\Scripts\Activate.ps1
   pip install qdrant-client pandas pyarrow
   
   # For Iteration 1: python init_db.py
   # For Iteration 2:
   python init_db_v2.py
   ```

3. **Populate Database**:
   Place the `.parquet` dataset in the root folder, verify the target filename inside the upload script, and run:
   ```powershell
   # For Iteration 1: python upload_data.py
   # For Iteration 2:
   python upload_data_v2.py
   ```
