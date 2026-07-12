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

We append a new section to the existing README.md (the one about dataset and scripts). Below is the **additional content** to place after the existing `## Scripts` section.

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