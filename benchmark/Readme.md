# Meme retrieval benchmark

Two files + a config: `bench_core.py` (dataset, OCR, augmentation, embeddings+cache,
metrics, space-safety) and `benchmark.py` (tests, runner, reports, CLI). Reuses the
`meme_embed` package from part 1.

## Dataset (very simple)
```
dataset/
  ├── metadata.jsonl        # {"id","title","image_desc","meaning"} per line
  └── images/{id}.webp
```
No query list needed: a query is generated from a metadata field (default `meaning`)
and must retrieve the meme it came from (identity ground truth). N items are sampled
randomly (seeded) - `dataset.n_samples` in the config.

OCR is pulled from each image via `from my_ocr import get_ocr` (`get_ocr(img)->str`)
and cached in `bench_cache/ocr.json`.

## Run
```bash
python benchmark.py --list-tests                 # list tests
python benchmark.py --config benchmark_models.yaml
python benchmark.py --only text2image,jpeg_robustness --models jina-clip-v2
python benchmark.py --export-config jina-clip-v2 --out prod_model.yaml
```

## Tests (selectable)
- `text2image` - text query → **image embeddings only** (no OCR). *The image-only test.*
- `text2ocr` - text query → OCR-text embeddings.
- `text2fused` - text query → fused (image+OCR) embeddings.
- `space_consistency` - diagnostic: image↔OCR ranking overlap + alignment; guards against mixing spaces.
- `jpeg_robustness` - **bad-quality memes**: repeated JPEG recompression (+downscale). Reports content
  retention `cos(orig,deg)`, degraded-image **identity** Recall@k, downstream text→degraded recall, and an
  artifact-clustering ratio. Produces a degradation curve.

## Outputs (`bench_results/`)
`results_raw.csv`, `results_retrieval.{csv,md}`, `results_jpeg.csv`, `results_space.csv`,
`chart_retrieval.png`, `chart_jpeg_robustness.png`, `SUMMARY.md`.

## Embedding-space safety
Every comparison goes through `check_same_space`: vectors are tagged `model|dim` and the
benchmark refuses to cosine-compare different models/dims. Query and corpus are always
embedded by the same model (same shared space), so text→image is valid; models are only
compared by *metrics*, never by raw vectors.

## Custom dataset / models
See the "HOW TO RUN ON A CUSTOM DATASET" block at the bottom of `benchmark.py`, and add
models under `models:` in `benchmark_models.yaml` (`enabled: true` to include heavier ones).
