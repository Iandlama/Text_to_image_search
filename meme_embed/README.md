# Meme Embedder - Architecture & Pipeline

A small, importable **package** that turns a meme (**image + OCR text**) into
vectors for semantic search.

```
meme_embed/
  ├── __init__.py     # re-exports the public API  ->  from meme_embed import load_embedder
  ├── config.py       # config / registry / .env / logging / image I/O (torch-free)
  └── embedders.py    # BaseMemeEmbedder + ClipEmbedder + MultimodalEmbedder
models.yaml · example.py · .env.example · requirements.txt
```

## Class hierarchy

```
BaseMemeEmbedder (ABC)               # device/dtype, batching, Matryoshka truncation,
      │                              # L2-norm, image coercion, timing, public API
      ├── ClipEmbedder               # CLIP / SigLIP / jina-clip: image + text towers,
      │                              # one shared space (production baseline)
      └── MultimodalEmbedder         # Gemma-style VLM, mean-pooled hidden states:
                                     # single fused space + encode_fused() (R&D)
```

Only **multimodal** imag<->text models are used - never a text-only encoder. The
image is the primary signal; OCR is embedded by the *same* model's text tower so
image and text vectors live in one comparable space. Subclasses implement just two
primitives - `_embed_images` and `_embed_texts` - everything else is shared.

## Public API (identical for every embedder)

| Method | Input -> Output |
|---|---|
| `encode(image, ocr_text)` | -> `(image_vec, text_vec)` - **separate** vectors, not fused |
| `encode_image(img)` | PIL / `np.ndarray` / `str`(path·URL·base64) -> `[D]` or `[N,D]` |
| `encode_text(ocr)` | OCR string(s) -> `[D]` or `[N,D]` |
| `encode_query(text=, image=)` | search query (text, image, or late-fused mean) -> `[D]` |
| `batch_encode(images, ocrs)` | -> `(image_matrix[N,D], text_matrix[N,D])` |
| `MultimodalEmbedder.encode_fused(image, ocr)` | interleaved image+OCR -> one meme vec |

## Pipeline

```
raw input (PIL | np.ndarray | str)
   -> to_pil()                      # coerce -> RGB PIL
   -> processor / tokenizer         # per-model
   -> model tower  (inference_mode + autocast, batched)   # image OR text
   -> to_numpy -> truncate to embedding_dim (Matryoshka) -> L2-normalize
   -> np.float32 vector, cosine-ready
```

## Config & customization

* **Registry** (`models.yaml` or built-in): each entry sets `type`, `model_name`,
  `api`, `embedding_dim` (256/512), `native_dim`, `matryoshka`, dtype/padding, etc.
* **Env / `.env`** (auto-loaded): `MEME_EMBEDDER_MODEL / _CONFIG / _DEVICE / _DTYPE /
  _LOG_LEVEL / _CACHE_DIR`, plus `HF_TOKEN` for gated models.
* **Custom models**: `from_pretrained("org/model", type="clip", embedding_dim=512)`
  or add a YAML entry - no code changes.
* **Dims**: 256/512 come "for free" on Matryoshka models (jina-clip-v2) via
  slice + renormalize; non-Matryoshka models log a warning when truncated.

## Optimizations

`torch.inference_mode`, CUDA `autocast` (bf16->fp16->fp32 auto), configurable
batching, **lazy** weight loading (+`warmup()`), `.eval()`, optional `torch.compile`,
and normalization done once per batch in NumPy. Runs on CUDA or CPU (auto-detected).

## Roadmap fit
* **Level 1 (prod):** `ClipEmbedder` -> separate image/text indices, rank-fusion at query time.
* **Level 2 (R&D):** `MultimodalEmbedder.encode_fused` -> single index; A/B vs baseline.
