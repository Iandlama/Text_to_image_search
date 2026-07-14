"""
example.py - end-to-end usage of the `meme_embed` package.

Run:  python example.py
(Downloads weights from HuggingFace on first run; needs torch + transformers.)
"""
import numpy as np
from PIL import Image

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent 
sys.path.insert(0, str(src_path.absolute()))

from meme_embed import load_embedder, from_pretrained


def main():
    # 1) One-liner: load default model from the registry; override dim on the fly.
    emb = load_embedder("clip-jina-v2", embedding_dim=256)   # or just load_embedder()

    # 2) A meme = image + OCR text. Images accept PIL / np.ndarray / str(path|url|b64).
    image = Image.new("RGB", (224, 224), "white")
    ocr_text = "one does not simply walk into Mordor"

    # 3) Core interface: separate, directly-comparable vectors (not fused).
    image_vec, text_vec = emb.encode(image, ocr_text)
    print("-- image_vec:", image_vec.shape, "| text_vec:", text_vec.shape)   # (256,) (256,)

    # 4) Individual towers, all input formats.
    _ = emb.encode_image("meme.jpg")             # str input
    _ = emb.encode_text(ocr_text)                               # OCR text
    _ = emb.encode_image(np.zeros((224, 224, 3), np.uint8))     # np.ndarray input

    # 5) Search-time query (text, image, or both) - same space as memes.
    q = emb.encode_query(text="the lord of the rings meme about walking")
    print("-- cosine(query, ocr):", float(q @ text_vec))           # L2-normalized

    # 6) Batched meme encoding -> (image_matrix [N,D], text_matrix [N,D]).
    images = [Image.new("RGB", (224, 224), c) for c in ("red", "green", "blue")]
    ocrs = ["hello there", "general kenobi", "you are a bold one"]
    img_mat, txt_mat = emb.batch_encode(images, ocrs)
    print("-- batch:", img_mat.shape, txt_mat.shape)               # (3,256) (3,256)

    # # 7) Custom model without a registry entry:
    # emb2 = from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    #                        type="clip", embedding_dim=512, native_dim=1024)

    # # 8) R&D single fused space (Gemma-style VLM): one vector per meme.
    # mm = load_embedder("gemma-mm")
    # meme_vec = mm.encode_fused(image, ocr_text)


if __name__ == "__main__":
    main()