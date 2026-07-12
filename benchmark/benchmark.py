"""
benchmark.py
============
Compare multimodal embedding models on a meme dataset. Runs a selectable set of
retrieval tests (incl. **image-only** retrieval and a **bad-quality / JPEG-artifact**
robustness suite), collects metrics, and writes comparison tables + charts. All
comparisons are embedding-space-safe (see bench_core.check_same_space).

Dataset (very simple):  metadata.jsonl  +  images/{id}.webp
    metadata line: {"id","title","image_desc","meaning"}
No query list is needed - a query is generated from a metadata field (default
"meaning") and must retrieve the meme it came from (identity ground truth).

CLI examples
------------
    python benchmark.py --config benchmark_models.yaml
    python benchmark.py --list-tests
    python benchmark.py --only text2image,jpeg_robustness --models clip-jina-v2
    python benchmark.py --export-config clip-jina-v2 --out prod_model.yaml

See "How to run on a custom dataset" at the bottom of this file.
"""


import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import yaml

from meme_embed import EmbedderConfig, build_embedder, setup_logging
from bench_core import (
    MemeDataset, OcrCache, EmbeddingStore, Embeddings,
    build_channels, jpeg_degrade,
    retrieval_metrics, mean_pair_cosine, offdiag_mean_cosine, topk_sets, mean_jaccard,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x=None, **k):
        return x if x is not None else iter(())

logger = logging.getLogger("meme_embedder.benchmark")


# --------------------------------------------------------------------------- #
# Context passed to each test                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class BenchContext:
    model_name: str
    dim: int
    embedder: object
    channels: dict[str, Embeddings]
    dataset: MemeDataset
    store: EmbeddingStore
    gt: np.ndarray
    run: dict
    ds_fp: str


# --------------------------------------------------------------------------- #
# Tests  (registry - add a function and decorate to expose it on the CLI)      #
# --------------------------------------------------------------------------- #
TEST_REGISTRY: dict[str, Callable[[BenchContext], list[dict]]] = {}
TEST_DOC: dict[str, str] = {}


def test(name: str, doc: str):
    def deco(fn):
        TEST_REGISTRY[name] = fn
        TEST_DOC[name] = doc
        return fn
    return deco


def _row(ctx: BenchContext, test_name: str, channel: str, metrics: dict) -> dict:
    return {"model": ctx.model_name, "dim": ctx.dim, "test": test_name, "channel": channel, **metrics}


def _retrieval_on(ctx: BenchContext, channel: str, test_name: str) -> list[dict]:
    if channel not in ctx.channels:
        logger.warning("[%s] channel '%s' not computed; skipping %s.", ctx.model_name, channel, test_name)
        return []
    m = retrieval_metrics(
        ctx.channels["query"], ctx.channels[channel], ctx.gt,
        ks=ctx.run["ks"], ndcg_k=ctx.run["ndcg_k"],
    )
    return [_row(ctx, test_name, channel, m)]


@test("text2image", "Text query -> IMAGE embeddings only (no OCR). Visual-semantic retrieval.")
def _t2i(ctx):  # the required image-only retrieval test
    return _retrieval_on(ctx, "image", "text2image")


@test("text2ocr", "Text query -> OCR-text embeddings only. 'What was written' retrieval.")
def _t2o(ctx):
    return _retrieval_on(ctx, "ocr", "text2ocr")


@test("text2fused", "Text query -> fused (image+OCR) embeddings. Combined retrieval.")
def _t2f(ctx):
    return _retrieval_on(ctx, "fused", "text2fused")


@test("space_consistency",
      "Diagnostic: image vs OCR channel agreement + alignment; guards against space mixing.")
def _space(ctx):
    if "image" not in ctx.channels or "ocr" not in ctx.channels:
        logger.warning("[%s] need image+ocr channels for space_consistency.", ctx.model_name)
        return []
    k = max(ctx.run["ks"])
    img_sets = topk_sets(ctx.channels["query"], ctx.channels["image"], k)
    ocr_sets = topk_sets(ctx.channels["query"], ctx.channels["ocr"], k)
    overlap = mean_jaccard(img_sets, ocr_sets)          # how similar the two channels rank
    alignment = mean_pair_cosine(ctx.channels["image"], ctx.channels["ocr"])  # same-space cos
    n = len(ctx.dataset)
    r1 = retrieval_metrics(ctx.channels["query"], ctx.channels["image"], ctx.gt,
                           ks=[1], ndcg_k=1)["recall@1"]
    return [_row(ctx, "space_consistency", "diagnostic", {
        f"img_ocr_overlap@{k}": overlap,
        "img_ocr_alignment_cos": alignment,
        "text2image_recall@1": r1,
        "random_recall@1": 1.0 / n,
    })]


@test("jpeg_robustness",
      "Bad-quality memes: repeated JPEG recompression. Content retention, identity "
      "recall of degraded images, downstream text->degraded recall, artifact clustering.")
def _jpeg(ctx):
    if "image" not in ctx.channels:
        logger.warning("[%s] need image channel for jpeg_robustness.", ctx.model_name)
        return []
    aug = ctx.run["augmentation"]
    nr, ds_scale = aug["n_recompress"], aug["downscale"]
    clean = ctx.channels["image"]
    clean_offdiag = offdiag_mean_cosine(clean)
    rows = []
    for q in aug["jpeg_qualities"]:
        deg_imgs = [jpeg_degrade(ctx.dataset.load_image(it), quality=q, n_recompress=nr, downscale=ds_scale)
                    for it in ctx.dataset.items]
        tag = f"{ctx.model_name}|{ctx.dim}|deg_q{q}_n{nr}_d{ds_scale}|image|{ctx.ds_fp}"
        deg_v = ctx.store.get_or_compute(tag, lambda: ctx.embedder.encode_image(deg_imgs, normalize=True))
        deg = Embeddings(np.asarray(deg_v), ctx.model_name, ctx.dim, f"deg_q{q}")

        retention = mean_pair_cosine(clean, deg)                    # orig[i]·deg[i]
        identity = retrieval_metrics(deg, clean, ctx.gt, ks=ctx.run["ks"], ndcg_k=ctx.run["ndcg_k"])
        downstream = retrieval_metrics(ctx.channels["query"], deg, ctx.gt,
                                       ks=ctx.run["ks"], ndcg_k=ctx.run["ndcg_k"])
        artifact_ratio = (offdiag_mean_cosine(deg) / clean_offdiag) if clean_offdiag else float("nan")
        rows.append(_row(ctx, "jpeg_robustness", f"q{q}", {
            "quality": q,
            "content_cos": retention,
            "identity_recall@1": identity["recall@1"],
            "identity_recall@5": identity.get("recall@5", float("nan")),
            "downstream_recall@1": downstream["recall@1"],
            "downstream_mrr": downstream["mrr"],
            "artifact_cluster_ratio": artifact_ratio,
        }))
    return rows


# --------------------------------------------------------------------------- #
# Embedder construction                                                        #
# --------------------------------------------------------------------------- #
def make_embedder(name: str, entry: dict, target_dim: Optional[int]):
    d = dict(entry)
    d.pop("enabled", None)
    if "embedding_dim" not in d and target_dim:
        d["embedding_dim"] = target_dim
    cfg = EmbedderConfig.from_dict(name, d)
    return build_embedder(cfg), (cfg.resolved_dim() or 0)


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #
def run_benchmark(cfg: dict, *, models_subset=None, tests_subset=None, progress=True) -> "list[dict]":
    dcfg, run = cfg["dataset"], cfg["run"]
    cache_dir = run["cache_dir"]

    ds = MemeDataset(
        path=dcfg["path"], n_samples=dcfg["n_samples"], seed=dcfg.get("seed", 42),
        query_field=dcfg.get("query_field", "meaning"),
        image_dir=dcfg.get("image_dir", "images"), image_ext=dcfg.get("image_ext", "webp"),
    )
    ocr = OcrCache(cache_dir)
    ocr.attach(ds, progress=progress)
    store = EmbeddingStore(cache_dir)
    ds_fp, ocr_fp = ds.fingerprint(), ocr.content_hash(ds)
    gt = np.arange(len(ds))

    tests_to_run = tests_subset or run.get("tests", list(TEST_REGISTRY))
    unknown = [t for t in tests_to_run if t not in TEST_REGISTRY]
    if unknown:
        raise KeyError(f"Unknown tests {unknown}. Available: {sorted(TEST_REGISTRY)}")

    model_items = [(n, e) for n, e in cfg["models"].items()
                   if e.get("enabled", True) and (not models_subset or n in models_subset)]
    if not model_items:
        raise RuntimeError("No models selected/enabled.")

    records: list[dict] = []
    for name, entry in tqdm(model_items, desc="models", disable=not progress):
        logger.info("=== Model: %s ===", name)
        try:
            embedder, dim = make_embedder(name, entry, cfg.get("target_dim"))
            channels = build_channels(
                embedder, ds, store, model_name=name, dim=dim, ds_fp=ds_fp, ocr_fp=ocr_fp,
                channels=run.get("channels", ["image", "ocr", "fused"]),
                fused_alpha=run.get("fused_alpha", 0.5),
                query_field=dcfg.get("query_field", "meaning"), progress=progress,
            )
            ctx = BenchContext(name, dim, embedder, channels, ds, store, gt, run, ds_fp)
            for t in tests_to_run:
                records.extend(TEST_REGISTRY[t](ctx))
        except Exception as exc:  # noqa - one bad model shouldn't kill the whole run
            logger.exception("Model '%s' failed: %s", name, exc)
    return records


# --------------------------------------------------------------------------- #
# Reporting: tables + charts                                                   #
# --------------------------------------------------------------------------- #
def write_reports(records: list[dict], out_dir: str) -> None:
    import pandas as pd
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("No records to report.")
        return
    df.to_csv(out / "results_raw.csv", index=False)

    # --- retrieval comparison table (models x test/metric) ---
    retr = df[df["test"].isin(["text2image", "text2ocr", "text2fused"])]
    if not retr.empty:
        metric_cols = [c for c in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10") if c in retr]
        pivot = retr.pivot_table(index="model", columns="test", values=metric_cols)
        pivot.columns = [f"{t}·{m}" for m, t in pivot.columns]
        pivot = pivot.round(4)
        pivot.to_csv(out / "results_retrieval.csv")
        (out / "results_retrieval.md").write_text(pivot.to_markdown(), encoding="utf-8")
        _chart_retrieval(retr, out)

    jp = df[df["test"] == "jpeg_robustness"]
    if not jp.empty:
        jp.round(4).to_csv(out / "results_jpeg.csv", index=False)
        _chart_jpeg(jp, out)

    sp = df[df["test"] == "space_consistency"]
    if not sp.empty:
        sp.round(4).to_csv(out / "results_space.csv", index=False)

    _write_summary_md(df, out)
    logger.info("Reports written to %s", out.resolve())


def _chart_retrieval(retr, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metric = "recall@10" if "recall@10" in retr else "mrr"
    pv = retr.pivot_table(index="model", columns="test", values=metric)
    ax = pv.plot(kind="bar", figsize=(max(6, 1.4 * len(pv)), 4.5))
    ax.set_ylabel(metric)
    ax.set_title(f"Text→Meme retrieval - {metric} by channel")
    ax.legend(title="channel", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out / "chart_retrieval.png", dpi=140)
    plt.close()


def _chart_jpeg(jp, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for model, g in jp.groupby("model"):
        g = g.sort_values("quality")
        ax1.plot(g["quality"], g["identity_recall@1"], marker="o", label=model)
        ax2.plot(g["quality"], g["content_cos"], marker="o", label=model)
    for ax, title, yl in ((ax1, "Degraded-image identity Recall@1", "recall@1"),
                          (ax2, "Content retention cos(orig, degraded)", "cosine")):
        ax.set_xlabel("JPEG quality (lower = worse)")
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "chart_jpeg_robustness.png", dpi=140)
    plt.close()


def _write_summary_md(df, out: Path):
    lines = ["# Benchmark summary", ""]
    lines.append(f"- Models: {', '.join(sorted(df['model'].unique()))}")
    lines.append(f"- Tests: {', '.join(sorted(df['test'].unique()))}")
    lines.append("")
    if (out / "results_retrieval.md").exists():
        lines += ["## Retrieval (Text→Meme)", "", (out / "results_retrieval.md").read_text(encoding="utf-8"),
                  "", "![retrieval](chart_retrieval.png)", ""]
    if (out / "chart_jpeg_robustness.png").exists():
        lines += ["## Bad-quality / JPEG robustness", "", "![jpeg](chart_jpeg_robustness.png)", ""]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Export a single-model config (production-ready, meme_embed registry format)  #
# --------------------------------------------------------------------------- #
def export_model_config(cfg: dict, name: str, out_path: str) -> None:
    if name not in cfg["models"]:
        raise KeyError(f"Model '{name}' not in config. Have: {sorted(cfg['models'])}")
    entry = dict(cfg["models"][name])
    entry.pop("enabled", None)
    if "embedding_dim" not in entry and cfg.get("target_dim"):
        entry["embedding_dim"] = cfg["target_dim"]
    doc = {"default_model": name, "models": {name: entry}}
    Path(out_path).write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    logger.info("Exported production config for '%s' -> %s", name, out_path)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main(argv=None):
    p = argparse.ArgumentParser(description="Meme embedding-model benchmark.")
    p.add_argument("--config", default="benchmark_models.yaml")
    p.add_argument("--data", help="override dataset.path")
    p.add_argument("--n", type=int, help="override dataset.n_samples")
    p.add_argument("--models", help="comma-separated subset of models")
    p.add_argument("--only", help="comma-separated subset of tests")
    p.add_argument("--out-dir", help="override run.out_dir")
    p.add_argument("--list-tests", action="store_true", help="list tests and exit")
    p.add_argument("--export-config", metavar="MODEL", help="export a single-model config and exit")
    p.add_argument("--out", default="prod_model.yaml", help="path for --export-config")
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args(argv)

    setup_logging()

    if args.list_tests:
        print("Available tests:")
        for name in TEST_REGISTRY:
            print(f"  {name:18s} {TEST_DOC[name]}")
        return

    cfg = _load_cfg(args.config)
    if args.data:
        cfg["dataset"]["path"] = args.data
    if args.n:
        cfg["dataset"]["n_samples"] = args.n
    if args.out_dir:
        cfg["run"]["out_dir"] = args.out_dir

    if args.export_config:
        export_model_config(cfg, args.export_config, args.out)
        return

    models_subset = args.models.split(",") if args.models else None
    tests_subset = args.only.split(",") if args.only else None
    records = run_benchmark(cfg, models_subset=models_subset,
                            tests_subset=tests_subset, progress=not args.no_progress)
    write_reports(records, cfg["run"]["out_dir"])


if __name__ == "__main__":
    main()


# =========================================================================== #
# HOW TO RUN ON A CUSTOM DATASET
# ---------------------------------------------------------------------------
# 1. Lay the data out as:
#        my_data/
#          ├── metadata.jsonl        # one JSON per line
#          └── images/{id}.webp      # (or .jpg/.png -> set dataset.image_ext)
#    Each metadata line needs at least: {"id": ..., "<query_field>": "..."}.
#    Default query_field is "meaning"; set dataset.query_field to any field you
#    want queries generated from (e.g. "image_desc" or "title").
#
# 2. Provide OCR:  create my_ocr.py exposing  get_ocr(img: PIL.Image) -> str
#    (Tesseract/EasyOCR/PaddleOCR wrapper). Results are cached in cache/ocr.json.
#    If my_ocr is missing, OCR text is empty and OCR/fused channels degrade
#    gracefully (a warning is logged).
#
# 3. Point the config at it and run a subset first:
#        python benchmark.py --data ./my_data --n 100 --only text2image
#    Then the full suite:
#        python benchmark.py --data ./my_data
#
# 4. Pick a winner and export a production config for the embedder library:
#        python benchmark.py --export-config clip-jina-v2 --out prod_model.yaml
#    Use it via:  load_embedder(registry_path="prod_model.yaml")
# =========================================================================== #
