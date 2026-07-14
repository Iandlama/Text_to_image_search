import os
import sys

# настройка относительных путей
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import time
import json
import torch
import pandas as pd
import ir_measures
from ir_measures import Recall, RR, nDCG
from qdrant_client import QdrantClient
from tqdm import tqdm

# --- Конфигурация ---
QDRANT_HOST = "localhost"
QDRANT_GRPC_PORT = 6334
COLLECTION_NAME = "meme_collection_v1"
TOP_K = 20  # Извлекаем топ-20 кандидатов для оценки (по визуальной части и по тексту OCR)
WEIGHTED_ALPHA = 0.3 # настройка alpha параметра для weighted sum

QRELS_PATH = os.path.join(ROOT_DIR, "data", "eval", "qrels.txt") # пути к валидационному набору
QUERIES_JSONL_PATH = os.path.join(ROOT_DIR, "data", "eval", "queries.jsonl")

qdrant_client = QdrantClient(host=QDRANT_HOST, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True)

from meme_embed import load_embedder
meme_embedder = load_embedder()
meme_embedder.warmup()


def rrf_score(rank, k=60):
    return 1.0 / (k + rank)


def main():
    # Загружаем тестовые запросы и информацию об их источниках
    queries = []
    with open(QUERIES_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    
    df_queries = pd.DataFrame(queries)

    if "qid" in df_queries.columns:
        df_queries = df_queries.rename(columns={"qid": "query_id"})
    elif "id" in df_queries.columns:
        df_queries = df_queries.rename(columns={"id": "query_id"})

    if "query" in df_queries.columns:
        df_queries = df_queries.rename(columns={"query": "query_text"})
    elif "text" in df_queries.columns:
        df_queries = df_queries.rename(columns={"text": "query_text"})

    if "query_id" not in df_queries.columns or "query_text" not in df_queries.columns:
        raise KeyError(f"Не удалось распознать структуру JSONL. Доступные колонки: {list(df_queries.columns)}")

    df_queries["query_id"] = df_queries["query_id"].astype(str)
    
    trec_runs = {
        "image_only": [],
        "text_only": [],
        "hybrid_rrf": [],
        "hybrid_weighted": []
    }
    
    latency_records = {k: [] for k in trec_runs.keys()}

    print(f"Начало бенчмаркинга для {len(df_queries)} запросов...")

    for _, row in tqdm(df_queries.iterrows(), total=len(df_queries)):
        qid = row["query_id"]
        q_text = row["query_text"].strip()

        # --- Инференс вектора (Общий для всех) ---
        kw = {"task": "retrieval.query", "device": meme_embedder.device, "normalize_embeddings": False}
        if meme_embedder.config.resolved_dim():
            kw["truncate_dim"] = meme_embedder.config.resolved_dim()

        with torch.inference_mode():
            query_features = meme_embedder.model.encode_text([q_text], **kw)
        query_vector = meme_embedder._postprocess(query_features, normalize=True)[0].tolist()

        # --- Только картинки (Image Only) ---
        t0 = time.perf_counter()
        res_img = qdrant_client.query_points(
            collection_name=COLLECTION_NAME, query=query_vector, using="vector_image", limit=TOP_K, with_payload=True
        ).points
        latency_records["image_only"].append(time.perf_counter() - t0)

        for rank, hit in enumerate(res_img, start=1):
            doc_id = hit.payload.get("original_hash_id", hit.id)
            trec_runs["image_only"].append(f"{qid} Q0 {doc_id} {rank} {hit.score:.4f} image_only")

        # --- Только текст/OCR (Text Only) ---
        t0 = time.perf_counter()
        res_txt = qdrant_client.query_points(
            collection_name=COLLECTION_NAME, query=query_vector, using="vector_text", limit=TOP_K, with_payload=True
        ).points
        latency_records["text_only"].append(time.perf_counter() - t0)

        for rank, hit in enumerate(res_txt, start=1):
            doc_id = hit.payload.get("original_hash_id", hit.id)
            trec_runs["text_only"].append(f"{qid} Q0 {doc_id} {rank} {hit.score:.4f} text_only")

        # --- RRF ---
        t0 = time.perf_counter()
        rrf_candidates = {}
        for rank, hit in enumerate(res_img, start=1):
            doc_id = hit.payload.get("original_hash_id", hit.id)
            rrf_candidates[doc_id] = rrf_candidates.get(doc_id, 0.0) + rrf_score(rank)
        for rank, hit in enumerate(res_txt, start=1):
            doc_id = hit.payload.get("original_hash_id", hit.id)
            rrf_candidates[doc_id] = rrf_candidates.get(doc_id, 0.0) + rrf_score(rank)
        
        sorted_rrf = sorted(rrf_candidates.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
        latency_records["hybrid_rrf"].append(time.perf_counter() - t0)

        for rank, (doc_id, score) in enumerate(sorted_rrf, start=1):
            trec_runs["hybrid_rrf"].append(f"{qid} Q0 {doc_id} {rank} {score:.4f} hybrid_rrf")

        # --- Weighted Sum ---
        t0 = time.perf_counter()
        wsum_candidates = {}
        alpha = WEIGHTED_ALPHA
        for hit in res_img:
            doc_id = hit.payload.get("original_hash_id", hit.id)
            wsum_candidates[doc_id] = wsum_candidates.get(doc_id, 0.0) + alpha * hit.score
        for hit in res_txt:
            doc_id = hit.payload.get("original_hash_id", hit.id)
            wsum_candidates[doc_id] = wsum_candidates.get(doc_id, 0.0) + (1 - alpha) * hit.score
            
        sorted_wsum = sorted(wsum_candidates.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
        latency_records["hybrid_weighted"].append(time.perf_counter() - t0)

        for rank, (doc_id, score) in enumerate(sorted_wsum, start=1):
            trec_runs["hybrid_weighted"].append(f"{qid} Q0 {doc_id} {rank} {score:.4f} hybrid_weighted")

    # Записываем TREC run-файлы на диск
    os.makedirs(os.path.join(CURRENT_DIR, "eval_runs"), exist_ok=True)
    run_paths = {}
    for strategy, lines in trec_runs.items():
        path = f"eval_runs/run_{strategy}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        run_paths[strategy] = path

    # Расчёт метрик через ir_measures
    measures = [Recall@1, Recall@5, Recall@10, RR, nDCG@10]
    qrels = list(ir_measures.read_trec_qrels(QRELS_PATH))

    print("\n" + "="*70)
    print("ФИНАЛЬНЫЙ СРАВНИТЕЛЬНЫЙ АНАЛИЗ СТРАТЕГИЙ ПОИСКА")
    print("="*70)

    for strategy, run_path in run_paths.items():
        run = list(ir_measures.read_trec_run(run_path))
        
        agg_metrics = ir_measures.calc_aggregate(measures, qrels, run)
        avg_latency_ms = (sum(latency_records[strategy]) / len(latency_records[strategy])) * 1000

        print(f"\nСтратегия: {strategy.upper()}")
        print(f"  Среднее время запроса (Qdrant Latency): {avg_latency_ms:.2f} ms")
        print("  Общие метрики:")
        for m, val in agg_metrics.items():
            print(f"    {str(m):<10}: {val:.4f}")

        # Вычисляем метрики отдельно для каждого запроса
        per_query_results = []
        for metric_res in ir_measures.iter_calc(measures, qrels, run):
            per_query_results.append({
                "query_id": str(metric_res.query_id),
                "metric": str(metric_res.measure),
                "value": metric_res.value
            })
        
        if per_query_results:
            df_per_query = pd.DataFrame(per_query_results)
            df_pivot = df_per_query.pivot(index="query_id", columns="metric", values="value").reset_index()
            df_merged = df_pivot.merge(df_queries[["query_id", "source"]], on="query_id", how="inner")
            
            df_sources = df_merged.groupby("source").mean(numeric_only=True)
            print("Метрики в разрезе источников (Sources):")
            print(df_sources.to_string())
        print("-" * 70)


if __name__ == "__main__":
    main()