import os
import sys

# настройка относительных путей
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


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
TOP_K = 20  # Число кандидатов из базы для последующего переранжирования

QRELS_PATH = os.path.join(ROOT_DIR, "data", "eval", "qrels.txt") # пути к валидационному набору
QUERIES_JSONL_PATH = os.path.join(ROOT_DIR, "data", "eval", "queries.jsonl")

qdrant_client = QdrantClient(host=QDRANT_HOST, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True)

from meme_embed import load_embedder
meme_embedder = load_embedder()
meme_embedder.warmup()


def main():
    # 1. Загружаем запросы
    queries = []
    with open(QUERIES_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    
    df_queries = pd.DataFrame(queries)
    
    # Нормализуем колонки
    if "qid" in df_queries.columns:
        df_queries = df_queries.rename(columns={"qid": "query_id"})
    elif "id" in df_queries.columns:
        df_queries = df_queries.rename(columns={"id": "query_id"})

    if "query" in df_queries.columns:
        df_queries = df_queries.rename(columns={"query": "query_text"})
    elif "text" in df_queries.columns:
        df_queries = df_queries.rename(columns={"text": "query_text"})

    df_queries["query_id"] = df_queries["query_id"].astype(str)

    # 2. Кэшируем первичные результаты поиска Qdrant в оперативную память
    print("Шаг 1: Сбор сырых результатов поиска Qdrant (выполняется один раз)...")
    cached_search_results = []

    for _, row in tqdm(df_queries.iterrows(), total=len(df_queries)):
        qid = row["query_id"]
        q_text = row["query_text"].strip()
        source = row.get("source", "unknown")

        # Получаем эмбеддинг
        kw = {"task": "retrieval.query", "device": meme_embedder.device, "normalize_embeddings": False}
        if meme_embedder.config.resolved_dim():
            kw["truncate_dim"] = meme_embedder.config.resolved_dim()

        with torch.inference_mode():
            query_features = meme_embedder.model.encode_text([q_text], **kw)
        query_vector = meme_embedder._postprocess(query_features, normalize=True)[0].tolist()

        # Ищем по картинкам
        res_img = qdrant_client.query_points(
            collection_name=COLLECTION_NAME, query=query_vector, using="vector_image", limit=TOP_K, with_payload=True
        ).points

        # Ищем по тексту
        res_txt = qdrant_client.query_points(
            collection_name=COLLECTION_NAME, query=query_vector, using="vector_text", limit=TOP_K, with_payload=True
        ).points

        cached_search_results.append({
            "qid": qid,
            "source": source,
            "img_hits": res_img,
            "txt_hits": res_txt
        })

    # 3. Запуск Grid Search по сетке alpha в памяти
    print("\nШаг 2: Запуск Grid Search по параметру alpha...")
    grid_alphas = [round(x * 0.1, 1) for x in range(11)]  # [0.0, 0.1, ..., 1.0]
    
    qrels = list(ir_measures.read_trec_qrels(QRELS_PATH))
    measures = [Recall@1, Recall@5, Recall@10, RR, nDCG@10]
    
    grid_results = []
    os.makedirs(os.path.join(CURRENT_DIR, "eval_runs_grid"), exist_ok=True)

    for alpha in grid_alphas:
        trec_lines = []
        
        # Считаем взвешенную сумму по кэшированным результатам
        for item in cached_search_results:
            qid = item["qid"]
            scores = {}
            
            for hit in item["img_hits"]:
                doc_id = hit.payload.get("original_hash_id", hit.id)
                scores[doc_id] = scores.get(doc_id, 0.0) + alpha * hit.score
                
            for hit in item["txt_hits"]:
                doc_id = hit.payload.get("original_hash_id", hit.id)
                scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 - alpha) * hit.score
                
            sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
            
            for rank, (doc_id, score) in enumerate(sorted_candidates, start=1):
                trec_lines.append(f"{qid} Q0 {doc_id} {rank} {score:.4f} weighted_alpha_{alpha}")
                
        # Записываем временный run-файл
        run_path = f"eval_runs_grid/run_alpha_{alpha:.1f}.txt"
        with open(run_path, "w", encoding="utf-8") as f:
            f.write("\n".join(trec_lines) + "\n")
            
        # Оцениваем
        run = list(ir_measures.read_trec_run(run_path))
        agg_metrics = ir_measures.calc_aggregate(measures, qrels, run)
        
        # Сохраняем агрегированные показатели
        result_entry = {"alpha (image weight)": alpha, "1 - alpha (text weight)": round(1.0 - alpha, 1)}
        for m, val in agg_metrics.items():
            result_entry[str(m)] = round(val, 4)
        grid_results.append(result_entry)

    # Вывод красивой сравнительной таблицы
    df_results = pd.DataFrame(grid_results)
    
    df_results = df_results.sort_values(by="RR", ascending=False).reset_index(drop=True)
    
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ СЕТЧАТОГО ПОИСКА (GRID SEARCH) ДЛЯ ALPHA")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)
    
    best_row = df_results.iloc[0]
    best_alpha = best_row["alpha (image weight)"]
    print(f"\n🏆 РЕКОМЕНДУЕМОЕ ЗНАЧЕНИЕ ALPHA: {best_alpha}")
    print(f"При alpha = {best_alpha} (веса: картинка {best_alpha*100:.0f}%, текст {(1-best_alpha)*100:.0f}%):")
    print(f"  -> MRR (RR)   : {best_row['RR']:.4f}")
    print(f"  -> nDCG@10    : {best_row['nDCG@10']:.4f}")
    print(f"  -> Recall@10  : {best_row['R@10']:.4f}")
    print("\nПроверьте run-файлы в папке 'eval_runs_grid/'")


if __name__ == "__main__":
    main()