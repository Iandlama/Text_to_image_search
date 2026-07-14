import os
import sys

# настройка относительных путей
#CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

#if ROOT_DIR not in sys.path:
#    sys.path.insert(0, ROOT_DIR)

#IMAGES_DIR = os.path.join(ROOT_DIR, "data", "images")

IMAGES_DIR = os.path.join(os.getcwd(), "data", "images") # директория с мемами

import torch
from fastapi import FastAPI, HTTPException, Query
from qdrant_client import QdrantClient
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from meme_embed import load_embedder

app = FastAPI(
    title="Meme Search Fusion API", 
    description="Гибридный поиск мемов: слияние визуального CLIP-контекста и OCR семантики через Jina CLIP",
    version="1.0.0"
)

if os.path.exists(IMAGES_DIR):
    app.mount("/static_images", StaticFiles(directory=IMAGES_DIR), name="static_images")

WEIGHTED_ALPHA = 0.3 # настройка alpha параметра для weighted sum
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", 6334)) # Используем gRPC порт 6334 (быстрее, стабильнее http)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "meme_collection_v1") # название коллекции Qdrant

qdrant_client = QdrantClient(host=QDRANT_HOST, grpc_port=6334, prefer_grpc=True)
#qdrant_client = QdrantClient(host=QDRANT_HOST, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True)

# подготовка jinja-clip v2
try:
    meme_embedder = load_embedder() 
    meme_embedder.warmup() # прогреваем модель
except Exception as e:
    raise SystemExit(f"Критическая ошибка при инициализации Jina CLIP: {e}")


# слияние рейтингов через weighted sum вместо RRF - для контроля веса картинки и текста в поиске
# alpha - вес визуальной части
def weighted_sum_fusion(visual_hits, text_hits, alpha: float = WEIGHTED_ALPHA, top_n: int = 5) -> list[dict]:
    scores = {}

    # визуальный поиск
    for hit in visual_hits:
        if hit.id not in scores:
            scores[hit.id] = {"hit": hit, "score": 0.0}
        scores[hit.id]["score"] += alpha * hit.score

    # текстовый поиск (OCR)
    for hit in text_hits:
        if hit.id not in scores:
            scores[hit.id] = {"hit": hit, "score": 0.0}
        scores[hit.id]["score"] += (1.0 - alpha) * hit.score

    sorted_candidates = sorted(scores.values(), key=lambda x: x["score"], reverse=True)

    return [
        {
            "id": item["hit"].id,
            "score": round(item["score"], 4),
            "payload": item["hit"].payload
        }
        for item in sorted_candidates[:top_n]
    ]

# ----------------------------- Настройка endpoints ---------------------------------

# Поиск по документам (картинкам) - без отображения изображений
@app.get("/search")
async def search_memes(
    q: str = Query(..., min_length=1, description="Текстовый запрос пользователя"),
    alpha: float = Query(WEIGHTED_ALPHA, ge=0.0, le=1.0, description="Вес визуального сходства в слиянии")
):
    cleaned_query = q.strip()
    if not cleaned_query:
        raise HTTPException(status_code=400, detail="Запрос не может быть пустым.")

    try:
        # Безопасное извлечение эмбеддингов с учетом специки Jina CLIP v2
        if meme_embedder.model is not None and hasattr(meme_embedder.model, "encode_text"):
            kw = {"task": "retrieval.query", "device": meme_embedder.device, "normalize_embeddings": False}
            
            if meme_embedder.config.resolved_dim():
                kw["truncate_dim"] = meme_embedder.config.resolved_dim()

            # Выполняем инференс в безопасном режиме контекста PyTorch
            with torch.inference_mode():
                query_features = meme_embedder.model.encode_text([cleaned_query], **kw)
            
            query_vector = meme_embedder._postprocess(query_features, normalize=True)[0].tolist()
            text_query_vector = query_vector
            visual_query_vector = query_vector
        else:
            vector = meme_embedder.encode_text(cleaned_query)
            text_query_vector = vector.tolist()
            visual_query_vector = vector.tolist()

        # Поиск по визуальному признаку
        visual_res = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=visual_query_vector,
            using="vector_image",
            limit=20,
            with_payload=True
        )
        visual_hits = visual_res.points

        # Поиск по текстовому признаку (OCR)
        text_res = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=text_query_vector,
            using="vector_text",
            limit=20,
            with_payload=True
        )
        text_hits = text_res.points

        # Слияние списков через Weighted Sum
        top_5_memes = weighted_sum_fusion(
            visual_hits=visual_hits, 
            text_hits=text_hits, 
            alpha=alpha, 
            top_n=5
        )

        return {
            "query": cleaned_query,
            "results": top_5_memes
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Search Error: {str(e)}")

# Поиск по документам с отображением карточек - мемов
@app.get("/search/view", response_class=HTMLResponse)
async def search_memes_visual(
    q: str = Query(..., min_length=1, description="Текстовый запрос пользователя"),
    alpha: float = Query(WEIGHTED_ALPHA, ge=0.0, le=1.0, description="Вес визуального сходства в слиянии")
):
    cleaned_query = q.strip()
    if not cleaned_query:
        return "<h1>Запрос не может быть пустым.</h1>"

    try:
        if meme_embedder.model is not None and hasattr(meme_embedder.model, "encode_text"):
            kw = {"task": "retrieval.query", "device": meme_embedder.device, "normalize_embeddings": False}
            if meme_embedder.config.resolved_dim():
                kw["truncate_dim"] = meme_embedder.config.resolved_dim()
            with torch.inference_mode():
                query_features = meme_embedder.model.encode_text([cleaned_query], **kw)
            query_vector = meme_embedder._postprocess(query_features, normalize=True)[0].tolist()
            text_query_vector = query_vector
            visual_query_vector = query_vector
        else:
            vector = meme_embedder.encode_text(cleaned_query)
            text_query_vector = vector.tolist()
            visual_query_vector = vector.tolist()

        visual_res = qdrant_client.query_points(
            collection_name=COLLECTION_NAME, query=visual_query_vector, using="vector_image", limit=20, with_payload=True
        )
        text_res = qdrant_client.query_points(
            collection_name=COLLECTION_NAME, query=text_query_vector, using="vector_text", limit=20, with_payload=True
        )

        # Слияние Weighted Sum
        top_5_memes = weighted_sum_fusion(
            visual_hits=visual_res.points, 
            text_hits=text_res.points, 
            alpha=alpha, 
            top_n=5
        )

        # Шаблон карточек - отображения мемов
        html_content = f"""
        <html>
            <head>
                <title>Meme Search Results</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f4f4f9; }}
                    .gallery {{ display: flex; gap: 20px; flex-wrap: wrap; margin-top: 20px; }}
                    .card {{ background: white; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); padding: 15px; width: 280px; }}
                    .card img {{ width: 100%; height: auto; border-radius: 4px; object-fit: contain; max-height: 250px; }}
                    .score {{ color: #2ecc71; font-weight: bold; margin: 10px 0 5px 0; }}
                    .ocr {{ font-size: 13px; color: #555; font-style: italic; }}
                </style>
            </head>
            <body>
                <h2>Результаты поиска по запросу: &ldquo;{cleaned_query}&rdquo; (Weighted Sum, &alpha;={alpha})</h2>
                <div class="gallery">
        """

        for meme in top_5_memes:
            file_name = meme["payload"].get("file_name")
            ocr_text = meme["payload"].get("ocr_text", "Нет текста")
            score = meme["score"]
            
            html_content += f"""
                <div class="card">
                    <img src="/static_images/{file_name}" alt="meme" onerror="this.src='https://placehold.co/280x250?text=Image+Not+Found'"/>
                    <div class="score">Weighted Score: {score}</div>
                    <div class="ocr"><b>Description:</b> {ocr_text}</div>
                </div>
            """
            
        html_content += """
                </div>
            </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=200)

    except Exception as e:
        return f"<h3>Внутренняя ошибка поиска: {str(e)}</h3>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)