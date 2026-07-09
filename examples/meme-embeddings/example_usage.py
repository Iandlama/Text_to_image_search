"""
example_usage.py
=================
Демонстрация использования meme_embeddings как готовой библиотеки.

"""
import dotenv
dotenv.load_dotenv()

import numpy as np
from meme_embeddings import create_embedder, EmbedderConfig, EmbedderType, ModelSpec, ProjectionConfig


# ---------------------------------------------------------------------------
# 1) Простейший случай: готовый пресет, только текст (вариант B, текстовая часть)
# ---------------------------------------------------------------------------

embedder = create_embedder("multilingual-e5-large")

meme_texts = [
    "программист чинит один баг и случайно ломает три других",
    "понедельник утро, кофе ещё не подействовал",
]

# индексация мемов (passage_prefix подставится автоматически)
meme_vectors = embedder.encode(texts=meme_texts)
print("meme_vectors:", meme_vectors.shape, "norm:", np.linalg.norm(meme_vectors[0]))

# запрос пользователя (query_prefix подставится автоматически)
query_vector = embedder.encode_query("мем про понедельник и кофе")

# итоговый поиск - обычный cosine similarity (векторы уже L2-нормализованы)
scores = meme_vectors @ query_vector.T
print("similarity scores:", scores.ravel())


# ---------------------------------------------------------------------------
# 2) Вариант A: CLIP text encoder напрямую из пресета
# ---------------------------------------------------------------------------

clip_embedder = create_embedder("jina-clip-v2")
clip_vectors = clip_embedder.encode(texts=["кот в костюме сложного жизненного выбора"])
print("clip_vectors:", clip_vectors.shape)  # (1, 512) - обрезано до target_dim


# ---------------------------------------------------------------------------
# 3) Вариант B целиком: картинка + OCR -> Gemma-описание -> e5-эмбеддинг
# ---------------------------------------------------------------------------

mm_embedder = create_embedder("gemma-caption+e5")

image_paths = ["./meme.jpg"]
ocr_texts = ["Do I look like I jnow what a\n\"jay peg\" is?"]

# сразу эмбеддинг (внутри: генерация rich_description + кодирование)
description_vectors = mm_embedder.encode(images=image_paths, ocr_texts=ocr_texts)
print("description_vectors:", description_vectors.shape)

# при желании можно получить сами тексты описаний - например, для QA/логов (п.3.4 репорта)
descriptions = mm_embedder.last_descriptions(image_paths, ocr_texts)
print("rich_description:", descriptions[0])


# ---------------------------------------------------------------------------
# 4) Свой конфиг вместо пресета: другая модель, target_dim=256, random-проекция
# ---------------------------------------------------------------------------

custom_config = EmbedderConfig(
    embedder_type=EmbedderType.TEXT,
    model=ModelSpec(
        name="my-e5-small",
        hf_id="intfloat/multilingual-e5-small",
        native_dim=384,
        device="cpu",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    projection=ProjectionConfig(enabled=True, target_dim=256, method="random"),
    model_version="me5-small-custom@1",
)
small_embedder = create_embedder(custom_config)
print("small embedder output_dim:", small_embedder.output_dim)  # 256


# ---------------------------------------------------------------------------
# 5) Полностью кастомная модель / внешний API - без готовых реализаций
# ---------------------------------------------------------------------------

def my_encode_fn(texts):
    # например, вызов вашего внутреннего сервиса/API
    return np.random.randn(len(texts), 512).astype(np.float32)

custom_model_config = EmbedderConfig(
    embedder_type=EmbedderType.CUSTOM,
    model=ModelSpec(name="internal-api-model", hf_id="n/a", native_dim=512, device="cpu"),
    projection=ProjectionConfig(enabled=False),
    model_version="internal-api@1",
)
custom_embedder = create_embedder(custom_model_config, encode_fn=my_encode_fn)
print("custom embedder:", custom_embedder.encode(texts=["любой текст"]).shape)


# ---------------------------------------------------------------------------
# 6) Метаданные для индекса (п.2.4 репорта: обязательный логинг версий)
# ---------------------------------------------------------------------------

metadata = {
    "model_version": embedder.config.model_version,
    "preprocessing_version": embedder.config.preprocessing_version,
    "config_fingerprint": embedder.config.fingerprint(),
    "output_dim": embedder.output_dim,
}
print("metadata to store per vector:", metadata)
