"""Строит validation set для поиска (known-item retrieval).

Идея: нет ручных судей → сэмплим мемы как тест-запросы, из текста каждого делаем
короткий «человекоподобный» запрос, а релевантный документ — сам этот мем (1 relevant).
Стратифицированно по источникам. Индекс при этом = весь корпус (мем остаётся в базе).

Выход (в eval/, версионируется — это ground truth):
  eval/queries.jsonl  — {qid, query, rel_id, source}
  eval/queries.tsv    — qid <tab> query          (для загрузки в поиск)
  eval/qrels.txt      — qid 0 doc_id 1            (TREC-формат для pytrec_eval / ir_measures)

Оговорка для защиты: запросы синтетические (деривация из описаний мемов) — это прокси
реального юзера. Можно докинуть рукописный набор запросов сверху.
"""
import os
import re
import json
import random
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED = os.path.join(PROJECT_ROOT, "meme_dataset", "merged.zip")
OUT_DIR = os.path.join(PROJECT_ROOT, "eval")

SAMPLE = {"reddit": 500, "phone": 200, "imgflip": 1300}   # сколько запросов с каждого источника
MIN_WORDS = 4          # мем годится в запрос, если из него выходит >= стольких слов
MAX_WORDS = 12         # длина запроса (как у живого юзера)
SEED = 42

# срезаем memecap-боилерплейт, чтобы запрос был короче и «живее», а не копией поля meaning
BOILER = re.compile(
    r"^\s*(the\s+)?meme(\s+poster)?(\s+is)?(\s+trying\s+to)?"
    r"(\s+convey|\s+saying|\s+stating|\s+telling|\s+expressing)?"
    r"(\s+that|\s+everyone|\s+us|\s+everybody)?[:,]?\s*", re.I)


def make_query(rec):
    if rec["source"] == "imgflip":
        base = rec.get("caption_text") or rec.get("title") or ""
    else:
        base = rec.get("meaning") or rec.get("caption_text") or rec.get("title") or ""
    base = BOILER.sub("", base).strip()
    words = base.split()
    if len(words) < MIN_WORDS:
        return None
    return " ".join(words[:MAX_WORDS]).lower()


def main():
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    with zipfile.ZipFile(MERGED) as z:
        meta = [json.loads(l) for l in
                z.read("metadata.jsonl").decode("utf-8").split("\n") if l.strip()]

    by_src = {}
    for m in meta:
        by_src.setdefault(m["source"], []).append(m)

    queries = []
    seen_q = set()
    for src, n in SAMPLE.items():
        pool = by_src.get(src, [])
        random.shuffle(pool)
        picked = 0
        for rec in pool:
            if picked >= n:
                break
            q = make_query(rec)
            if not q or q in seen_q:      # пропускаем пустые и дубли-запросы (ambiguous relevance)
                continue
            seen_q.add(q)
            queries.append({"qid": "q%05d" % len(queries),
                            "query": q, "rel_id": rec["id"], "source": src})
            picked += 1
        print(f"{src}: запросов {picked} (просили {n}, в пуле {len(pool)})")

    with open(os.path.join(OUT_DIR, "queries.jsonl"), "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_DIR, "queries.tsv"), "w", encoding="utf-8") as f:
        for q in queries:
            f.write("%s\t%s\n" % (q["qid"], q["query"]))
    with open(os.path.join(OUT_DIR, "qrels.txt"), "w", encoding="utf-8") as f:
        for q in queries:
            f.write("%s 0 %s 1\n" % (q["qid"], q["rel_id"]))

    print("\nвсего запросов:", len(queries))
    print("файлы в:", OUT_DIR, "(queries.jsonl / queries.tsv / qrels.txt)")


if __name__ == "__main__":
    main()
