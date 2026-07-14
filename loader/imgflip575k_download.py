"""Слайс ImgFlip575K -> картинки + json в схеме проекта (source: "imgflip").

Балансированно: топ CAP_PER_TEMPLATE по votes на каждый из ~100 шаблонов
(=> сохраняет вариативность оригинала и отсеивает шлак по голосам — это и есть
"justify the slice" из требований). Качает готовые картинки, webp, md5-id.

Устойчив к обрывам: пишет чанк-zip'ы на диск, при перезапуске пропускает
уже скачанные (по src_url из готовых чанков). Многопоточно.

ПРЕДВАРИТЕЛЬНО (один раз) склонировать датасет:
    git clone --depth 1 https://github.com/schesa/ImgFlip575K_Dataset.git
и указать путь в MEMES_DIR.
"""
import os
import io
import re
import json
import glob
import time
import hashlib
import zipfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# путь к склонированному ImgFlip575K (папка dataset/memes)
MEMES_DIR = os.environ.get("IMGFLIP_MEMES", os.path.join(
    "/private/tmp/claude-501/-Users-anton-Codes-DL-for-search-project-Text-to-image-search"
    "/f5026deb-5fa7-4037-a14f-9059e549127c/scratchpad",
    "ImgFlip575K_Dataset", "dataset", "memes"))

OUT_DIR   = os.path.join(PROJECT_ROOT, "meme_dataset", "imgflip_chunks")
CAP_PER_TEMPLATE = int(os.environ.get("CAP", 450))   # сколько топ-votes брать с шаблона
CHUNK_SIZE       = 500
CONCURRENCY      = 8            # мягче к CDN, меньше rate-limit
TIMEOUT          = 15
HEADERS = {"User-Agent": "Mozilla/5.0 (research dataset collection)"}

# сессия с авто-ретраями/backoff на 429/5xx и переиспользованием соединений
_session = requests.Session()
_retry = Retry(total=4, backoff_factor=0.6,
               status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
_session.mount("https://", HTTPAdapter(max_retries=_retry, pool_maxsize=CONCURRENCY * 2))


def votes_of(rec):
    v = str(rec.get("metadata", {}).get("img-votes", "0")).replace(",", "").strip()
    return int(v) if v.isdigit() else 0


def build_worklist():
    """[(url, caption_text, title, template, votes)] — топ CAP по каждому шаблону."""
    work = []
    for path in sorted(glob.glob(os.path.join(MEMES_DIR, "*.json"))):
        tmpl = os.path.splitext(os.path.basename(path))[0].replace("-", " ")
        recs = json.load(open(path, encoding="utf-8"))
        recs.sort(key=votes_of, reverse=True)
        for r in recs[:CAP_PER_TEMPLATE]:
            url = r.get("url")
            if not url:
                continue
            caption = " ".join(b.strip() for b in r.get("boxes", []) if b and b.strip())
            title = r.get("metadata", {}).get("title") or tmpl
            work.append((url, caption, title.strip(), tmpl, votes_of(r)))
    return work


def load_done():
    """src_url и id из уже записанных чанков (точка восстановления)."""
    done_urls, done_ids = set(), set()
    for c in sorted(glob.glob(os.path.join(OUT_DIR, "chunk_*.zip"))):
        try:
            with zipfile.ZipFile(c) as z:
                for line in z.read("metadata.jsonl").decode("utf-8").splitlines():
                    if line.strip():
                        o = json.loads(line)
                        done_urls.add(o.get("src_url"))
                        done_ids.add(o["id"])
        except Exception:
            pass
    return done_urls, done_ids


def fetch(item):
    url, caption, title, tmpl, votes = item
    try:
        r = _session.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "WEBP", quality=90)
        b = buf.getvalue()
        h = hashlib.md5(b).hexdigest()
        rec = {"id": h, "source": "imgflip", "title": title,
               "image_desc": tmpl, "meaning": "", "caption_text": caption,
               "template": tmpl, "votes": votes, "src_url": url}
        return h, b, rec
    except Exception:
        return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    work = build_worklist()
    done_urls, seen_ids = load_done()
    todo = [w for w in work if w[0] not in done_urls]
    chunk_idx = len(glob.glob(os.path.join(OUT_DIR, "chunk_*.zip"))) + 1
    print(f"всего в слайсе: {len(work)} | уже готово: {len(work) - len(todo)} | качать: {len(todo)}")
    print(f"следующий чанк: {chunk_idx}, CAP={CAP_PER_TEMPLATE}")

    buffer = []
    ok = fail = dup = 0
    t0 = time.time()

    def flush():
        nonlocal chunk_idx, buffer
        if not buffer:
            return
        tmp = os.path.join(OUT_DIR, ".tmp_chunk.zip")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            meta = []
            for h, b, rec in buffer:
                z.writestr("images/%s.webp" % h, b)
                meta.append(json.dumps(rec, ensure_ascii=False))
            z.writestr("metadata.jsonl", "\n".join(meta) + "\n")
        os.replace(tmp, os.path.join(OUT_DIR, "chunk_%05d.zip" % chunk_idx))
        chunk_idx += 1
        buffer = []

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = {ex.submit(fetch, w): w for w in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res is None:
                fail += 1
            else:
                h, b, rec = res
                if h in seen_ids:
                    dup += 1
                else:
                    seen_ids.add(h)
                    buffer.append((h, b, rec))
                    ok += 1
                    if len(buffer) >= CHUNK_SIZE:
                        flush()
            if i % 1000 == 0:
                rate = i / max(time.time() - t0, 1)
                print(f"  {i}/{len(todo)} | ok {ok} dup {dup} fail {fail} | {rate:.1f}/s")
    flush()
    print(f"готово. ok {ok}, дублей {dup}, ошибок {fail}. чанки в {OUT_DIR}")


if __name__ == "__main__":
    main()
