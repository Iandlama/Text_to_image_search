"""Быстрый NSFW-фильтр ТОЛЬКО для reddit-мемов (MemeCap). Локально, на CPU.

Слой 1 (мгновенный, без зависимостей): по ключевым словам в title/image_desc/meaning.
Слой 2 (опционально): NudeNet по картинкам — ловит наготу, которую текст не описал.
                      Включается, если установлен `pip install nudenet`.

Выход:
  meme_dataset/reddit_nsfw_ids.txt   — блоклист id (их исключит merge_dataset.py)
  meme_dataset/reddit_flagged.jsonl  — что именно помечено (для глазами-проверки)
"""
import os
import re
import json
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT_ROOT, "meme_dataset")
MEMECAP_ZIP = os.path.join(DATA, "meme_dataset.zip")
BLOCKLIST = os.path.join(DATA, "reddit_nsfw_ids.txt")
FLAGGED = os.path.join(DATA, "reddit_flagged.jsonl")

USE_NUDENET = True        # если установлен nudenet — прогнать картинки, что текст пропустил
NUDE_THRESH = 0.55        # порог уверенности детектора наготы

# Явные термины (word-boundary, регистронезависимо). Список — правь под себя.
KEYWORDS = [
    "porn", "pornhub", "nsfw", "xxx", "hentai", "onlyfans", "rule 34", "rule34",
    "nude", "nudes", "naked", "nudity", "topless", "genital", "genitalia",
    "penis", "vagina", "masturbat", "ejaculat", "blowjob", "deepthroat",
    "dildo", "orgasm", "erotic", "fetish", "bdsm", "incest", "bestiality",
    "molest", "\\brape\\b", "\\bcum\\b", "\\banus\\b", "boobs", "titties",
]
PATTERN = re.compile("|".join(KEYWORDS), re.IGNORECASE)

# Явные классы NudeNet (обнажённые интимные зоны)
NUDE_CLASSES = {
    "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED", "ANUS_EXPOSED", "BUTTOCKS_EXPOSED",
}


def text_of(rec):
    return " ".join(str(rec.get(k, "")) for k in ("title", "image_desc", "meaning"))


def main():
    flagged = {}   # id -> (reason, rec)

    with zipfile.ZipFile(MEMECAP_ZIP) as zin:
        records = [json.loads(l) for l in
                   zin.read("metadata.jsonl").decode("utf-8").splitlines() if l.strip()]

        # --- Слой 1: текст ---
        for rec in records:
            m = PATTERN.search(text_of(rec))
            if m:
                flagged[rec["id"]] = ("text:" + m.group(0), rec)
        print("Слой 1 (текст): помечено", len(flagged), "из", len(records))

        # --- Слой 2: NudeNet по картинкам (опционально) ---
        if USE_NUDENET:
            try:
                from nudenet import NudeDetector
                from PIL import Image
                from io import BytesIO
                import tempfile
                det = NudeDetector()
                checked = added = 0
                for rec in records:
                    if rec["id"] in flagged:
                        continue
                    name = "images/%s.webp" % rec["id"]
                    try:
                        data = zin.read(name)
                    except KeyError:
                        continue
                    try:                                   # webp -> jpg: детектор надёжнее читает
                        img = Image.open(BytesIO(data)).convert("RGB")
                    except Exception:
                        continue
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                        img.save(tf, "JPEG")
                        tmp = tf.name
                    try:
                        dets = det.detect(tmp)
                    except Exception:
                        dets = []
                    finally:
                        os.unlink(tmp)
                    checked += 1
                    if any((d.get("class") or d.get("label")) in NUDE_CLASSES
                           and d.get("score", 0) >= NUDE_THRESH for d in dets):
                        flagged[rec["id"]] = ("image", rec)
                        added += 1
                    if checked % 500 == 0:
                        print("  ...NudeNet проверил", checked, "| добавлено", added)
                print("Слой 2 (NudeNet): проверено", checked, ", добавлено", added)
            except ImportError:
                print("Слой 2 пропущен: nudenet не установлен (pip install nudenet).")

    # --- Запись ---
    with open(BLOCKLIST, "w", encoding="utf-8") as f:
        for rid in flagged:
            f.write(rid + "\n")
    with open(FLAGGED, "w", encoding="utf-8") as f:
        for rid, (reason, rec) in flagged.items():
            rec = dict(rec, _reason=reason)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\nИТОГО помечено к удалению:", len(flagged))
    print("  блоклист :", BLOCKLIST)
    print("  на ревью :", FLAGGED, "(проверь глазами перед merge)")


if __name__ == "__main__":
    main()
