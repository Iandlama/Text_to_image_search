"""Сшивает источники в единый датасет проекта.

Источники:
  1. MemeCap ("reddit") — meme_dataset/meme_dataset.zip        (schema: id/title/image_desc/meaning)
  2. Phone (fixed)      — meme_dataset/fixed_memes/{metadata.jsonl, images/}

Выход: meme_dataset/merged.zip  (images/{id}.webp + metadata.jsonl)
Схема (утверждённая reddit-стиль + source): {id, source, title, image_desc, meaning}
Дедуп между источниками по id (md5 картинки). Phone-записи с flag из DROP_FLAGS выкидываются.
"""
import os
import re
import glob
import json
import zipfile

# юникод-разделители строк, которые ломают splitlines/JSONL (но не экранируются json)
LINE_SEPS = re.compile("[\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]+")


def clean(s):
    return LINE_SEPS.sub(" ", s).strip() if isinstance(s, str) else s

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT_ROOT, "meme_dataset")

MEMECAP_ZIP    = os.path.join(DATA, "meme_dataset.zip")
MEMECAP_SOURCE = "reddit"                 # так ты называешь MemeCap-набор; поменяй при желании
PHONE_DIR      = os.path.join(DATA, "fixed_memes")
OUT_ZIP        = os.path.join(DATA, "merged.zip")

DROP_FLAGS        = {"junk"}              # какие phone-флаги выкидывать (можно добавить "low_info")
KEEP_CAPTION_TEXT = True                  # imgflip несёт весь сигнал в caption_text — обязателен
BLOCKLIST         = os.path.join(DATA, "reddit_nsfw_ids.txt")  # NSFW-id для reddit (из filter_reddit_nsfw.py)
IMGFLIP_CHUNKS    = os.path.join(DATA, "imgflip_chunks")       # чанки из imgflip575k_download.py


def load_blocklist():
    if not os.path.exists(BLOCKLIST):
        return set()
    with open(BLOCKLIST, encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def norm(rec, source):
    out = {
        "id": rec["id"],
        "source": rec.get("source", source),
        "title": clean(rec.get("title", "")),
        "image_desc": clean(rec.get("image_desc", "")),
        "meaning": clean(rec.get("meaning", "")),
    }
    if KEEP_CAPTION_TEXT:
        out["caption_text"] = clean(rec.get("caption_text", ""))
    return out


def main():
    seen = set()
    blocklist = load_blocklist()
    stats = {"reddit": 0, "phone": 0, "imgflip": 0, "dup": 0, "flagged": 0, "no_image": 0, "nsfw": 0}

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zout:
        meta = []

        # 1) MemeCap / reddit из zip-архива
        with zipfile.ZipFile(MEMECAP_ZIP) as zin:
            names = set(n for n in zin.namelist() if n.startswith("images/"))
            for line in zin.read("metadata.jsonl").decode("utf-8").split("\n"):
                if not line.strip():
                    continue
                rec = json.loads(line)
                rid = rec["id"]
                if rid in blocklist:              # NSFW reddit — выкидываем
                    stats["nsfw"] += 1
                    continue
                if rid in seen:
                    stats["dup"] += 1
                    continue
                img = "images/%s.webp" % rid
                if img not in names:
                    stats["no_image"] += 1
                    continue
                zout.writestr(img, zin.read(img))
                meta.append(json.dumps(norm(rec, MEMECAP_SOURCE), ensure_ascii=False))
                seen.add(rid)
                stats["reddit"] += 1

        # 2) Phone / fixed из папки
        pmeta = os.path.join(PHONE_DIR, "metadata.jsonl")
        pimg = os.path.join(PHONE_DIR, "images")
        with open(pmeta, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("flag") in DROP_FLAGS:
                    stats["flagged"] += 1
                    continue
                rid = rec["id"]
                if rid in seen:
                    stats["dup"] += 1
                    continue
                src_img = os.path.join(pimg, "%s.webp" % rid)
                if not os.path.exists(src_img):
                    stats["no_image"] += 1
                    continue
                with open(src_img, "rb") as im:
                    zout.writestr("images/%s.webp" % rid, im.read())
                meta.append(json.dumps(norm(rec, "phone"), ensure_ascii=False))
                seen.add(rid)
                stats["phone"] += 1

        # 3) ImgFlip575K из чанков (imgflip575k_download.py)
        for c in sorted(glob.glob(os.path.join(IMGFLIP_CHUNKS, "chunk_*.zip"))):
            with zipfile.ZipFile(c) as zin:
                names = set(n for n in zin.namelist() if n.startswith("images/"))
                for line in zin.read("metadata.jsonl").decode("utf-8").split("\n"):
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    rid = rec["id"]
                    if rid in seen:
                        stats["dup"] += 1
                        continue
                    img = "images/%s.webp" % rid
                    if img not in names:
                        stats["no_image"] += 1
                        continue
                    zout.writestr(img, zin.read(img))
                    meta.append(json.dumps(norm(rec, "imgflip"), ensure_ascii=False))
                    seen.add(rid)
                    stats["imgflip"] += 1

        zout.writestr("metadata.jsonl", "\n".join(meta) + "\n")

    total = stats["reddit"] + stats["phone"] + stats["imgflip"]
    print("merged.zip создан:", OUT_ZIP)
    print("  reddit (memecap):", stats["reddit"])
    print("  phone           :", stats["phone"])
    print("  imgflip         :", stats["imgflip"])
    print("  ИТОГО в датасете :", total)
    print("  дублей пропущено :", stats["dup"])
    print("  по флагу выкинуто:", stats["flagged"])
    print("  NSFW выкинуто    :", stats["nsfw"])
    print("  без картинки     :", stats["no_image"])


if __name__ == "__main__":
    main()
