"""Пересобирает metadata.jsonl внутри уже скачанного meme_dataset.zip.

Старый формат:  {"id": ..., "caption": [{user prompt}, {assistant}]}
Новый формат:   {"id": ..., "title": ..., "image_desc": ..., "meaning": ...}

Картинки не трогаются и заново не качаются. Новый zip пишется во временный
файл и атомарно заменяет исходный (os.replace) — при сбое оригинал цел.
"""
import os
import json
import zipfile
import tempfile

from meme_utils import parse_caption

# Путь привязан к корню проекта (папка над loader/), а не к cwd —
# скрипт можно запускать из любого места.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_PATH = os.path.join(PROJECT_ROOT, 'meme_dataset', 'meme_dataset.zip')
META_NAME = 'metadata.jsonl'


def main() -> None:
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(f'Не найден архив: {ZIP_PATH}')

    stats = {'total': 0, 'title_ok': 0, 'desc_ok': 0, 'meaning_ok': 0}

    with zipfile.ZipFile(ZIP_PATH, 'r') as zin:
        names = zin.namelist()
        raw = zin.read(META_NAME).decode('utf-8')

        new_lines = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            title, image_desc, meaning = parse_caption(obj.get('caption'))
            stats['total'] += 1
            stats['title_ok'] += title is not None
            stats['desc_ok'] += image_desc is not None
            stats['meaning_ok'] += bool(meaning)
            new_lines.append(json.dumps({
                'id': obj['id'],
                'title': title,
                'image_desc': image_desc,
                'meaning': meaning,
            }, ensure_ascii=False))
        new_meta = '\n'.join(new_lines) + '\n'

        # Пишем свежий архив рядом, затем атомарно заменяем оригинал
        fd, tmp_path = tempfile.mkstemp(suffix='.zip', dir=os.path.dirname(ZIP_PATH))
        os.close(fd)
        try:
            with zipfile.ZipFile(tmp_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
                for name in names:
                    if name == META_NAME:
                        continue
                    zout.writestr(name, zin.read(name))
                zout.writestr(META_NAME, new_meta)
        except Exception:
            os.remove(tmp_path)
            raise

    os.replace(tmp_path, ZIP_PATH)

    print(f"Готово. Обновлено записей: {stats['total']}")
    print(f"  с title:    {stats['title_ok']}")
    print(f"  с image_desc:{stats['desc_ok']}")
    print(f"  с meaning:  {stats['meaning_ok']}")
    miss = stats['total'] - min(stats['title_ok'], stats['desc_ok'])
    if miss:
        print(f"  ВНИМАНИЕ: не распарсено полей у {miss} записей (title=None/desc=None)")


if __name__ == '__main__':
    main()
