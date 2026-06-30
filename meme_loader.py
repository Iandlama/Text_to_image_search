import os
import json
import hashlib
import zipfile
from datasets import load_dataset
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
import ssl

# Отключаем проверку SSL, если Hugging Face ругается
ssl._create_default_https_context = ssl._create_unverified_context

OUTPUT_DIR = 'meme_dataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- НАСТРОЙКИ РАЗДЕЛЬНОГО ЗИПИРОВАНИЯ ---
CHUNK_SIZE = 6000  # Количество мемов в одном zip-архиве
processed_count = 0
chunk_idx = 1
current_zip = None
metadata_lines = []

print("Загрузка датасета...")
# Стриминг включен, память не забивается
ds = load_dataset("Leonardo6/memecap", split="train", streaming=True)
print("Датасет загружен. Начинаем обработку...")


def close_current_chunk(zip_obj, lines, idx):
    """Вшивает метадату внутрь архива и закрывает его."""
    if zip_obj is not None:
        # Сохраняем metadata.jsonl прямо в корень текущего архива
        metadata_content = "\n".join(lines) + "\n"
        zip_obj.writestr("metadata.jsonl", metadata_content)
        zip_obj.close()
        print(f" Архив meme_dataset_part_{idx}.zip успешно создан!")


try:
    for item in tqdm(ds, desc="Processing memes"):
        images = item.get('images', [])
        if not images:
            continue
        image = images[0] if isinstance(images, list) else images

        # Инициализируем или меняем архив при достижении лимита чанка
        if processed_count % CHUNK_SIZE == 0:
            if current_zip is not None:
                close_current_chunk(current_zip, metadata_lines, chunk_idx)
                chunk_idx += 1
                metadata_lines = []

            zip_path = os.path.join(
                OUTPUT_DIR, f"meme_dataset.zip")
            current_zip = zipfile.ZipFile(
                zip_path, 'w', compression=zipfile.ZIP_DEFLATED)

        # Забираем готовый текст/описание из датасета
        caption = item.get('messages', "")

        # Конвертируем картинку в WEBP прямо в оперативной памяти
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='WEBP')
        img_bytes = img_byte_arr.getvalue()

        # Генерируем хэш от байт картинки
        img_hash = hashlib.md5(img_bytes).hexdigest()
        img_filename = f"images/{img_hash}.webp"

        # Записываем байты картинки в ZIP без сохранения на диск
        current_zip.writestr(img_filename, img_bytes)

        # Формируем строку метаданных для текущего архива
        entry = {
            "id": img_hash,
            "caption": caption
        }
        metadata_lines.append(json.dumps(entry, ensure_ascii=False))

        processed_count += 1

    # Закрываем последний оставшийся чанк
    if current_zip is not None:
        close_current_chunk(current_zip, metadata_lines, chunk_idx)

except Exception as e:
    if current_zip is not None:
        current_zip.close()
    raise e

print(f"\n Всё готово! Ищи разделенные архивы в папке: {OUTPUT_DIR}")