#!/bin/bash
# Завершать выполнение при любой ошибке
set -e

echo "=== [1/4] Ожидание готовности Qdrant ==="
# Ждем, пока Qdrant поднимется и начнет отвечать на запросы
until curl -s "http://${QDRANT_HOST}:6333/readyz" | grep -q "all shards are ready"; do
  echo "Qdrant еще не готов, ждем 2 секунды..."
  sleep 2
done
echo "Qdrant успешно запущен!"

echo "=== [2/4] Проверка и создание коллекции в Qdrant ==="
python -m app.init_db || python init_db.py

echo "=== [3/4] Импорт датасета из Parquet в Qdrant ==="
python -m app.upload_data || python upload_data.py

echo "=== [4/4] Запуск веб-сервера FastAPI ==="
exec python -m uvicorn app.app:app --host 0.0.0.0 --port 8000