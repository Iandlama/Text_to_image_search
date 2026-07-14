FROM python:3.12-slim

WORKDIR /workspace

# Системные зависимости (добавлен curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

# 1. Обновляем pip
RUN pip install --no-cache-dir --upgrade pip

# 2. Устанавливаем CPU-версии PyTorch И Torchvision
RUN pip install --no-cache-dir --default-timeout=1000 --retries 10 \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Очищаем requirements.txt от конфликтующих библиотек и ставим остальные
COPY requirements.txt .
RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/torchvision/d' requirements.txt && \
    sed -i '/transformers/d' requirements.txt && \
    pip install --no-cache-dir --default-timeout=1000 --retries 10 -r requirements.txt

# 4. Устанавливаем СОВМЕСТИМУЮ версию transformers (4.43.3) и сопутствующие пакеты
RUN pip install --no-cache-dir -U \
    huggingface_hub "transformers==4.43.3" accelerate fastapi uvicorn jinja2 timm einops Pillow

# Включаем мгновенный вывод логов без задержек и буферизации
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

# Копируем код приложения и автоматизации
COPY app/ ./app/
COPY meme_embed/ ./meme_embed/
COPY init_db.py .
COPY upload_data.py .
COPY entrypoint.sh .

# Даем права на выполнение скрипта автоматизации
RUN chmod +x entrypoint.sh

EXPOSE 8000

# Точка входа теперь запускает наш сценарий
ENTRYPOINT ["./entrypoint.sh"]