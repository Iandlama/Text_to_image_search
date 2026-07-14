FROM python:3.12-slim

WORKDIR /workspace

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# 1. Установка базовых библиотек (добавили transformers и accelerate)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir jinja2 typing-extensions filelock networkx fsspec uvicorn fastapi \
    transformers accelerate

# 2. Установка PyTorch (cpu-версия)
RUN pip install --no-cache-dir --default-timeout=1000 --retries 10 torch --index-url https://download.pytorch.org/whl/cpu

# 3. Установка остальных зависимостей из файла
COPY requirements.txt .
RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/transformers/d' requirements.txt && \
    sed -i '/accelerate/d' requirements.txt && \
    pip install --no-cache-dir --default-timeout=1000 --retries 10 -r requirements.txt

# Копируем остальной код
COPY . .

ENV PYTHONPATH=/workspace

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]