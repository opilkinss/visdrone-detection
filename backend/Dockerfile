FROM python:3.11-slim

WORKDIR /app

# Системные зависимости для opencv/pillow
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

COPY . .

# Hugging Face Spaces ожидает порт 7860 по умолчанию
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
