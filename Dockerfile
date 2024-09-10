FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
COPY . .

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && pip install --no-cache-dir -r requirements.txt

# Sao chép các file mô hình vào thư mục /app/weight trong container
COPY weight/lastv3.pt /app/weight/lastv3.pt
COPY weight/resnet50_model.keras /app/weight/resnet50_model.keras

EXPOSE 5000

CMD ["gunicorn", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:5000", "--timeout", "600", "--preload", "main:app"]
