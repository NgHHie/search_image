# Sử dụng image chính thức của Python
FROM python:3.11-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép các file yêu cầu vào container
COPY requirements.txt ./
COPY . .

# Cài đặt các thư viện cần thiết
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && pip install --no-cache-dir -r requirements.txt

# Expose cổng mà ứng dụng sẽ chạy
EXPOSE 5000

# Lệnh để chạy ứng dụng Flask
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "main:app"]
