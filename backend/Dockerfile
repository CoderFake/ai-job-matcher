# Sử dụng multi-stage build để giảm kích thước image
FROM python:3.10-slim as builder

# Cài đặt các gói build-essential để biên dịch thư viện Python
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép requirements.txt trước để tận dụng cache Docker
COPY requirements.txt .

# Tạo và kích hoạt môi trường ảo
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Cài đặt các gói Python trong môi trường ảo
RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Sử dụng image chính
FROM python:3.9-slim

# Thiết lập các biến môi trường
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Cài đặt các phụ thuộc hệ thống cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-vie \
    poppler-utils \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Sao chép môi trường ảo từ builder
COPY --from=builder /opt/venv /opt/venv

# Tạo thư mục ứng dụng
WORKDIR /app

# Sao chép mã nguồn
COPY . .

# Tạo các thư mục cần thiết
RUN mkdir -p logs models temp

# Kiểm tra cài đặt Tesseract
RUN tesseract --version || (echo "Tesseract not installed properly" && exit 1)

# Kiểm tra xem thư viện OpenCV cần thiết cho OCR có hoạt động không
RUN python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')" || echo "WARNING: OpenCV might not be working properly"

# Kiểm tra ffmpeg
RUN ffmpeg -version || echo "WARNING: ffmpeg might not be working properly"

# Người dùng không phải root để bảo mật hơn
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app /opt/venv logs models temp
USER appuser

# Đặt cổng
EXPOSE 8000

# Lệnh khởi động ứng dụng
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Cung cấp thông tin sức khỏe
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1