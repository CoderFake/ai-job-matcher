import os
from typing import List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Cài đặt ứng dụng sử dụng pydantic"""

    # Thông tin ứng dụng
    PROJECT_NAME: str = "AI Job Matcher"
    PROJECT_DESCRIPTION: str = "Hệ thống phân tích CV và tìm kiếm việc làm tự động với AI"
    VERSION: str = "0.1.0"

    # Cài đặt API
    API_PREFIX: str = "/api"

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Thư mục lưu trữ tệp tạm thời
    TEMP_DIR: str = "temp"

    # Cài đặt mô hình AI
    MODEL_DIR: str = "models"
    USE_GPU: bool = False
    AUTO_CONFIG_GPU: bool = True

    # Cơ sở dữ liệu
    DATABASE_URL: str = "sqlite:///./ai_job_matcher.db"

    # Cấu hình tìm kiếm việc làm
    MAX_SEARCH_RESULTS: int = 20
    SEARCH_TIMEOUT: int = 60  # Giây

    # Cài đặt tác nhân web
    USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"

    # Tìm kiếm trang web
    JOB_SITES: List[str] = [
        "linkedin.com",
        "topcv.vn",
        "careerbuilder.vn",
        "vietnamworks.com",
        "careerviet.vn"
    ]

    # Add these two missing fields
    ENABLE_DOCS: bool = False
    DEBUG: bool = False

    @field_validator("USE_GPU")
    def validate_gpu(cls, v):
        """Xác nhận nếu GPU có thể được sử dụng"""
        if v:
            try:
                import torch
                return torch.cuda.is_available()
            except ImportError:
                return False
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True


# Tạo đối tượng settings toàn cục
settings = Settings()

# Tạo các thư mục cần thiết nếu chúng không tồn tại
os.makedirs(settings.TEMP_DIR, exist_ok=True)
os.makedirs(settings.MODEL_DIR, exist_ok=True)