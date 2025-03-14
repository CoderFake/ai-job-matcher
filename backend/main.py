"""
AI Job Matcher - Ứng dụng phân tích CV và tìm kiếm việc làm tự động
Phiên bản cải tiến
"""

import logging
import json
import os
import sys
import asyncio
import time
import traceback
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.gzip import GZipMiddleware

from app.api.router import api_router
from app.core.config import get_model_config, configure_gpu, check_disk_space, get_fallback_model_path
from app.core.settings import settings
from app.core.logging import setup_logging

# Thiết lập logging
setup_logging()
logger = logging.getLogger("main")

# Kiểm tra môi trường để quyết định có bật docs hay không
ENABLE_DOCS = os.getenv("ENABLE_DOCS", "False").lower() in ("true", "1", "t")
# Trong môi trường development và test, mặc định bật docs
if os.getenv("DEBUG", "False").lower() in ("true", "1", "t"):
    ENABLE_DOCS = True

def create_application() -> FastAPI:
    """Tạo ứng dụng FastAPI"""

    application = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.VERSION,
        # Thiết lập docs_url và redoc_url dựa vào biến môi trường
        docs_url="/api/docs" if ENABLE_DOCS else None,
        redoc_url="/api/redoc" if ENABLE_DOCS else None,
        openapi_url="/api/openapi.json" if ENABLE_DOCS else None,
    )

    # Tùy chỉnh OpenAPI schema nếu bật docs
    if ENABLE_DOCS:
        def custom_openapi():
            if application.openapi_schema:
                return application.openapi_schema

            openapi_schema = get_openapi(
                title=settings.PROJECT_NAME,
                version=settings.VERSION,
                description=settings.PROJECT_DESCRIPTION,
                routes=application.routes,
            )

            # Tùy chỉnh thông tin
            openapi_schema["info"]["x-logo"] = {
                "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
            }

            # Thêm thông tin liên hệ
            openapi_schema["info"]["contact"] = {
                "name": "AI Job Matcher Team",
                "email": "support@aijobmatcher.com",
                "url": "https://aijobmatcher.com"
            }

            # Thêm tag
            openapi_schema["tags"] = [
                {
                    "name": "CV",
                    "description": "Các API liên quan đến quản lý và phân tích CV"
                },
                {
                    "name": "Jobs",
                    "description": "Các API liên quan đến tìm kiếm và quản lý công việc"
                },
                {
                    "name": "Users",
                    "description": "Các API liên quan đến quản lý người dùng"
                }
            ]

            # Thêm bảo mật (nếu cần)
            openapi_schema["components"]["securitySchemes"] = {
                "APIKeyHeader": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                }
            }

            application.openapi_schema = openapi_schema
            return application.openapi_schema

        application.openapi = custom_openapi

    # Thêm middleware GZip cho nén dữ liệu
    application.add_middleware(GZipMiddleware, minimum_size=1000)

    # Thêm middleware CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Đăng ký các routes API
    application.include_router(api_router, prefix=settings.API_PREFIX)

    # Tạo custom swagger UI docs nếu bật docs
    if ENABLE_DOCS:
        @application.get("/api/docs", include_in_schema=False)
        async def custom_swagger_ui_html():
            return get_swagger_ui_html(
                openapi_url="/api/openapi.json",
                title=f"{settings.PROJECT_NAME} - Swagger UI",
                oauth2_redirect_url="/api/docs/oauth2-redirect",
                swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
                swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
                swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
            )

        @application.get("/api/redoc", include_in_schema=False)
        async def custom_redoc_html():
            return get_redoc_html(
                openapi_url="/api/openapi.json",
                title=f"{settings.PROJECT_NAME} - ReDoc",
                redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
                redoc_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
            )

    # Trình xử lý lỗi chung
    @application.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        logger.error(f"HTTP error: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"message": exc.detail},
        )

    @application.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        # Ghi log chi tiết với traceback
        exc_traceback = traceback.format_exc()
        logger.error(f"Unhandled error: {str(exc)}\n{exc_traceback}")

        # Phản hồi an toàn cho người dùng
        return JSONResponse(
            status_code=500,
            content={"message": "Đã xảy ra lỗi không mong muốn. Vui lòng thử lại sau."},
        )

    # Middleware ghi log yêu cầu
    @application.middleware("http")
    async def log_requests(request: Request, call_next):
        # Lấy đường dẫn và phương thức
        path = request.url.path
        method = request.method
        client = request.client.host if request.client else "unknown"

        # Ghi log nếu không phải là yêu cầu static files
        if not path.startswith("/static"):
            start_time = time.time()
            logger.info(f"[{client}] {method} {path}")

            # Xử lý yêu cầu và đo thời gian
            try:
                response = await call_next(request)
                process_time = time.time() - start_time
                response.headers["X-Process-Time"] = str(process_time)

                # Ghi log kết quả
                logger.info(f"[{client}] {method} {path} - {response.status_code} ({process_time:.4f}s)")
                return response
            except Exception as e:
                # Ghi log lỗi nếu có
                process_time = time.time() - start_time
                logger.error(f"[{client}] {method} {path} - Error: {str(e)} ({process_time:.4f}s)")
                raise
        else:
            return await call_next(request)

    # Health check endpoint
    @application.get("/health", tags=["System"])
    async def health_check():
        """
        Kiểm tra sức khỏe hệ thống

        Returns:
            Dict: Thông tin sức khỏe hệ thống
        """
        # Kiểm tra các thành phần
        import psutil
        from datetime import datetime

        # Thông tin cơ bản
        health_data = {
            "status": "healthy",
            "version": settings.VERSION,
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - app_start_time,
            "system": {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
        }

        # Kiểm tra GPU nếu có
        try:
            import torch
            if torch.cuda.is_available():
                health_data["gpu"] = {
                    "available": True,
                    "name": torch.cuda.get_device_name(0),
                    "memory_allocated": torch.cuda.memory_allocated(0) / (1024**3),  # GB
                    "memory_reserved": torch.cuda.memory_reserved(0) / (1024**3)     # GB
                }
            else:
                health_data["gpu"] = {"available": False}
        except:
            health_data["gpu"] = {"available": False, "error": "GPU check failed"}

        # Kiểm tra không gian đĩa
        health_data["disk_space_sufficient"] = check_disk_space(1.0)

        # Trả về kết quả
        return health_data

    return application

# Biến lưu thời gian bắt đầu ứng dụng
app_start_time = time.time()

# Tạo ứng dụng
app = create_application()

# Cờ để kiểm tra xem các mô hình đã được tải hay chưa
models_loaded = False

@app.on_event("startup")
async def startup_event():
    """Hàm được gọi khi ứng dụng khởi động"""
    logger.info("Ứng dụng AI Job Matcher đang khởi động...")
    logger.info(f"API documentation: {'Bật' if ENABLE_DOCS else 'Tắt'}")

    # Đảm bảo các thư mục tồn tại
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(settings.TEMP_DIR, "crawler_cache"), exist_ok=True)
    os.makedirs(os.path.join(settings.TEMP_DIR, "matcher_cache"), exist_ok=True)

    # Cấu hình GPU nếu có
    configure_gpu()

    # Tải các mô hình trong background để không chặn khởi động ứng dụng
    asyncio.create_task(load_models_async())

    logger.info("Ứng dụng AI Job Matcher đã sẵn sàng phục vụ!")

@app.on_event("shutdown")
async def shutdown_event():
    """Hàm được gọi khi ứng dụng tắt"""
    logger.info("Ứng dụng AI Job Matcher đang tắt...")

    # Giải phóng tài nguyên GPU nếu có
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Đã giải phóng bộ nhớ GPU")
    except:
        pass

    # Dọn dẹp tệp tạm nếu cần
    try:
        logger.info("Đang dọn dẹp tệp tạm...")
        # Xóa các tệp cache quá hạn nếu cần
    except:
        pass

    logger.info("Ứng dụng AI Job Matcher đã tắt an toàn")

@app.get("/", tags=["Root"])
async def root():
    """Route gốc"""
    docs_url = "/api/docs" if ENABLE_DOCS else None

    response = {
        "message": "Chào mừng đến với AI Job Matcher API",
        "version": settings.VERSION,
        "status": "Sẵn sàng" if models_loaded else "Đang khởi động",
    }

    if docs_url:
        response["docs"] = docs_url

    return response

async def load_models_async():
    """
    Tải các mô hình cần thiết trong background để không chặn startup
    """
    global models_loaded

    try:
        logger.info("Đang tải các mô hình cần thiết...")

        # Kiểm tra không gian đĩa
        if not check_disk_space(1.0):  # Cần ít nhất 1GB
            logger.error("Không đủ không gian đĩa để tải mô hình!")
            return

        # Danh sách các mô hình cần tải
        models_to_load = [
            {"type": "sentence_transformer", "priority": "high"},
            {"type": "spacy", "priority": "high"},
        ]

        # Tải các mô hình ưu tiên cao trước
        high_priority_models = [m for m in models_to_load if m["priority"] == "high"]
        for model in high_priority_models:
            await load_model(model["type"])

        # Đánh dấu là đã tải xong
        models_loaded = True
        logger.info("Hoàn tất tải mô hình!")

        # Tải các mô hình ưu tiên thấp trong background
        low_priority_models = [m for m in models_to_load if m["priority"] != "high"]
        for model in low_priority_models:
            try:
                await load_model(model["type"])
            except Exception as e:
                logger.warning(f"Không thể tải mô hình ưu tiên thấp {model['type']}: {e}")

    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình: {e}")
        # Đánh dấu là đã tải nếu có ít nhất những mô hình cần thiết
        models_loaded = True

async def load_model(model_type: str) -> bool:
    """
    Tải một mô hình cụ thể

    Args:
        model_type: Loại mô hình cần tải

    Returns:
        bool: True nếu tải thành công
    """
    try:
        logger.info(f"Đang tải mô hình {model_type}...")

        # Lấy cấu hình mô hình
        model_config = get_model_config().get(model_type)
        if not model_config:
            logger.error(f"Không tìm thấy cấu hình cho mô hình {model_type}")
            return False

        # Tùy theo loại mô hình
        if model_type == "sentence_transformer":
            # Tải mô hình sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                # Cho phép asyncio nhường quyền để không chặn event loop
                await asyncio.sleep(0)
                # Tải mô hình
                model_name = model_config.get("model_name")
                _ = SentenceTransformer(model_name)
                logger.info(f"Đã tải mô hình Sentence Transformer: {model_name}")
                return True
            except Exception as e:
                logger.error(f"Lỗi khi tải mô hình Sentence Transformer: {str(e)}")
                # Thử tải mô hình dự phòng
                fallback_path = get_fallback_model_path(model_type)
                if fallback_path:
                    try:
                        _ = SentenceTransformer(fallback_path)
                        logger.info(f"Đã tải mô hình Sentence Transformer dự phòng: {fallback_path}")
                        return True
                    except:
                        pass
                return False

        elif model_type == "spacy":
            # Tải mô hình spaCy
            try:
                import spacy
                # Cho phép asyncio nhường quyền để không chặn event loop
                await asyncio.sleep(0)
                # Tải mô hình
                model_name = model_config.get("model_name")
                _ = spacy.load(model_name)
                logger.info(f"Đã tải mô hình spaCy: {model_name}")
                return True
            except Exception as e:
                logger.error(f"Lỗi khi tải mô hình spaCy: {str(e)}")
                # Thử tải mô hình dự phòng
                fallback_model = model_config.get("fallback_model_name")
                if fallback_model:
                    try:
                        _ = spacy.load(fallback_model)
                        logger.info(f"Đã tải mô hình spaCy dự phòng: {fallback_model}")
                        return True
                    except:
                        pass
                return False

        logger.warning(f"Không hỗ trợ tải mô hình loại {model_type}")
        return False

    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình {model_type}: {str(e)}")
        return False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)