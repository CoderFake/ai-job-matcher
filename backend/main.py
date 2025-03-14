"""
AI Job Matcher - Ứng dụng phân tích CV và tìm kiếm việc làm tự động
"""

import logging
import json
import os
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi

from app.api.router import api_router
from app.core.config import get_model_config
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
        logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
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

        # Ghi log nếu không phải là yêu cầu static files
        if not path.startswith("/static"):
            logger.info(f"{method} {path}")

        # Xử lý yêu cầu
        response = await call_next(request)

        return response

    return application

app = create_application()

@app.on_event("startup")
async def startup_event():
    """Hàm được gọi khi ứng dụng khởi động"""
    logger.info("Ứng dụng AI Job Matcher đang khởi động...")
    logger.info(f"API documentation: {'Bật' if ENABLE_DOCS else 'Tắt'}")

    # Cấu hình GPU nếu có
    from app.core.config import configure_gpu
    configure_gpu()

    # Load mô hình cần thiết
    load_required_models()

@app.on_event("shutdown")
async def shutdown_event():
    """Hàm được gọi khi ứng dụng tắt"""
    logger.info("Ứng dụng AI Job Matcher đang tắt...")
    # Dọn dẹp tài nguyên khi cần thiết

@app.get("/", tags=["Root"])
async def root():
    """Route gốc"""
    docs_url = "/api/docs" if ENABLE_DOCS else None

    response = {
        "message": "Chào mừng đến với AI Job Matcher API",
        "version": settings.VERSION,
    }

    if docs_url:
        response["docs"] = docs_url

    return response

def load_required_models():
    """Load các mô hình cần thiết"""
    try:
        logger.info("Đang tải các mô hình cần thiết...")

        # Tải mô hình sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer

            # Lấy tên mô hình từ cấu hình
            model_config = get_model_config()
            model_name = model_config.get("sentence_transformer", {}).get("model_name", "paraphrase-multilingual-MiniLM-L6-v2")

            # Tải mô hình
            _ = SentenceTransformer(model_name)
            logger.info(f"Đã tải mô hình Sentence Transformer: {model_name}")
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình Sentence Transformer: {str(e)}")

        # Tải mô hình spaCy
        try:
            import spacy

            # Lấy tên mô hình từ cấu hình
            model_config = get_model_config()
            model_name = model_config.get("spacy", {}).get("model_name", "vi_core_news_md")

            # Tải mô hình
            _ = spacy.load(model_name)
            logger.info(f"Đã tải mô hình spaCy: {model_name}")
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình spaCy: {str(e)}")

        logger.info("Tải mô hình hoàn tất!")
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)