import logging
import json
import os
import sys
import asyncio
import time
import traceback
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

import psutil
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

    # Cải thiện xử lý lỗi và quản lý tài nguyên trong main.py

    # Trình xử lý lỗi chung nâng cao
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        """Xử lý lỗi HTTP một cách chi tiết hơn"""
        # Ghi log với thông tin bổ sung
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        endpoint = request.url.path

        logger.error(
            f"HTTP error: {exc.detail} | "
            f"Status code: {exc.status_code} | "
            f"Client: {client_host} | "
            f"Endpoint: {endpoint} | "
            f"User-Agent: {user_agent[:50]}..."
        )

        # Trả về phản hồi chi tiết hơn
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "type": "http_exception"
                },
                "success": False
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        """Xử lý các loại lỗi không mong đợi"""
        # Ghi log chi tiết với traceback
        exc_traceback = traceback.format_exc()

        # Lấy thông tin client
        client_host = request.client.host if request.client else "unknown"
        endpoint = request.url.path
        method = request.method

        error_id = str(uuid.uuid4())[:8]  # Tạo ID cho lỗi để theo dõi

        logger.critical(
            f"Unhandled error [{error_id}]: {str(exc)} | "
            f"Method: {method} | "
            f"Endpoint: {endpoint} | "
            f"Client: {client_host}\n{exc_traceback}"
        )

        # Lưu thông tin lỗi vào tệp riêng để phân tích sau
        error_log_dir = Path("logs/errors")
        error_log_dir.mkdir(exist_ok=True, parents=True)

        error_log_path = error_log_dir / f"error_{error_id}_{int(time.time())}.log"

        try:
            with open(error_log_path, "w") as f:
                f.write(f"Time: {datetime.now().isoformat()}\n")
                f.write(f"Error ID: {error_id}\n")
                f.write(f"Error Type: {type(exc).__name__}\n")
                f.write(f"Error Message: {str(exc)}\n")
                f.write(f"Endpoint: {endpoint}\n")
                f.write(f"Method: {method}\n")
                f.write(f"Client: {client_host}\n")
                f.write("\nTraceback:\n")
                f.write(exc_traceback)
        except Exception as log_error:
            logger.error(f"Failed to write error log: {str(log_error)}")

        # Phản hồi an toàn cho người dùng
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": 500,
                    "message": "Đã xảy ra lỗi không mong muốn. Vui lòng thử lại sau.",
                    "error_id": error_id,  # Thêm ID để người dùng có thể tham chiếu khi báo lỗi
                    "type": "internal_server_error"
                },
                "success": False
            },
        )

    # Cải thiện middleware ghi log yêu cầu
    @app.middleware("http")
    async def log_and_process_request(request: Request, call_next):
        """
        Middleware ghi log và xử lý yêu cầu với nhiều thông tin hơn và theo dõi hiệu suất
        """
        # Lấy đường dẫn và phương thức
        path = request.url.path
        method = request.method
        client = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        content_type = request.headers.get("content-type", "")
        request_id = str(uuid.uuid4())[:8]  # Tạo ID cho yêu cầu

        # Bỏ qua ghi log cho các tệp tĩnh và yêu cầu health check
        skip_detailed_logging = path.startswith("/static") or path == "/health"

        # Ghi log bắt đầu yêu cầu
        if not skip_detailed_logging:
            logger.info(f"[{request_id}] {client} {method} {path} started")
            if len(user_agent) > 0:
                logger.debug(f"[{request_id}] User-Agent: {user_agent[:100]}")

        # Xử lý yêu cầu và đo thời gian
        start_time = time.time()

        try:
            # Chờ phản hồi
            response = await call_next(request)

            # Tính thời gian xử lý
            process_time = time.time() - start_time

            # Thêm headers hữu ích
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id

            # Ghi log kết thúc yêu cầu
            if not skip_detailed_logging:
                status_code = response.status_code
                logger.info(f"[{request_id}] {client} {method} {path} completed - {status_code} ({process_time:.4f}s)")

                # Ghi log chi tiết nếu là yêu cầu chậm
                if process_time > 1.0:  # Yêu cầu chậm > 1 giây
                    logger.warning(f"Slow request: [{request_id}] {method} {path} - {process_time:.4f}s")

            return response

        except Exception as e:
            # Tính thời gian xử lý
            process_time = time.time() - start_time

            # Ghi log lỗi
            logger.error(f"[{request_id}] {client} {method} {path} failed - Error: {str(e)} ({process_time:.4f}s)")

            # Tiếp tục nâng lỗi
            raise

    # Cải thiện sự kiện startup
    @app.on_event("startup")
    async def startup_event():
        """Hàm được gọi khi ứng dụng khởi động với thông tin chi tiết hơn"""
        logger.info("=" * 40)
        logger.info("Ứng dụng AI Job Matcher đang khởi động...")

        # Ghi log thông tin hệ thống
        logger.info(f"Phiên bản Python: {sys.version}")
        logger.info(f"API docs: {'Bật' if ENABLE_DOCS else 'Tắt'}")
        logger.info(f"Chế độ debug: {'Bật' if os.getenv('DEBUG', 'False').lower() == 'true' else 'Tắt'}")

        # Ghi log thông tin phần cứng
        logger.info(f"CPU: {psutil.cpu_count()} luồng")
        ram_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
        logger.info(f"RAM: {ram_gb:.2f} GB")

        # Kiểm tra GPU
        has_gpu = False
        try:
            import torch
            has_gpu = torch.cuda.is_available()
            if has_gpu:
                gpu_count = torch.cuda.device_count()
                logger.info(f"GPU: {gpu_count} thiết bị")
                for i in range(gpu_count):
                    logger.info(f"- GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                logger.info("GPU: Không có")
        except ImportError:
            logger.info("GPU: Không kiểm tra được (torch không được cài đặt)")

        # Đảm bảo các thư mục tồn tại
        os.makedirs(settings.TEMP_DIR, exist_ok=True)
        os.makedirs(settings.MODEL_DIR, exist_ok=True)
        os.makedirs(os.path.join(settings.TEMP_DIR, "crawler_cache"), exist_ok=True)
        os.makedirs(os.path.join(settings.TEMP_DIR, "matcher_cache"), exist_ok=True)
        os.makedirs("logs/errors", exist_ok=True)

        # Dọn dẹp tệp cache cũ
        await cleanup_old_cache_files()

        # Cấu hình GPU nếu có
        configure_gpu()

        # Khởi tạo các biến toàn cục
        app.state.models_loaded = False
        app.state.startup_time = time.time()
        app.state.request_count = 0
        app.state.error_count = 0

        # Tải các mô hình trong background để không chặn khởi động ứng dụng
        asyncio.create_task(load_models_async())

        logger.info("Ứng dụng AI Job Matcher đã sẵn sàng phục vụ!")
        logger.info("=" * 40)

    async def cleanup_old_cache_files():
        """Dọn dẹp các tệp cache cũ"""
        try:
            # Các thư mục cần dọn dẹp
            cache_dirs = [
                os.path.join(settings.TEMP_DIR, "crawler_cache"),
                os.path.join(settings.TEMP_DIR, "matcher_cache"),
                os.path.join(settings.TEMP_DIR, "web_search_cache")
            ]

            max_age = 7 * 24 * 60 * 60  # 7 ngày
            current_time = time.time()
            total_removed = 0
            total_size_freed = 0

            for cache_dir in cache_dirs:
                if not os.path.exists(cache_dir):
                    continue

                for filename in os.listdir(cache_dir):
                    file_path = os.path.join(cache_dir, filename)
                    if not os.path.isfile(file_path):
                        continue

                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age:
                        file_size = os.path.getsize(file_path)
                        try:
                            os.unlink(file_path)
                            total_removed += 1
                            total_size_freed += file_size
                        except Exception as e:
                            logger.warning(f"Không thể xóa tệp cache {file_path}: {e}")

            if total_removed > 0:
                logger.info(
                    f"Đã dọn dẹp {total_removed} tệp cache cũ, giải phóng {total_size_freed / (1024 * 1024):.2f} MB")

        except Exception as e:
            logger.error(f"Lỗi khi dọn dẹp tệp cache: {e}")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Hàm được gọi khi ứng dụng tắt với thông tin chi tiết hơn"""
        logger.info("=" * 40)
        logger.info("Ứng dụng AI Job Matcher đang tắt...")

        # Hiển thị thống kê
        uptime = time.time() - app.state.startup_time
        uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"

        logger.info(f"Thời gian hoạt động: {uptime_str}")

        if hasattr(app.state, "request_count"):
            logger.info(f"Tổng số yêu cầu đã xử lý: {app.state.request_count}")

        if hasattr(app.state, "error_count"):
            logger.info(f"Tổng số lỗi: {app.state.error_count}")

        # Giải phóng tài nguyên GPU nếu có
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("Đang giải phóng bộ nhớ GPU...")
                torch.cuda.empty_cache()
                logger.info("Đã giải phóng bộ nhớ GPU")
        except Exception as e:
            logger.warning(f"Không thể giải phóng bộ nhớ GPU: {e}")

        # Đóng các kết nối mạng nếu còn mở
        try:
            import aiohttp
            if hasattr(app.state, "http_session") and not app.state.http_session.closed:
                logger.info("Đóng phiên HTTP...")
                await app.state.http_session.close()
        except Exception as e:
            logger.warning(f"Không thể đóng phiên HTTP: {e}")

        # Đồng bộ hóa tệp log cuối cùng
        try:
            for handler in logger.handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()
        except Exception as e:
            pass

        logger.info("Ứng dụng AI Job Matcher đã tắt an toàn")
        logger.info("=" * 40)

    # Cải thiện health check endpoint
    @app.get("/health", tags=["System"])
    async def health_check():
        """
        Kiểm tra sức khỏe hệ thống chi tiết hơn

        Returns:
            Dict: Thông tin sức khỏe hệ thống
        """
        # Kiểm tra các thành phần
        import psutil
        from datetime import datetime

        # Cập nhật số lượng yêu cầu
        if hasattr(app.state, "request_count"):
            app.state.request_count += 1

        # Thông tin cơ bản
        uptime = time.time() - app.state.startup_time

        health_data = {
            "status": "healthy",
            "version": settings.VERSION,
            "timestamp": datetime.now().isoformat(),
            "uptime": uptime,
            "uptime_formatted": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
            "system": {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "process_memory": psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            },
            "services": {
                "models_loaded": app.state.models_loaded
            },
            "stats": {
                "request_count": app.state.request_count if hasattr(app.state, "request_count") else 0,
                "error_count": app.state.error_count if hasattr(app.state, "error_count") else 0
            }
        }

        # Kiểm tra GPU nếu có
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = {
                    "available": True,
                    "count": torch.cuda.device_count(),
                    "name": torch.cuda.get_device_name(0),
                }

                # Thêm thông tin sử dụng GPU nếu có thể
                try:
                    gpu_info["memory_allocated"] = torch.cuda.memory_allocated(0) / (1024 ** 3)  # GB
                    gpu_info["memory_reserved"] = torch.cuda.memory_reserved(0) / (1024 ** 3)  # GB
                    gpu_info["max_memory_allocated"] = torch.cuda.max_memory_allocated(0) / (1024 ** 3)  # GB
                except Exception:
                    pass

                health_data["gpu"] = gpu_info
            else:
                health_data["gpu"] = {"available": False}
        except:
            health_data["gpu"] = {"available": False, "error": "GPU check failed"}

        # Kiểm tra không gian đĩa
        disk_space_sufficient = check_disk_space(1.0)
        health_data["disk_space_sufficient"] = disk_space_sufficient

        # Nếu hệ thống gặp vấn đề nghiêm trọng, đánh dấu là không khỏe mạnh
        if (psutil.virtual_memory().percent > 95 or
                psutil.cpu_percent() > 95 or
                psutil.disk_usage('/').percent > 95 or
                not disk_space_sufficient):
            health_data["status"] = "unhealthy"

            # Chi tiết về vấn đề
            health_data["issues"] = []
            if psutil.virtual_memory().percent > 95:
                health_data["issues"].append("memory_critical")
            if psutil.cpu_percent() > 95:
                health_data["issues"].append("cpu_critical")
            if psutil.disk_usage('/').percent > 95:
                health_data["issues"].append("disk_critical")
            if not disk_space_sufficient:
                health_data["issues"].append("disk_space_insufficient")

        # Trả về với mã trạng thái thích hợp
        if health_data["status"] == "healthy":
            return health_data
        else:
            return JSONResponse(
                status_code=503,  # Service Unavailable
                content=health_data
            )

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