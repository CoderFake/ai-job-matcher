import os
import time
import tempfile
import logging
import shutil
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr

from app.core.logging import get_logger
from app.models.cv import CVData, PersonalInfo
from app.services.cv_parser.extract import CVExtractor
from app.services.llm import LocalLLM

logger = get_logger("api")
router = APIRouter()

# Khởi tạo bộ trích xuất CV
cv_extractor = CVExtractor()

# Kích thước file tối đa (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Danh sách các định dạng tệp hỗ trợ
SUPPORTED_EXTENSIONS = {
    # PDF
    ".pdf": "application/pdf",
    # Word
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".rtf": "application/rtf",
    # Excel
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv": "text/csv",
    # Hình ảnh
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".webp": "image/webp"
}


class CVUploadResponse(BaseModel):
    """Phản hồi khi tải lên CV"""
    cv_id: str = Field(..., description="ID của CV đã tải lên")
    file_name: str = Field(..., description="Tên tệp gốc")
    content_type: str = Field(..., description="Loại nội dung")
    status: str = Field(..., description="Trạng thái xử lý")
    message: str = Field(..., description="Thông báo")


class CVAnalysisRequest(BaseModel):
    """Yêu cầu phân tích CV"""
    cv_id: str = Field(..., description="ID của CV đã tải lên")
    use_llm: bool = Field(True, description="Sử dụng LLM để phân tích")


class CVAnalysisResponse(BaseModel):
    """Phản hồi khi phân tích CV"""
    cv_id: str = Field(..., description="ID của CV đã tải lên")
    status: str = Field(..., description="Trạng thái xử lý")
    data: Optional[CVData] = Field(None, description="Dữ liệu CV đã phân tích")
    message: str = Field(..., description="Thông báo")
    processing_time: float = Field(..., description="Thời gian xử lý (giây)")


class CVExtractTextRequest(BaseModel):
    """Yêu cầu trích xuất văn bản từ CV"""
    cv_id: str = Field(..., description="ID của CV đã tải lên")


class CVExtractTextResponse(BaseModel):
    """Phản hồi khi trích xuất văn bản từ CV"""
    cv_id: str = Field(..., description="ID của CV đã tải lên")
    text: str = Field(..., description="Văn bản đã trích xuất")
    content_type: str = Field(..., description="Loại nội dung")
    file_name: str = Field(..., description="Tên tệp gốc")


# Thêm cải tiến vào phương thức upload_cv
@router.post("/upload", response_model=CVUploadResponse, summary="Tải lên CV")
async def upload_cv(
        file: UploadFile = File(..., description="Tệp CV (PDF, Word, Excel, hình ảnh)"),
        background_tasks: BackgroundTasks = None
):
    """
    Tải lên tệp CV để phân tích.

    - **file**: Tệp CV (hỗ trợ các định dạng PDF, Word, Excel và hình ảnh)

    Trả về thông tin về tệp đã tải lên và ID để sử dụng trong các API khác.
    """
    try:
        # Kiểm tra kích thước tệp với cơ chế tối ưu
        file_size = 0
        temp_file_path = None

        # Tạo thư mục tạm thời để lưu tệp nếu chưa có
        upload_dir = Path(tempfile.gettempdir()) / "ai_job_matcher_uploads"
        upload_dir.mkdir(exist_ok=True)

        # Tạo tệp tạm với context manager để đảm bảo được xóa nếu có lỗi
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                temp_file_path = tmp.name
                chunk_size = 1024 * 1024  # 1MB chunks
                chunk = await file.read(chunk_size)

                while chunk:
                    size = len(chunk)
                    file_size += size

                    if file_size > MAX_FILE_SIZE:
                        # Đóng và xóa tệp nếu vượt quá giới hạn
                        tmp.close()
                        os.unlink(temp_file_path)
                        raise HTTPException(
                            status_code=413,
                            detail=f"Kích thước tệp quá lớn. Giới hạn {MAX_FILE_SIZE / (1024 * 1024):.1f}MB."
                        )

                    tmp.write(chunk)
                    chunk = await file.read(chunk_size)
        except Exception as e:
            # Xóa tệp tạm nếu có lỗi
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e

        # Kiểm tra định dạng tệp và tính xác thực
        file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
        if file_ext not in SUPPORTED_EXTENSIONS:
            # Xóa tệp tạm nếu định dạng không được hỗ trợ
            if temp_file_path:
                os.unlink(temp_file_path)

            raise HTTPException(
                status_code=400,
                detail=f"Không hỗ trợ loại tệp {file_ext}. Hỗ trợ các định dạng: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
            )

        # Kiểm tra nội dung tệp có phù hợp với phần mở rộng không (chống giả mạo)
        if not is_valid_file_content(temp_file_path, file_ext):
            os.unlink(temp_file_path)
            raise HTTPException(
                status_code=400,
                detail=f"Nội dung tệp không phù hợp với định dạng {file_ext}"
            )

        # Xử lý tên tệp để tránh các ký tự không hợp lệ
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._- ").strip()
        if not safe_filename:
            safe_filename = f"uploaded_file{file_ext}"

        # Tạo ID cho CV dựa trên timestamp và UUID
        cv_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"

        # Tạo đường dẫn tệp cuối cùng
        file_path = upload_dir / f"{cv_id}_{safe_filename}"

        # Di chuyển tệp từ thư mục tạm thời
        shutil.move(temp_file_path, file_path)

        # Ghi log
        logger.info(f"Đã tải lên CV: {file.filename} -> {file_path} ({file_size} bytes)")

        # Lưu thông tin tệp để sử dụng sau này
        if not hasattr(router, "uploaded_files"):
            router.uploaded_files = {}

        router.uploaded_files[cv_id] = {
            "file_path": str(file_path),
            "file_name": file.filename,
            "content_type": file.content_type or SUPPORTED_EXTENSIONS.get(file_ext, "application/octet-stream"),
            "upload_time": time.time(),
            "file_size": file_size
        }

        # Thêm nhiệm vụ xóa tệp cũ vào background (tệp đã quá 24 giờ)
        if background_tasks:
            background_tasks.add_task(cleanup_old_files, upload_dir)

        return CVUploadResponse(
            cv_id=cv_id,
            file_name=file.filename,
            content_type=file.content_type or SUPPORTED_EXTENSIONS.get(file_ext, "application/octet-stream"),
            status="success",
            message="Tệp đã được tải lên thành công"
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Lỗi khi tải lên CV: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi tải lên CV: {str(e)}"
        )


# Thêm hàm kiểm tra nội dung tệp
def is_valid_file_content(file_path: str, extension: str) -> bool:
    """
    Kiểm tra xem nội dung tệp có phù hợp với phần mở rộng không

    Args:
        file_path: Đường dẫn tệp
        extension: Phần mở rộng của tệp

    Returns:
        bool: True nếu nội dung hợp lệ
    """
    try:
        # Đọc vài byte đầu tiên của tệp
        with open(file_path, "rb") as f:
            header = f.read(8)  # 8 byte đầu tiên

        # Kiểm tra signature của tệp theo từng loại
        if extension == ".pdf" and header.startswith(b"%PDF"):
            return True
        elif extension in [".doc", ".docx"] and header.startswith(b"\xD0\xCF\x11\xE0") or header.startswith(
                b"PK\x03\x04"):
            return True
        elif extension in [".xls", ".xlsx"] and header.startswith(b"\xD0\xCF\x11\xE0") or header.startswith(
                b"PK\x03\x04"):
            return True
        elif extension in [".jpg", ".jpeg"] and header.startswith(b"\xFF\xD8\xFF"):
            return True
        elif extension == ".png" and header.startswith(b"\x89PNG\r\n\x1A\n"):
            return True
        elif extension == ".bmp" and header.startswith(b"BM"):
            return True
        elif extension == ".tif" or extension == ".tiff":
            return header.startswith(b"II*\x00") or header.startswith(b"MM\x00*")
        elif extension == ".webp" and header[4:8] == b"WEBP":
            return True

        # Mặc định trả về True cho các loại không kiểm tra được
        return True
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra nội dung tệp {file_path}: {str(e)}")
        return False


def cleanup_old_files(directory: Path, max_age_hours: int = 24):
    """
    Xóa các tệp tạm thời cũ

    Args:
        directory: Thư mục chứa tệp
        max_age_hours: Thời gian tối đa (giờ) để giữ tệp
    """
    try:
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        removed_count = 0
        total_size_freed = 0

        # Kiểm tra từng tệp trong thư mục
        for file_path in directory.glob("*"):
            if file_path.is_file():
                try:
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        # Lưu kích thước tệp trước khi xóa
                        file_size = os.path.getsize(file_path)
                        os.unlink(file_path)
                        removed_count += 1
                        total_size_freed += file_size
                        logger.info(f"Đã xóa tệp cũ: {file_path} (kích thước: {file_size / 1024:.2f} KB)")
                except Exception as e:
                    logger.warning(f"Không thể xóa tệp {file_path}: {str(e)}")

        if removed_count > 0:
            logger.info(f"Đã dọn dẹp {removed_count} tệp cũ, giải phóng {total_size_freed / (1024 * 1024):.2f} MB")
    except Exception as e:
        logger.error(f"Lỗi khi dọn dẹp tệp cũ: {str(e)}")


@router.post("/analyze", response_model=CVAnalysisResponse, summary="Phân tích CV")
async def analyze_cv(
        request: CVAnalysisRequest,
        background_tasks: BackgroundTasks = None
):
    """
    Phân tích CV đã tải lên.

    - **cv_id**: ID của CV đã tải lên
    - **use_llm**: Sử dụng LLM để phân tích (mặc định là True)

    Trả về dữ liệu CV đã phân tích.
    """
    try:
        # Kiểm tra xem CV có tồn tại không
        if not hasattr(router, "uploaded_files") or request.cv_id not in router.uploaded_files:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy CV với ID: {request.cv_id}"
            )

        # Lấy thông tin tệp
        file_info = router.uploaded_files[request.cv_id]
        file_path = file_info["file_path"]

        # Bắt đầu tính thời gian xử lý
        start_time = time.time()

        # Kiểm tra tệp
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy tệp CV: {file_path}"
            )

        # Phân tích CV
        cv_data = cv_extractor.process_cv_file(file_path, use_llm=request.use_llm)

        # Nâng cao dữ liệu CV
        cv_data = cv_extractor.enhance_cv_data(cv_data)

        # Lưu kết quả phân tích vào bộ nhớ
        if not hasattr(router, "analyzed_cvs"):
            router.analyzed_cvs = {}

        router.analyzed_cvs[request.cv_id] = cv_data

        # Tính thời gian xử lý
        processing_time = time.time() - start_time

        # Nếu background_tasks không None, thêm nhiệm vụ xóa các kết quả phân tích cũ
        if background_tasks:
            background_tasks.add_task(cleanup_old_analyses)

        return CVAnalysisResponse(
            cv_id=request.cv_id,
            status="success",
            data=cv_data,
            message="CV đã được phân tích thành công",
            processing_time=processing_time
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Lỗi khi phân tích CV: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi phân tích CV: {str(e)}"
        )


def cleanup_old_analyses(max_count: int = 100):
    """
    Xóa các kết quả phân tích cũ để tránh sử dụng quá nhiều bộ nhớ

    Args:
        max_count: Số lượng kết quả tối đa cần giữ lại
    """
    try:
        if hasattr(router, "analyzed_cvs") and len(router.analyzed_cvs) > max_count:
            # Lấy danh sách CV ID và thời gian tải lên
            cv_times = []
            for cv_id, _ in router.analyzed_cvs.items():
                if cv_id in router.uploaded_files:
                    cv_times.append((cv_id, router.uploaded_files[cv_id]["upload_time"]))
                else:
                    # Nếu không có thông tin tải lên, sử dụng thời gian 0
                    cv_times.append((cv_id, 0))

            # Sắp xếp theo thời gian (cũ nhất trước)
            cv_times.sort(key=lambda x: x[1])

            # Xóa các kết quả cũ
            for cv_id, _ in cv_times[:len(cv_times) - max_count]:
                if cv_id in router.analyzed_cvs:
                    del router.analyzed_cvs[cv_id]
                    logger.info(f"Đã xóa kết quả phân tích cũ cho CV ID: {cv_id}")
    except Exception as e:
        logger.error(f"Lỗi khi dọn dẹp kết quả phân tích cũ: {str(e)}")


@router.post("/extract-text", response_model=CVExtractTextResponse, summary="Trích xuất văn bản từ CV")
async def extract_text_from_cv(
        request: CVExtractTextRequest
):
    """
    Trích xuất văn bản thô từ CV.

    - **cv_id**: ID của CV đã tải lên

    Trả về văn bản đã trích xuất từ CV.
    """
    try:
        # Kiểm tra xem CV có tồn tại không
        if not hasattr(router, "uploaded_files") or request.cv_id not in router.uploaded_files:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy CV với ID: {request.cv_id}"
            )

        # Lấy thông tin tệp
        file_info = router.uploaded_files[request.cv_id]
        file_path = file_info["file_path"]

        # Kiểm tra tệp
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy tệp CV: {file_path}"
            )

        # Trích xuất văn bản
        text, file_type = cv_extractor.extract_text_from_file(file_path)

        # Trả về văn bản đã trích xuất
        return CVExtractTextResponse(
            cv_id=request.cv_id,
            text=text,
            content_type=file_info["content_type"],
            file_name=file_info["file_name"]
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Lỗi khi trích xuất văn bản từ CV: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi trích xuất văn bản từ CV: {str(e)}"
        )


@router.get("/info/{cv_id}", response_model=Dict[str, Any], summary="Lấy thông tin về CV")
async def get_cv_info(
        cv_id: str
):
    """
    Lấy thông tin về CV đã tải lên.

    - **cv_id**: ID của CV đã tải lên

    Trả về thông tin chi tiết về CV.
    """
    try:
        # Kiểm tra xem CV có tồn tại không
        if not hasattr(router, "uploaded_files") or cv_id not in router.uploaded_files:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy CV với ID: {cv_id}"
            )

        # Lấy thông tin tệp
        file_info = router.uploaded_files[cv_id]

        # Kiểm tra tệp
        file_path = file_info["file_path"]
        file_exists = os.path.exists(file_path)
        file_size = os.path.getsize(file_path) if file_exists else 0

        # Lấy thông tin phân tích nếu có
        cv_data = None
        if hasattr(router, "analyzed_cvs") and cv_id in router.analyzed_cvs:
            cv_data = router.analyzed_cvs[cv_id]

        return {
            "cv_id": cv_id,
            "file_name": file_info["file_name"],
            "content_type": file_info["content_type"],
            "upload_time": file_info["upload_time"],
            "is_analyzed": cv_data is not None,
            "file_exists": file_exists,
            "file_size": file_size,
            "file_path": file_path if file_exists else None
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin CV: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lấy thông tin CV: {str(e)}"
        )


@router.get("/list", response_model=List[Dict[str, Any]], summary="Liệt kê các CV đã tải lên")
async def list_cvs():
    """
    Liệt kê tất cả các CV đã tải lên.

    Trả về danh sách các CV đã tải lên.
    """
    try:
        if not hasattr(router, "uploaded_files"):
            return []

        cv_list = []
        for cv_id, file_info in router.uploaded_files.items():
            # Kiểm tra xem tệp có tồn tại không
            file_path = file_info["file_path"]
            file_exists = os.path.exists(file_path)

            # Kiểm tra xem CV đã được phân tích chưa
            is_analyzed = hasattr(router, "analyzed_cvs") and cv_id in router.analyzed_cvs

            cv_list.append({
                "cv_id": cv_id,
                "file_name": file_info["file_name"],
                "content_type": file_info["content_type"],
                "upload_time": file_info["upload_time"],
                "is_analyzed": is_analyzed,
                "file_exists": file_exists
            })

        # Sắp xếp theo thời gian tải lên giảm dần (mới nhất trước)
        cv_list.sort(key=lambda x: x["upload_time"], reverse=True)

        return cv_list

    except Exception as e:
        logger.error(f"Lỗi khi liệt kê CV: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi liệt kê CV: {str(e)}"
        )


@router.delete("/{cv_id}", response_model=Dict[str, Any], summary="Xóa CV")
async def delete_cv(
        cv_id: str
):
    """
    Xóa CV đã tải lên.

    - **cv_id**: ID của CV cần xóa

    Trả về thông tin về CV đã xóa.
    """
    try:
        # Kiểm tra xem CV có tồn tại không
        if not hasattr(router, "uploaded_files") or cv_id not in router.uploaded_files:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy CV với ID: {cv_id}"
            )

        # Lấy thông tin tệp
        file_info = router.uploaded_files[cv_id]
        file_path = file_info["file_path"]

        # Xóa tệp nếu tồn tại
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Đã xóa tệp: {file_path}")

        # Xóa thông tin tệp
        deleted_info = router.uploaded_files.pop(cv_id)

        # Xóa thông tin phân tích nếu có
        if hasattr(router, "analyzed_cvs") and cv_id in router.analyzed_cvs:
            del router.analyzed_cvs[cv_id]
            logger.info(f"Đã xóa kết quả phân tích cho CV ID: {cv_id}")

        return {
            "cv_id": cv_id,
            "file_name": deleted_info["file_name"],
            "status": "success",
            "message": "CV đã được xóa thành công"
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Lỗi khi xóa CV: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xóa CV: {str(e)}"
        )


@router.post("/image-analysis", response_model=Dict[str, Any], summary="Phân tích CV từ hình ảnh")
async def analyze_cv_image(
        file: UploadFile = File(..., description="Hình ảnh CV"),
        use_llm: bool = Form(True, description="Sử dụng LLM để phân tích")
):
    """
    Phân tích CV từ hình ảnh.

    - **file**: Hình ảnh CV (hỗ trợ các định dạng hình ảnh phổ biến)
    - **use_llm**: Sử dụng LLM để phân tích (mặc định là True)

    Trả về thông tin đã phân tích từ hình ảnh CV.
    """
    try:
        # Kiểm tra định dạng hình ảnh
        file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
        supported_image_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]

        if file_ext not in supported_image_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Không hỗ trợ định dạng hình ảnh {file_ext}. Hỗ trợ các định dạng: {', '.join(supported_image_formats)}"
            )

        # Kiểm tra kích thước file
        content = await file.read(1024 * 1024 * 10)  # Đọc tối đa 10MB
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Kích thước hình ảnh quá lớn. Giới hạn {MAX_FILE_SIZE / (1024 * 1024):.1f}MB"
            )

        # Tạo tệp tạm thời
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_path = temp_file.name
            temp_file.write(content)

        try:
            # Bắt đầu tính thời gian xử lý
            start_time = time.time()

            # Sử dụng trình phân tích hình ảnh
            from app.services.cv_parser.image import ImageParser
            image_parser = ImageParser()

            # Cải thiện chất lượng hình ảnh cho OCR
            improved_image_path = image_parser.improve_image_for_ocr(temp_path)

            # Phát hiện loại tài liệu
            document_type = image_parser.detect_document_type(improved_image_path)

            if document_type != "cv":
                return {
                    "status": "warning",
                    "message": "Hình ảnh có thể không phải là CV",
                    "document_type": document_type,
                    "confidence": 0.5,
                    "processing_time": time.time() - start_time
                }

            # Trích xuất thông tin
            result = image_parser.extract_structured_info(improved_image_path)

            # Trích xuất văn bản
            text = image_parser.extract_text(improved_image_path)

            # Đánh giá chất lượng hình ảnh
            quality_info = image_parser.get_image_quality(improved_image_path)

            # Chuyển đổi kết quả thành CVData
            cv_data = cv_extractor._convert_json_to_cv_data(result, file.filename)
            cv_data.raw_text = text
            cv_data.extracted_from_image = True

            # Tính thời gian xử lý
            processing_time = time.time() - start_time

            return {
                "status": "success",
                "message": "CV đã được phân tích thành công",
                "data": cv_data,
                "text": text,
                "quality_info": quality_info,
                "document_type": document_type,
                "processing_time": processing_time
            }

        finally:
            # Xóa tệp tạm thời
            if os.path.exists(temp_path):
                os.unlink(temp_path)

            # Xóa tệp cải thiện nếu khác với tệp gốc
            if improved_image_path != temp_path and os.path.exists(improved_image_path):
                os.unlink(improved_image_path)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Lỗi khi phân tích CV từ hình ảnh: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi phân tích CV từ hình ảnh: {str(e)}"
        )


@router.post("/summarize/{cv_id}", response_model=Dict[str, Any], summary="Tóm tắt CV")
async def summarize_cv(
    cv_id: str,
    max_length: int = Query(250, description="Độ dài tối đa của tóm tắt")
):
    """
    Tạo tóm tắt ngắn gọn về CV.

    - **cv_id**: ID của CV đã tải lên
    - **max_length**: Độ dài tối đa của tóm tắt (mặc định là 250 từ)

    Trả về tóm tắt về CV.
    """
    try:
        # Kiểm tra xem CV có tồn tại không
        if not hasattr(router, "analyzed_cvs") or cv_id not in router.analyzed_cvs:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy dữ liệu phân tích CV với ID: {cv_id}"
            )

        # Lấy dữ liệu CV
        cv_data = router.analyzed_cvs[cv_id]

        # Sử dụng LLM để tạo tóm tắt

        llm = LocalLLM()

        # Tạo prompt
        prompt = f"""
        Tóm tắt CV sau đây một cách ngắn gọn:
        
        Tên: {cv_data.personal_info.name or "Không có"}
        Email: {cv_data.personal_info.email or "Không có"}
        Số điện thoại: {cv_data.personal_info.phone or "Không có"}
        Vị trí: {cv_data.job_title or "Không có"}
        Số năm kinh nghiệm: {cv_data.years_of_experience or "Không có"}
        
        Học vấn:
        {', '.join([f"{edu.institution} - {edu.field_of_study}" for edu in cv_data.education]) if cv_data.education else "Không có"}
        
        Kinh nghiệm làm việc:
        {', '.join([f"{exp.company} - {exp.position}" for exp in cv_data.work_experience]) if cv_data.work_experience else "Không có"}
        
        Kỹ năng:
        {', '.join([skill.name for skill in cv_data.skills]) if cv_data.skills else "Không có"}
        
        Tóm tắt ngắn gọn trong khoảng {max_length} từ, tập trung vào điểm mạnh, kinh nghiệm và kỹ năng nổi bật.
        """

        # Tạo tóm tắt
        summary = llm.generate(prompt, max_tokens=max_length)

        return {
            "cv_id": cv_id,
            "summary": summary,
            "character_count": len(summary),
            "word_count": len(summary.split())
        }

    except Exception as e:
        logger.error(f"Lỗi khi tóm tắt CV: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi tóm tắt CV: {str(e)}"
        )