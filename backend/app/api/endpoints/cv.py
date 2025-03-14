"""
API endpoints cho quản lý và phân tích CV
"""

import os
import time
import tempfile
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.models.cv import CVData, PersonalInfo
from app.services.cv_parser.extract import CVExtractor

logger = get_logger("api")
router = APIRouter()

# Khởi tạo bộ trích xuất CV
cv_extractor = CVExtractor()

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

@router.post("/upload", response_model=CVUploadResponse, summary="Tải lên CV")
async def upload_cv(
    file: UploadFile = File(..., description="Tệp CV (PDF, Word, Excel, hình ảnh)")
):
    """
    Tải lên tệp CV để phân tích.

    - **file**: Tệp CV (hỗ trợ các định dạng PDF, Word, Excel và hình ảnh)

    Trả về thông tin về tệp đã tải lên và ID để sử dụng trong các API khác.
    """
    try:
        # Kiểm tra loại tệp
        file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""

        supported_extensions = [
            # PDF
            ".pdf",
            # Word
            ".doc", ".docx", ".rtf",
            # Excel
            ".xls", ".xlsx", ".csv",
            # Hình ảnh
            ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
        ]

        if file_ext not in supported_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Không hỗ trợ loại tệp {file_ext}. Hỗ trợ các định dạng: {', '.join(supported_extensions)}"
            )

        # Tạo thư mục tạm thời để lưu tệp
        upload_dir = tempfile.mkdtemp()

        # Tạo ID cho CV dựa trên timestamp và tên tệp
        cv_id = f"{int(time.time())}_{os.path.splitext(file.filename)[0]}"
        cv_id = cv_id.replace(" ", "_")

        # Lưu tệp
        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Lưu thông tin tệp vào bộ nhớ hoặc cơ sở dữ liệu
        # Trong thực tế, bạn nên lưu vào cơ sở dữ liệu
        # Ở đây, chúng ta sẽ sử dụng biến toàn cục cho đơn giản
        if not hasattr(router, "uploaded_files"):
            router.uploaded_files = {}

        router.uploaded_files[cv_id] = {
            "file_path": file_path,
            "file_name": file.filename,
            "content_type": file.content_type,
            "upload_time": time.time()
        }

        return CVUploadResponse(
            cv_id=cv_id,
            file_name=file.filename,
            content_type=file.content_type,
            status="success",
            message="Tệp đã được tải lên thành công"
        )

    except Exception as e:
        logger.error(f"Lỗi khi tải lên CV: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi tải lên CV: {str(e)}"
        )

@router.post("/analyze", response_model=CVAnalysisResponse, summary="Phân tích CV")
async def analyze_cv(
    request: CVAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Phân tích CV để trích xuất thông tin.

    - **cv_id**: ID của CV đã tải lên
    - **use_llm**: Sử dụng LLM để phân tích (mặc định là True)

    Trả về thông tin đã phân tích từ CV.
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

        # Phân tích CV
        cv_data = cv_extractor.process_cv_file(file_path, use_llm=request.use_llm)

        # Nâng cao dữ liệu CV
        cv_data = cv_extractor.enhance_cv_data(cv_data)

        # Lưu kết quả phân tích vào bộ nhớ hoặc cơ sở dữ liệu
        if not hasattr(router, "analyzed_cvs"):
            router.analyzed_cvs = {}

        router.analyzed_cvs[request.cv_id] = cv_data

        # Tính thời gian xử lý
        processing_time = time.time() - start_time

        return CVAnalysisResponse(
            cv_id=request.cv_id,
            status="success",
            data=cv_data,
            message="CV đã được phân tích thành công",
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Lỗi khi phân tích CV: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi phân tích CV: {str(e)}"
        )

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

        # Trích xuất văn bản
        text, file_type = cv_extractor.extract_text_from_file(file_path)

        return CVExtractTextResponse(
            cv_id=request.cv_id,
            text=text,
            content_type=file_info["content_type"],
            file_name=file_info["file_name"]
        )

    except Exception as e:
        logger.error(f"Lỗi khi trích xuất văn bản từ CV: {str(e)}")
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
            "file_size": os.path.getsize(file_info["file_path"])
        }

    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin CV: {str(e)}")
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
            is_analyzed = hasattr(router, "analyzed_cvs") and cv_id in router.analyzed_cvs

            cv_list.append({
                "cv_id": cv_id,
                "file_name": file_info["file_name"],
                "content_type": file_info["content_type"],
                "upload_time": file_info["upload_time"],
                "is_analyzed": is_analyzed
            })

        # Sắp xếp theo thời gian tải lên giảm dần (mới nhất trước)
        cv_list.sort(key=lambda x: x["upload_time"], reverse=True)

        return cv_list

    except Exception as e:
        logger.error(f"Lỗi khi liệt kê CV: {str(e)}")
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

        # Xóa tệp
        if os.path.exists(file_path):
            os.remove(file_path)

        # Xóa thông tin tệp
        deleted_info = router.uploaded_files.pop(cv_id)

        # Xóa thông tin phân tích nếu có
        if hasattr(router, "analyzed_cvs") and cv_id in router.analyzed_cvs:
            router.analyzed_cvs.pop(cv_id)

        return {
            "cv_id": cv_id,
            "file_name": deleted_info["file_name"],
            "status": "success",
            "message": "CV đã được xóa thành công"
        }

    except Exception as e:
        logger.error(f"Lỗi khi xóa CV: {str(e)}")
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
        # Kiểm tra loại tệp
        file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""

        supported_extensions = [
            ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
        ]

        if file_ext not in supported_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Không hỗ trợ loại tệp {file_ext}. Hỗ trợ các định dạng: {', '.join(supported_extensions)}"
            )

        # Tạo thư mục tạm thời để lưu tệp
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_path = temp_file.name
            content = await file.read()
            temp_file.write(content)

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

        # Xóa tệp tạm thời
        os.unlink(temp_path)
        if improved_image_path != temp_path:
            os.unlink(improved_image_path)

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

    except Exception as e:
        logger.error(f"Lỗi khi phân tích CV từ hình ảnh: {str(e)}")
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
        from app.services.llm.local_model import LocalLLM
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