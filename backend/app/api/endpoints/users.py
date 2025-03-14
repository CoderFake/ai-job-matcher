"""
API endpoints cho quản lý người dùng
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr

from app.core.logging import get_logger
from app.models.user import UserData, UserPreference, SavedSearch, SavedJob

logger = get_logger("api")
router = APIRouter()


class UserCreateRequest(BaseModel):
    """Yêu cầu tạo người dùng mới"""
    email: EmailStr = Field(..., description="Email người dùng")
    name: str = Field(..., description="Tên người dùng")
    preferences: Optional[UserPreference] = Field(None, description="Tùy chọn người dùng")


class UserUpdateRequest(BaseModel):
    """Yêu cầu cập nhật thông tin người dùng"""
    name: Optional[str] = Field(None, description="Tên người dùng")
    preferences: Optional[UserPreference] = Field(None, description="Tùy chọn người dùng")


class SavedSearchRequest(BaseModel):
    """Yêu cầu lưu tìm kiếm"""
    name: str = Field(..., description="Tên tìm kiếm")
    query: Dict[str, Any] = Field(..., description="Truy vấn tìm kiếm")
    search_id: Optional[str] = Field(None, description="ID của kết quả tìm kiếm")
    notification_enabled: bool = Field(False, description="Bật thông báo khi có kết quả mới")


class SavedJobRequest(BaseModel):
    """Yêu cầu lưu công việc"""
    job_id: str = Field(..., description="ID công việc")
    notes: Optional[str] = Field(None, description="Ghi chú")
    status: str = Field("saved", description="Trạng thái (đã lưu, đã ứng tuyển, đã phỏng vấn, v.v.)")


@router.post("/", response_model=UserData, summary="Tạo người dùng mới")
async def create_user(
        request: UserCreateRequest
):
    """
    Tạo người dùng mới.

    - **email**: Email người dùng
    - **name**: Tên người dùng
    - **preferences**: Tùy chọn người dùng (tùy chọn)

    Trả về thông tin người dùng đã tạo.
    """
    try:
        # Kiểm tra xem người dùng đã tồn tại chưa
        if not hasattr(router, "users"):
            router.users = {}

        for user_id, user in router.users.items():
            if user.email == request.email:
                raise HTTPException(
                    status_code=400,
                    detail=f"Email {request.email} đã được sử dụng"
                )

        # Tạo ID người dùng
        user_id = f"user_{int(time.time())}"

        # Tạo đối tượng UserData
        user = UserData(
            id=user_id,
            email=request.email,
            name=request.name,
            preferences=request.preferences or UserPreference()
        )

        # Lưu người dùng
        router.users[user_id] = user

        return user

    except Exception as e:
        logger.error(f"Lỗi khi tạo người dùng: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi tạo người dùng: {str(e)}"
        )


@router.get("/{user_id}", response_model=UserData, summary="Lấy thông tin người dùng")
async def get_user(
        user_id: str
):
    """
    Lấy thông tin người dùng.

    - **user_id**: ID người dùng

    Trả về thông tin người dùng.
    """
    try:
        # Kiểm tra xem người dùng có tồn tại không
        if not hasattr(router, "users") or user_id not in router.users:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy người dùng với ID: {user_id}"
            )

        # Lấy thông tin người dùng
        user = router.users[user_id]

        return user

    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin người dùng: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lấy thông tin người dùng: {str(e)}"
        )


@router.put("/{user_id}", response_model=UserData, summary="Cập nhật thông tin người dùng")
async def update_user(
        user_id: str,
        request: UserUpdateRequest
):
    """
    Cập nhật thông tin người dùng.

    - **user_id**: ID người dùng
    - **request**: Thông tin cần cập nhật

    Trả về thông tin người dùng đã cập nhật.
    """
    try:
        # Kiểm tra xem người dùng có tồn tại không
        if not hasattr(router, "users") or user_id not in router.users:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy người dùng với ID: {user_id}"
            )

        # Lấy thông tin người dùng
        user = router.users[user_id]

        # Cập nhật thông tin
        if request.name:
            user.name = request.name

        if request.preferences:
            # Cập nhật từng trường trong preferences
            for field, value in request.preferences.model_dump(exclude_unset=True).items():
                setattr(user.preferences, field, value)

        # Cập nhật thời gian cập nhật
        import datetime
        user.updated_at = datetime.datetime.now()

        return user

    except Exception as e:
        logger.error(f"Lỗi khi cập nhật thông tin người dùng: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi cập nhật thông tin người dùng: {str(e)}"
        )


@router.delete("/{user_id}", response_model=Dict[str, Any], summary="Xóa người dùng")
async def delete_user(
        user_id: str
):
    """
    Xóa người dùng.

    - **user_id**: ID người dùng

    Trả về thông tin xác nhận xóa.
    """
    try:
        # Kiểm tra xem người dùng có tồn tại không
        if not hasattr(router, "users") or user_id not in router.users:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy người dùng với ID: {user_id}"
            )

        # Lấy thông tin người dùng
        user = router.users[user_id]

        # Xóa người dùng
        del router.users[user_id]

        return {
            "user_id": user_id,
            "email": user.email,
            "status": "success",
            "message": "Người dùng đã được xóa thành công"
        }

    except Exception as e:
        logger.error(f"Lỗi khi xóa người dùng: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xóa người dùng: {str(e)}"
        )


@router.post("/{user_id}/cv/{cv_id}", response_model=UserData, summary="Liên kết CV với người dùng")
async def link_cv_to_user(
        user_id: str,
        cv_id: str
):
    """
    Liên kết CV với người dùng.

    - **user_id**: ID người dùng
    - **cv_id**: ID của CV

    Trả về thông tin người dùng đã cập nhật.
    """
    try:
        # Kiểm tra xem người dùng có tồn tại không
        if not hasattr(router, "users") or user_id not in router.users:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy người dùng với ID: {user_id}"
            )

        # Kiểm tra xem CV có tồn tại không
        if not hasattr(router, "uploaded_files") or cv_id not in router.uploaded_files:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy CV với ID: {cv_id}"
            )

        # Lấy thông tin người dùng
        user = router.users[user_id]

        # Thêm CV vào danh sách CV của người dùng nếu chưa có
        if cv_id not in user.cv_ids:
            user.cv_ids.append(cv_id)

        return user

    except Exception as e:
        logger.error(f"Lỗi khi liên kết CV với người dùng: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi liên kết CV với người dùng: {str(e)}"
        )


@router.post("/{user_id}/saved-search", response_model=SavedSearch, summary="Lưu tìm kiếm")
async def save_search(
        user_id: str,
        request: SavedSearchRequest
):
    """
    Lưu tìm kiếm.

    - **user_id**: ID người dùng
    - **request**: Thông tin tìm kiếm cần lưu

    Trả về thông tin tìm kiếm đã lưu.
    """
    try:
        # Kiểm tra xem người dùng có tồn tại không
        if not hasattr(router, "users") or user_id not in router.users:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy người dùng với ID: {user_id}"
            )

        # Kiểm tra xem kết quả tìm kiếm có tồn tại không (nếu có)
        if request.search_id and (
                not hasattr(router, "search_results") or request.search_id not in router.search_results):
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy kết quả tìm kiếm với ID: {request.search_id}"
            )

        # Lấy thông tin người dùng
        user = router.users[user_id]

        # Tạo ID cho tìm kiếm đã lưu
        search_id = f"saved_search_{int(time.time())}"

        # Tạo đối tượng SavedSearch
        import datetime
        saved_search = SavedSearch(
            id=search_id,
            name=request.name,
            query=request.query,
            notification_enabled=request.notification_enabled,
            last_executed=datetime.datetime.now() if request.search_id else None
        )

        # Thêm vào danh sách tìm kiếm đã lưu của người dùng
        user.saved_searches.append(saved_search)

        return saved_search

    except Exception as e:
        logger.error(f"Lỗi khi lưu tìm kiếm: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lưu tìm kiếm: {str(e)}"
        )


@router.get("/{user_id}/saved-searches", response_model=List[SavedSearch], summary="Lấy danh sách tìm kiếm đã lưu")
async def get_saved_searches(
        user_id: str
):
    """
    Lấy danh sách tìm kiếm đã lưu của người dùng.

    - **user_id**: ID người dùng

    Trả về danh sách tìm kiếm đã lưu.
    """
    try:
        # Kiểm tra xem người dùng có tồn tại không
        if not hasattr(router, "users") or user_id not in router.users:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy người dùng với ID: {user_id}"
            )

        # Lấy thông tin người dùng
        user = router.users[user_id]

        return user.saved_searches

    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách tìm kiếm đã lưu: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lấy danh sách tìm kiếm đã lưu: {str(e)}"
        )


@router.delete("/{user_id}/saved-search/{search_id}", response_model=Dict[str, Any], summary="Xóa tìm kiếm đã lưu")
async def delete_saved_search(
        user_id: str,
        search_id: str
):
    """
    Xóa tìm kiếm đã lưu.

    - **user_id**: ID người dùng
    - **search_id**: ID của tìm kiếm đã lưu

    Trả về thông tin xác nhận xóa.
    """
    try:
        # Kiểm tra xem người dùng có tồn tại không
        if not hasattr(router, "users") or user_id not in router.users:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy người dùng với ID: {user_id}"
            )

        # Lấy thông tin người dùng
        user = router.users[user_id]

        # Tìm và xóa tìm kiếm đã lưu
        for i, saved_search in enumerate(user.saved_searches):
            if saved_search.id == search_id:
                deleted_search = user.saved_searches.pop(i)

                return {
                    "search_id": search_id,
                    "name": deleted_search.name,
                    "status": "success",
                    "message": "Tìm kiếm đã lưu đã được xóa thành công"
                }

        # Không tìm thấy tìm kiếm đã lưu
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy tìm kiếm đã lưu với ID: {search_id}"
        )

    except Exception as e:
        logger.error(f"Lỗi khi xóa tìm kiếm đã lưu: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xóa tìm kiếm đã lưu: {str(e)}"
        )


@router.post("/{user_id}/saved-job", response_model=SavedJob, summary="Lưu công việc")
async def save_job(
        user_id: str,
        request: SavedJobRequest
):
    """
    Lưu công việc.

    - **user_id**: ID người dùng
    - **request**: Thông tin công việc cần lưu

    Trả về thông tin công việc đã lưu.
    """
    try:
        # Kiểm tra xem người dùng có tồn tại không
        if not hasattr(router, "users") or user_id not in router.users:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy người dùng với ID: {user_id}"
            )

        # Tìm kiếm công việc
        job = None
        if hasattr(router, "search_results"):
            for search_id, jobs in router.search_results.items():
                for search_job in jobs:
                    if search_job.id == request.job_id:
                        job = search_job
                        break
                if job:
                    break

        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy công việc với ID: {request.job_id}"
            )

        # Lấy thông tin người dùng
        user = router.users[user_id]

        # Tạo đối tượng SavedJob
        saved_job = SavedJob(
            job_id=request.job_id,
            notes=request.notes,
            status=request.status
        )

        # Thêm vào danh sách công việc đã lưu của người dùng
        user.saved_jobs.append(saved_job)

        return saved_job

    except Exception as e:
        logger.error(f"Lỗi khi lưu công việc: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lưu công việc: {str(e)}"
        )


@router.get("/{user_id}/saved-jobs", response_model=List[Dict[str, Any]], summary="Lấy danh sách công việc đã lưu")
async def get_saved_jobs(
        user_id: str
):
    """
    Lấy danh sách công việc đã lưu của người dùng.

    - **user_id**: ID người dùng

    Trả về danh sách công việc đã lưu.
    """
    try:
        # Kiểm tra xem người dùng có tồn tại không
        if not hasattr(router, "users") or user_id not in router.users:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy người dùng với ID: {user_id}"
            )

        # Lấy thông tin người dùng
        user = router.users[user_id]

        # Chuẩn bị kết quả
        result = []

        for saved_job in user.saved_jobs:
            # Tìm thông tin công việc
            job = None
            if hasattr(router, "search_results"):
                for search_id, jobs in router.search_results.items():
                    for search_job in jobs:
                        if search_job.id == saved_job.job_id:
                            job = search_job
                            break
                    if job:
                        break

            job_info = {
                "saved_job": saved_job,
                "job": job
            }

            result.append(job_info)

        return result

    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách công việc đã lưu: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lấy danh sách công việc đã lưu: {str(e)}"
        )


@router.delete("/{user_id}/saved-job/{job_id}", response_model=Dict[str, Any], summary="Xóa công việc đã lưu")
async def delete_saved_job(
        user_id: str,
        job_id: str
):
    """
    Xóa công việc đã lưu.

    - **user_id**: ID người dùng
    - **job_id**: ID của công việc đã lưu

    Trả về thông tin xác nhận xóa.
    """
    try:
        # Kiểm tra xem người dùng có tồn tại không
        if not hasattr(router, "users") or user_id not in router.users:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy người dùng với ID: {user_id}"
            )

        # Lấy thông tin người dùng
        user = router.users[user_id]

        # Tìm và xóa công việc đã lưu
        for i, saved_job in enumerate(user.saved_jobs):
            if saved_job.job_id == job_id:
                deleted_job = user.saved_jobs.pop(i)

                return {
                    "job_id": job_id,
                    "status": "success",
                    "message": "Công việc đã lưu đã được xóa thành công"
                }

        # Không tìm thấy công việc đã lưu
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy công việc đã lưu với ID: {job_id}"
        )

    except Exception as e:
        logger.error(f"Lỗi khi xóa công việc đã lưu: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xóa công việc đã lưu: {str(e)}"
        )


@router.post("/{user_id}/recent-search", response_model=Dict[str, Any], summary="Thêm tìm kiếm gần đây")
async def add_recent_search(
        user_id: str,
        search_id: str
):
    """
    Thêm tìm kiếm vào danh sách tìm kiếm gần đây của người dùng.

    - **user_id**: ID người dùng
    - **search_id**: ID của kết quả tìm kiếm

    Trả về thông tin xác nhận.
    """
    try:
        # Kiểm tra xem người dùng có tồn tại không
        if not hasattr(router, "users") or user_id not in router.users:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy người dùng với ID: {user_id}"
            )

        # Kiểm tra xem kết quả tìm kiếm có tồn tại không
        if not hasattr(router, "search_results") or search_id not in router.search_results:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy kết quả tìm kiếm với ID: {search_id}"
            )

        # Lấy thông tin người dùng
        user = router.users[user_id]

        # Tạo thông tin tìm kiếm gần đây
        recent_search = {
            "search_id": search_id,
            "timestamp": time.time(),
            "result_count": len(router.search_results[search_id])
        }

        # Xóa tìm kiếm trùng lặp nếu có
        user.recent_searches = [s for s in user.recent_searches if s.get("search_id") != search_id]

        # Thêm vào đầu danh sách
        user.recent_searches.insert(0, recent_search)

        # Giữ tối đa 10 tìm kiếm gần đây
        if len(user.recent_searches) > 10:
            user.recent_searches = user.recent_searches[:10]

        return {
            "user_id": user_id,
            "search_id": search_id,
            "status": "success",
            "message": "Đã thêm tìm kiếm vào danh sách tìm kiếm gần đây"
        }

    except Exception as e:
        logger.error(f"Lỗi khi thêm tìm kiếm gần đây: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi thêm tìm kiếm gần đây: {str(e)}"
        )


@router.get("/{user_id}/recent-searches", response_model=List[Dict[str, Any]], summary="Lấy danh sách tìm kiếm gần đây")
async def get_recent_searches(
        user_id: str
):
    """
    Lấy danh sách tìm kiếm gần đây của người dùng.

    - **user_id**: ID người dùng

    Trả về danh sách tìm kiếm gần đây.
    """
    try:
        # Kiểm tra xem người dùng có tồn tại không
        if not hasattr(router, "users") or user_id not in router.users:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy người dùng với ID: {user_id}"
            )

        # Lấy thông tin người dùng
        user = router.users[user_id]

        return user.recent_searches

    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách tìm kiếm gần đây: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lấy danh sách tìm kiếm gần đây: {str(e)}"
        )


@router.get("/{user_id}/preferences", response_model=UserPreference, summary="Lấy tùy chọn người dùng")
async def get_user_preferences(
        user_id: str
):
    """
    Lấy tùy chọn của người dùng.

    - **user_id**: ID người dùng

    Trả về tùy chọn của người dùng.
    """
    try:
        # Kiểm tra xem người dùng có tồn tại không
        if not hasattr(router, "users") or user_id not in router.users:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy người dùng với ID: {user_id}"
            )

        # Lấy thông tin người dùng
        user = router.users[user_id]

        return user.preferences

    except Exception as e:
        logger.error(f"Lỗi khi lấy tùy chọn người dùng: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lấy tùy chọn người dùng: {str(e)}"
        )


@router.put("/{user_id}/preferences", response_model=UserPreference, summary="Cập nhật tùy chọn người dùng")
async def update_user_preferences(
        user_id: str,
        preferences: UserPreference
):
    """
    Cập nhật tùy chọn của người dùng.

    - **user_id**: ID người dùng
    - **preferences**: Tùy chọn mới

    Trả về tùy chọn của người dùng đã cập nhật.
    """
    try:
        # Kiểm tra xem người dùng có tồn tại không
        if not hasattr(router, "users") or user_id not in router.users:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy người dùng với ID: {user_id}"
            )

        # Lấy thông tin người dùng
        user = router.users[user_id]

        # Cập nhật tùy chọn
        user.preferences = preferences

        # Cập nhật thời gian cập nhật
        import datetime
        user.updated_at = datetime.datetime.now()

        return user.preferences

    except Exception as e:
        logger.error(f"Lỗi khi cập nhật tùy chọn người dùng: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi cập nhật tùy chọn người dùng: {str(e)}"
        )