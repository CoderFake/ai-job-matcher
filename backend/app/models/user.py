"""
Mô hình dữ liệu cho người dùng
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, EmailStr

class UserPreference(BaseModel):
    """Tùy chọn của người dùng"""
    preferred_job_types: List[str] = Field(default_factory=list, description="Các loại công việc ưa thích")
    preferred_locations: List[str] = Field(default_factory=list, description="Các địa điểm làm việc ưa thích")
    min_salary: Optional[float] = Field(None, description="Mức lương tối thiểu")
    max_commute_distance: Optional[float] = Field(None, description="Khoảng cách đi lại tối đa (km)")
    preferred_benefits: List[str] = Field(default_factory=list, description="Các phúc lợi ưa thích")
    preferred_company_types: List[str] = Field(default_factory=list, description="Các loại công ty ưa thích")
    preferred_industries: List[str] = Field(default_factory=list, description="Các ngành công nghiệp ưa thích")
    remote_preference: bool = Field(False, description="Ưu tiên làm việc từ xa")
    notification_preferences: Dict[str, bool] = Field(default_factory=dict, description="Tùy chọn thông báo")

class SavedSearch(BaseModel):
    """Tìm kiếm đã lưu"""
    id: Optional[str] = Field(None, description="ID tìm kiếm")
    name: str = Field(..., description="Tên tìm kiếm")
    query: Dict[str, Any] = Field(..., description="Truy vấn tìm kiếm")
    created_at: datetime = Field(default_factory=datetime.now, description="Thời gian tạo")
    last_executed: Optional[datetime] = Field(None, description="Thời gian thực thi cuối cùng")
    notification_enabled: bool = Field(False, description="Có bật thông báo không")

class SavedJob(BaseModel):
    """Công việc đã lưu"""
    job_id: str = Field(..., description="ID công việc")
    saved_at: datetime = Field(default_factory=datetime.now, description="Thời gian lưu")
    notes: Optional[str] = Field(None, description="Ghi chú")
    status: str = Field("saved", description="Trạng thái (đã lưu, đã ứng tuyển, đã phỏng vấn, v.v.)")

class UserData(BaseModel):
    """Dữ liệu người dùng"""
    id: Optional[str] = Field(None, description="ID người dùng")
    email: EmailStr = Field(..., description="Email")
    name: str = Field(..., description="Họ và tên")
    created_at: datetime = Field(default_factory=datetime.now, description="Thời gian tạo")
    updated_at: datetime = Field(default_factory=datetime.now, description="Thời gian cập nhật")
    last_login: Optional[datetime] = Field(None, description="Thời gian đăng nhập cuối cùng")
    active: bool = Field(True, description="Người dùng có đang hoạt động không")
    preferences: UserPreference = Field(default_factory=UserPreference, description="Tùy chọn")
    saved_searches: List[SavedSearch] = Field(default_factory=list, description="Các tìm kiếm đã lưu")
    saved_jobs: List[SavedJob] = Field(default_factory=list, description="Các công việc đã lưu")
    cv_ids: List[str] = Field(default_factory=list, description="ID các CV đã tải lên")
    recent_searches: List[Dict[str, Any]] = Field(default_factory=list, description="Các tìm kiếm gần đây")

class SearchHistory(BaseModel):
    """Lịch sử tìm kiếm"""
    user_id: str = Field(..., description="ID người dùng")
    query: Dict[str, Any] = Field(..., description="Truy vấn tìm kiếm")
    timestamp: datetime = Field(default_factory=datetime.now, description="Thời gian tìm kiếm")
    results_count: int = Field(0, description="Số lượng kết quả")
    selected_job_ids: List[str] = Field(default_factory=list, description="ID các công việc đã chọn")