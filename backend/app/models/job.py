"""
Mô hình dữ liệu cho công việc
"""

from typing import List, Optional, Dict, Any, Set
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl


class JobType(str, Enum):
    """Loại công việc"""
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    TEMPORARY = "temporary"
    INTERNSHIP = "internship"
    FREELANCE = "freelance"
    OTHER = "other"


class ExperienceLevel(str, Enum):
    """Cấp độ kinh nghiệm"""
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    MANAGER = "manager"
    DIRECTOR = "director"
    EXECUTIVE = "executive"


class JobSource(str, Enum):
    """Nguồn công việc"""
    LINKEDIN = "linkedin"
    TOPCV = "topcv"
    VIETNAMWORKS = "vietnamworks"
    CAREERBUILDER = "careerbuilder"
    CAREERVIET = "careerviet"
    OTHER = "other"


class SalaryRange(BaseModel):
    """Phạm vi lương"""
    min: Optional[float] = Field(None, description="Lương tối thiểu")
    max: Optional[float] = Field(None, description="Lương tối đa")
    currency: Optional[str] = Field("VND", description="Đơn vị tiền tệ")
    is_disclosed: bool = Field(False, description="Lương có được tiết lộ không")
    is_negotiable: bool = Field(False, description="Lương có thể thương lượng không")
    period: Optional[str] = Field("monthly", description="Kỳ lương (hàng tháng, hàng năm)")

    def to_display_string(self) -> str:
        """Chuyển đổi thành chuỗi hiển thị"""
        if not self.is_disclosed:
            return "Thỏa thuận"

        if self.min is None and self.max is None:
            return "Không xác định"

        if self.min is not None and self.max is not None:
            return f"{self.min:,.0f} - {self.max:,.0f} {self.currency}/{self.period}"

        if self.min is not None:
            return f"Từ {self.min:,.0f} {self.currency}/{self.period}"

        return f"Đến {self.max:,.0f} {self.currency}/{self.period}"


class CompanyInfo(BaseModel):
    """Thông tin công ty"""
    name: str = Field(..., description="Tên công ty")
    description: Optional[str] = Field(None, description="Mô tả công ty")
    website: Optional[HttpUrl] = Field(None, description="Trang web công ty")
    industry: Optional[str] = Field(None, description="Ngành công nghiệp")
    company_size: Optional[str] = Field(None, description="Quy mô công ty")
    company_type: Optional[str] = Field(None, description="Loại hình công ty")
    logo_url: Optional[HttpUrl] = Field(None, description="URL logo")
    founded_year: Optional[int] = Field(None, description="Năm thành lập")
    headquarters: Optional[str] = Field(None, description="Trụ sở chính")


class Location(BaseModel):
    """Thông tin vị trí địa lý"""
    address: Optional[str] = Field(None, description="Địa chỉ")
    city: Optional[str] = Field(None, description="Thành phố")
    district: Optional[str] = Field(None, description="Quận/Huyện")
    state: Optional[str] = Field(None, description="Tỉnh/Thành phố")
    country: Optional[str] = Field(None, description="Quốc gia")
    postal_code: Optional[str] = Field(None, description="Mã bưu chính")
    latitude: Optional[float] = Field(None, description="Vĩ độ")
    longitude: Optional[float] = Field(None, description="Kinh độ")
    remote: bool = Field(False, description="Có thể làm việc từ xa không")
    hybrid: bool = Field(False, description="Có thể làm việc kết hợp không")


class Benefit(BaseModel):
    """Phúc lợi công việc"""
    name: str = Field(..., description="Tên phúc lợi")
    description: Optional[str] = Field(None, description="Mô tả phúc lợi")
    category: Optional[str] = Field(None, description="Danh mục phúc lợi")


class JobRequirement(BaseModel):
    """Yêu cầu công việc"""
    skills: List[str] = Field(default_factory=list, description="Kỹ năng yêu cầu")
    education: Optional[str] = Field(None, description="Yêu cầu học vấn")
    experience: Optional[str] = Field(None, description="Yêu cầu kinh nghiệm")
    languages: List[str] = Field(default_factory=list, description="Ngôn ngữ yêu cầu")
    certifications: List[str] = Field(default_factory=list, description="Chứng chỉ yêu cầu")
    other_requirements: List[str] = Field(default_factory=list, description="Yêu cầu khác")


class JobData(BaseModel):
    """Dữ liệu công việc"""
    id: Optional[str] = Field(None, description="ID công việc")
    title: str = Field(..., description="Tiêu đề công việc")
    description: str = Field(..., description="Mô tả công việc")
    company: CompanyInfo = Field(..., description="Thông tin công ty")
    location: Location = Field(..., description="Vị trí địa lý")
    job_type: JobType = Field(JobType.FULL_TIME, description="Loại công việc")
    experience_level: Optional[ExperienceLevel] = Field(None, description="Cấp độ kinh nghiệm")
    salary: Optional[SalaryRange] = Field(None, description="Phạm vi lương")
    requirements: JobRequirement = Field(..., description="Yêu cầu công việc")
    benefits: List[Benefit] = Field(default_factory=list, description="Phúc lợi")
    responsibilities: List[str] = Field(default_factory=list, description="Trách nhiệm công việc")
    application_url: Optional[HttpUrl] = Field(None, description="URL ứng tuyển")
    source: JobSource = Field(JobSource.OTHER, description="Nguồn công việc")
    source_url: Optional[HttpUrl] = Field(None, description="URL nguồn")
    posted_date: Optional[datetime] = Field(None, description="Ngày đăng")
    deadline: Optional[datetime] = Field(None, description="Hạn nộp đơn")
    is_active: bool = Field(True, description="Công việc có đang hoạt động không")
    views_count: Optional[int] = Field(None, description="Số lượt xem")
    applications_count: Optional[int] = Field(None, description="Số lượt ứng tuyển")
    created_at: datetime = Field(default_factory=datetime.now, description="Thời gian tạo")
    updated_at: datetime = Field(default_factory=datetime.now, description="Thời gian cập nhật")
    raw_text: Optional[str] = Field(None, description="Văn bản thô")

    class Config:
        use_enum_values = True


class JobMatch(BaseModel):
    """Kết quả so khớp giữa CV và công việc"""
    job: JobData = Field(..., description="Thông tin công việc")
    match_score: float = Field(..., description="Điểm số khớp")
    skill_matches: List[str] = Field(default_factory=list, description="Kỹ năng khớp")
    missing_skills: List[str] = Field(default_factory=list, description="Kỹ năng thiếu")
    experience_match: bool = Field(False, description="Kinh nghiệm có khớp không")
    education_match: bool = Field(False, description="Học vấn có khớp không")
    location_distance: Optional[float] = Field(None, description="Khoảng cách địa lý (km)")
    salary_match: bool = Field(False, description="Lương có khớp không")
    match_reasons: List[str] = Field(default_factory=list, description="Lý do khớp")
    created_at: datetime = Field(default_factory=datetime.now, description="Thời gian tạo")

    def get_match_summary(self) -> str:
        """Tạo tóm tắt về độ khớp"""
        summary = []

        # Phân tích điểm số khớp
        if self.match_score >= 0.8:
            summary.append(f"Phù hợp tuyệt vời ({self.match_score * 100:.0f}%)")
        elif self.match_score >= 0.6:
            summary.append(f"Phù hợp tốt ({self.match_score * 100:.0f}%)")
        elif self.match_score >= 0.4:
            summary.append(f"Phù hợp trung bình ({self.match_score * 100:.0f}%)")
        else:
            summary.append(f"Phù hợp thấp ({self.match_score * 100:.0f}%)")

        # Thông tin về kỹ năng
        if self.skill_matches:
            summary.append(
                f"Có {len(self.skill_matches)}/{len(self.skill_matches) + len(self.missing_skills)} kỹ năng phù hợp")

        # Thông tin về kinh nghiệm và học vấn
        if self.experience_match:
            summary.append("Kinh nghiệm phù hợp")
        if self.education_match:
            summary.append("Học vấn phù hợp")

        # Thông tin về khoảng cách
        if self.location_distance is not None:
            summary.append(f"Khoảng cách: {self.location_distance:.1f} km")

        # Thông tin về lương
        if self.salary_match:
            summary.append("Mức lương phù hợp")

        return ", ".join(summary)