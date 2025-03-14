from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr


class Education(BaseModel):
    """Thông tin về học vấn"""
    institution: str = Field(..., description="Tên trường")
    degree: Optional[str] = Field(None, description="Bằng cấp")
    field_of_study: Optional[str] = Field(None, description="Ngành học")
    start_date: Optional[datetime] = Field(None, description="Ngày bắt đầu")
    end_date: Optional[datetime] = Field(None, description="Ngày kết thúc")
    description: Optional[str] = Field(None, description="Mô tả")
    gpa: Optional[float] = Field(None, description="Điểm trung bình")


class WorkExperience(BaseModel):
    """Thông tin về kinh nghiệm làm việc"""
    company: str = Field(..., description="Tên công ty")
    position: str = Field(..., description="Vị trí")
    start_date: Optional[datetime] = Field(None, description="Ngày bắt đầu")
    end_date: Optional[datetime] = Field(None, description="Ngày kết thúc")
    current: bool = Field(False, description="Có phải công việc hiện tại không")
    description: Optional[str] = Field(None, description="Mô tả")
    achievements: Optional[List[str]] = Field(None, description="Thành tựu")
    location: Optional[str] = Field(None, description="Địa điểm")


class Skill(BaseModel):
    """Thông tin về kỹ năng"""
    name: str = Field(..., description="Tên kỹ năng")
    level: Optional[str] = Field(None, description="Cấp độ kỹ năng")
    years: Optional[int] = Field(None, description="Số năm kinh nghiệm")
    category: Optional[str] = Field(None, description="Danh mục kỹ năng")


class Language(BaseModel):
    """Thông tin về ngôn ngữ"""
    name: str = Field(..., description="Tên ngôn ngữ")
    proficiency: Optional[str] = Field(None, description="Mức độ thành thạo")


class Project(BaseModel):
    """Thông tin về dự án"""
    name: str = Field(..., description="Tên dự án")
    description: Optional[str] = Field(None, description="Mô tả")
    start_date: Optional[datetime] = Field(None, description="Ngày bắt đầu")
    end_date: Optional[datetime] = Field(None, description="Ngày kết thúc")
    technologies: Optional[List[str]] = Field(None, description="Công nghệ sử dụng")
    url: Optional[str] = Field(None, description="Liên kết đến dự án")
    role: Optional[str] = Field(None, description="Vai trò trong dự án")


class Certificate(BaseModel):
    """Thông tin về chứng chỉ"""
    name: str = Field(..., description="Tên chứng chỉ")
    issuer: Optional[str] = Field(None, description="Tổ chức cấp")
    date_issued: Optional[datetime] = Field(None, description="Ngày cấp")
    expiration_date: Optional[datetime] = Field(None, description="Ngày hết hạn")
    credential_id: Optional[str] = Field(None, description="ID chứng chỉ")
    url: Optional[str] = Field(None, description="Liên kết đến chứng chỉ")


class PersonalInfo(BaseModel):
    """Thông tin cá nhân"""
    name: str = Field(..., description="Họ và tên")
    email: Optional[EmailStr] = Field(None, description="Email")
    phone: Optional[str] = Field(None, description="Số điện thoại")
    address: Optional[str] = Field(None, description="Địa chỉ")
    city: Optional[str] = Field(None, description="Thành phố")
    country: Optional[str] = Field(None, description="Quốc gia")
    linkedin: Optional[str] = Field(None, description="LinkedIn")
    github: Optional[str] = Field(None, description="GitHub")
    website: Optional[str] = Field(None, description="Trang web cá nhân")
    summary: Optional[str] = Field(None, description="Tóm tắt bản thân")
    date_of_birth: Optional[datetime] = Field(None, description="Ngày sinh")


class CVData(BaseModel):
    """Dữ liệu đã phân tích từ CV"""
    personal_info: PersonalInfo = Field(..., description="Thông tin cá nhân")
    education: List[Education] = Field(default_factory=list, description="Học vấn")
    work_experience: List[WorkExperience] = Field(default_factory=list, description="Kinh nghiệm làm việc")
    skills: List[Skill] = Field(default_factory=list, description="Kỹ năng")
    languages: List[Language] = Field(default_factory=list, description="Ngôn ngữ")
    projects: List[Project] = Field(default_factory=list, description="Dự án")
    certificates: List[Certificate] = Field(default_factory=list, description="Chứng chỉ")
    resume_summary: Optional[str] = Field(None, description="Tóm tắt hồ sơ")
    job_title: Optional[str] = Field(None, description="Vị trí công việc")
    years_of_experience: Optional[int] = Field(None, description="Số năm kinh nghiệm")
    salary_expectation: Optional[str] = Field(None, description="Mức lương mong muốn")
    preferred_location: Optional[str] = Field(None, description="Địa điểm làm việc mong muốn")
    created_at: datetime = Field(default_factory=datetime.now, description="Thời gian tạo")
    extracted_from_file: Optional[str] = Field(None, description="Tên tệp nguồn")
    extracted_from_image: bool = Field(False, description="Được trích xuất từ hình ảnh")
    confidence_score: Optional[float] = Field(None, description="Độ tin cậy của việc trích xuất")
    raw_text: Optional[str] = Field(None, description="Văn bản thô")

    def to_job_search_query(self) -> Dict[str, Any]:
        """Chuyển đổi CV thành truy vấn tìm kiếm việc làm"""
        skills = [skill.name for skill in self.skills]
        experiences = [exp.position for exp in self.work_experience]

        query = {
            "job_title": self.job_title or (self.work_experience[0].position if self.work_experience else ""),
            "skills": skills,
            "experience": experiences,
            "education": [edu.field_of_study for edu in self.education if edu.field_of_study],
            "location": self.preferred_location or (self.personal_info.city if self.personal_info.city else ""),
            "languages": [lang.name for lang in self.languages],
            "years_of_experience": self.years_of_experience
        }

        return query