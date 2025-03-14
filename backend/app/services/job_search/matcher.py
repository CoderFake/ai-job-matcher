"""
Module phân tích mức độ phù hợp giữa CV và công việc
"""

import os
import logging
import json
import re
from typing import Dict, Any, List, Tuple, Optional, Set
from datetime import datetime, timedelta
import math
from pathlib import Path

from app.core.logging import get_logger
from app.core.settings import settings
from app.models.cv import CVData
from app.models.job import JobData, JobMatch

logger = get_logger("job_search")

class JobMatcher:
    """
    Lớp phân tích mức độ phù hợp giữa CV và công việc
    """

    def __init__(self):
        """
        Khởi tạo
        """
        # Trọng số cho các tiêu chí
        self.weights = {
            "skills": 0.40,  # Kỹ năng
            "experience": 0.20,  # Kinh nghiệm
            "education": 0.15,  # Học vấn
            "job_title": 0.15,  # Tiêu đề công việc
            "location": 0.05,  # Địa điểm
            "salary": 0.05  # Mức lương
        }

        # Danh sách các kỹ năng tương tự
        self.similar_skills = {
            "python": ["python", "django", "flask", "pytorch", "tensorflow", "scikit-learn", "pandas", "jupyter", "python3"],
            "javascript": ["javascript", "js", "typescript", "ts", "nodejs", "node.js", "react", "angular", "vue", "jquery"],
            "php": ["php", "laravel", "symfony", "codeigniter", "wordpress", "drupal"],
            "java": ["java", "spring", "spring boot", "hibernate", "j2ee", "jee", "maven", "gradle"],
            "dotnet": ["c#", ".net", "asp.net", "vb.net", "entity framework", "wpf", "xamarin"],
            "database": ["sql", "mysql", "postgresql", "oracle", "mongodb", "cassandra", "redis", "sqlite", "nosql", "db"],
            "mobile": ["android", "ios", "swift", "kotlin", "react native", "flutter", "mobile development", "mobile app"],
            "devops": ["devops", "docker", "kubernetes", "jenkins", "gitlab", "ci/cd", "aws", "azure", "gcp", "cloud"],
            "data": ["data science", "data analysis", "big data", "machine learning", "ai", "artificial intelligence", "deep learning", "neural network", "nlp", "data mining"],
            "web": ["html", "css", "scss", "sass", "bootstrap", "tailwind", "frontend", "web development", "responsive design", "ui"],
            "qa": ["qa", "quality assurance", "testing", "selenium", "automated testing", "test automation", "cypress", "unit testing"],
            "english": ["english", "tiếng anh", "ielts", "toeic"]
        }

        # Thư mục cache
        self.cache_dir = os.path.join(settings.TEMP_DIR, "matcher_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def match(self, cv: CVData, job: JobData) -> JobMatch:
        """
        Phân tích mức độ phù hợp giữa CV và công việc

        Args:
            cv: Dữ liệu CV
            job: Dữ liệu công việc

        Returns:
            JobMatch: Kết quả phân tích
        """
        # Tính điểm cho từng tiêu chí
        skill_score, matched_skills, missing_skills = self._match_skills(cv, job)
        experience_score, experience_match = self._match_experience(cv, job)
        education_score, education_match = self._match_education(cv, job)
        job_title_score = self._match_job_title(cv, job)
        location_score, location_distance = self._match_location(cv, job)
        salary_score, salary_match = self._match_salary(cv, job)

        # Tính tổng điểm
        total_score = (
            skill_score * self.weights["skills"] +
            experience_score * self.weights["experience"] +
            education_score * self.weights["education"] +
            job_title_score * self.weights["job_title"] +
            location_score * self.weights["location"] +
            salary_score * self.weights["salary"]
        )

        # Tạo danh sách lý do khớp
        match_reasons = []

        if matched_skills:
            skills_str = ", ".join(matched_skills[:5])
            if len(matched_skills) > 5:
                skills_str += f" và {len(matched_skills) - 5} kỹ năng khác"
            match_reasons.append(f"Có các kỹ năng phù hợp: {skills_str}")

        if experience_match:
            match_reasons.append("Kinh nghiệm phù hợp với yêu cầu công việc")

        if education_match:
            match_reasons.append("Trình độ học vấn đáp ứng yêu cầu")

        if job_title_score > 0.7:
            match_reasons.append("Vị trí công việc phù hợp với kinh nghiệm")

        if location_distance is not None and location_distance < 10:
            match_reasons.append(f"Địa điểm làm việc gần (chỉ cách {location_distance:.1f} km)")

        if salary_match:
            match_reasons.append("Mức lương phù hợp với mong muốn")

        # Tạo đối tượng JobMatch
        job_match = JobMatch(
            job=job,
            match_score=total_score,
            skill_matches=matched_skills,
            missing_skills=missing_skills,
            experience_match=experience_match,
            education_match=education_match,
            location_distance=location_distance,
            salary_match=salary_match,
            match_reasons=match_reasons
        )

        return job_match

    def match_multiple(self, cv: CVData, jobs: List[JobData]) -> List[JobMatch]:
        """
        Phân tích mức độ phù hợp giữa một CV và nhiều công việc

        Args:
            cv: Dữ liệu CV
            jobs: Danh sách công việc

        Returns:
            List[JobMatch]: Danh sách kết quả phân tích
        """
        results = []

        for job in jobs:
            match_result = self.match(cv, job)
            results.append(match_result)

        # Sắp xếp kết quả theo điểm số giảm dần
        results.sort(key=lambda x: x.match_score, reverse=True)

        return results

    def find_best_matches(self, cv: CVData, jobs: List[JobData], limit: int = 10) -> List[JobMatch]:
        """
        Tìm những công việc phù hợp nhất với CV

        Args:
            cv: Dữ liệu CV
            jobs: Danh sách công việc
            limit: Số lượng kết quả tối đa

        Returns:
            List[JobMatch]: Danh sách kết quả phân tích đã sắp xếp
        """
        # Kiểm tra cache
        cache_key = self._get_cache_key(cv, jobs)
        cached_results = self._get_from_cache(cache_key)

        if cached_results:
            return cached_results[:limit]

        # Phân tích mức độ phù hợp
        results = self.match_multiple(cv, jobs)

        # Lưu vào cache
        self._save_to_cache(cache_key, results)

        # Trả về kết quả
        return results[:limit]

    def get_job_recommendations(self, cv: CVData, jobs: List[JobData],
                              min_score: float = 0.6, limit: int = 10) -> List[JobMatch]:
        """
        Lấy các đề xuất công việc phù hợp với CV

        Args:
            cv: Dữ liệu CV
            jobs: Danh sách công việc
            min_score: Điểm số tối thiểu để đề xuất
            limit: Số lượng đề xuất tối đa

        Returns:
            List[JobMatch]: Danh sách đề xuất công việc
        """
        # Phân tích mức độ phù hợp
        match_results = self.match_multiple(cv, jobs)

        # Lọc kết quả theo điểm số tối thiểu
        recommendations = [match for match in match_results if match.match_score >= min_score]

        # Trả về kết quả
        return recommendations[:limit]

    def analyze_job_requirements(self, cv: CVData, jobs: List[JobData]) -> Dict[str, Any]:
        """
            Phân tích yêu cầu công việc và so sánh với kỹ năng trong CV

            Args:
                cv: Dữ liệu CV
                jobs: Danh sách công việc

            Returns:
                Dict[str, Any]: Kết quả phân tích
        """
        skill_counts = {}
        experience_levels = {}
        education_levels = {}
        salary_ranges = {"min": [], "max": []}

        # Lấy kỹ năng từ CV
        cv_skills = []
        if cv.skills:
            cv_skills = [skill.name.lower() for skill in cv.skills]

        # Tạo tập hợp kỹ năng từ CV (bao gồm cả kỹ năng tương tự)
        cv_skill_variations = set()
        for skill in cv_skills:
            cv_skill_variations.add(skill)
            # Thêm các biến thể của kỹ năng
            for category, similar_skills in self.similar_skills.items():
                if skill in similar_skills:
                    for similar_skill in similar_skills:
                        cv_skill_variations.add(similar_skill)

        # Phân tích công việc
        for job in jobs:
            # Phân tích kỹ năng
            if job.requirements and job.requirements.skills:
                for skill in job.requirements.skills:
                    skill_lower = skill.lower()
                    if skill_lower not in skill_counts:
                        skill_counts[skill_lower] = {"count": 0, "in_cv": False}

                    skill_counts[skill_lower]["count"] += 1

                    # Kiểm tra xem kỹ năng có trong CV không
                    if any(re.search(r'\b' + re.escape(cv_skill) + r'\b', skill_lower) for cv_skill in cv_skill_variations):
                        skill_counts[skill_lower]["in_cv"] = True

            # Phân tích cấp độ kinh nghiệm
            if job.experience_level:
                exp_level = job.experience_level.value
                if exp_level not in experience_levels:
                    experience_levels[exp_level] = 0
                experience_levels[exp_level] += 1

            # Phân tích trình độ học vấn
            if job.requirements and job.requirements.education:
                edu = job.requirements.education.lower()
                if edu not in education_levels:
                    education_levels[edu] = 0
                education_levels[edu] += 1

            # Phân tích mức lương
            if job.salary and job.salary.is_disclosed:
                if job.salary.min is not None:
                    salary_ranges["min"].append(job.salary.min)
                if job.salary.max is not None:
                    salary_ranges["max"].append(job.salary.max)

        # Sắp xếp kỹ năng theo số lần xuất hiện giảm dần
        skill_counts = {k: v for k, v in sorted(skill_counts.items(), key=lambda item: item[1]["count"], reverse=True)}

        # Tính toán thông tin về mức lương
        salary_stats = {}
        if salary_ranges["min"]:
            salary_stats["min_avg"] = sum(salary_ranges["min"]) / len(salary_ranges["min"])
            salary_stats["min_median"] = sorted(salary_ranges["min"])[len(salary_ranges["min"]) // 2]
        if salary_ranges["max"]:
            salary_stats["max_avg"] = sum(salary_ranges["max"]) / len(salary_ranges["max"])
            salary_stats["max_median"] = sorted(salary_ranges["max"])[len(salary_ranges["max"]) // 2]

        # Thống kê kỹ năng thiếu
        missing_skills = []
        for skill, info in skill_counts.items():
            if info["count"] >= 3 and not info["in_cv"]:  # Yêu cầu xuất hiện ít nhất 3 lần
                missing_skills.append({"name": skill, "count": info["count"]})

        # Tạo kết quả phân tích
        analysis = {
            "skill_counts": skill_counts,
            "experience_levels": experience_levels,
            "education_levels": education_levels,
            "salary_stats": salary_stats,
            "missing_skills": missing_skills,
            "total_jobs": len(jobs)
        }

        return analysis

    def calculate_skill_gap(self, cv: CVData, job: JobData) -> Dict[str, Any]:
        """
        Tính toán khoảng cách kỹ năng giữa CV và công việc

        Args:
            cv: Dữ liệu CV
            job: Dữ liệu công việc

        Returns:
            Dict[str, Any]: Kết quả phân tích
        """
        # Lấy kỹ năng từ CV
        cv_skills = []
        if cv.skills:
            cv_skills = [skill.name.lower() for skill in cv.skills]

        # Lấy kỹ năng từ công việc
        job_skills = []
        if job.requirements and job.requirements.skills:
            job_skills = [skill.lower() for skill in job.requirements.skills]

        # Tính toán kỹ năng trùng và kỹ năng thiếu
        matched_skills = []
        missing_skills = []

        for job_skill in job_skills:
            is_matched = False
            for cv_skill in cv_skills:
                if self._is_skill_match(cv_skill, job_skill):
                    matched_skills.append(job_skill)
                    is_matched = True
                    break

            if not is_matched:
                missing_skills.append(job_skill)

        # Tính điểm phù hợp
        if job_skills:
            match_score = len(matched_skills) / len(job_skills)
        else:
            match_score = 1.0  # Nếu không yêu cầu kỹ năng, mặc định điểm là 1.0

        # Tạo kết quả phân tích
        skill_gap = {
            "total_required_skills": len(job_skills),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "match_score": match_score,
            "gap_percentage": (1 - match_score) * 100
        }

        return skill_gap

    def get_skill_recommendations(self, cv: CVData, jobs: List[JobData], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Đề xuất kỹ năng cần học để tăng cơ hội việc làm

        Args:
            cv: Dữ liệu CV
            jobs: Danh sách công việc
            limit: Số lượng đề xuất tối đa

        Returns:
            List[Dict[str, Any]]: Danh sách đề xuất kỹ năng
        """
        # Phân tích yêu cầu công việc
        analysis = self.analyze_job_requirements(cv, jobs)

        # Lấy danh sách kỹ năng thiếu
        missing_skills = analysis["missing_skills"]

        # Sắp xếp theo số lần xuất hiện giảm dần
        missing_skills.sort(key=lambda x: x["count"], reverse=True)

        # Đề xuất khóa học
        recommendations = []

        for skill in missing_skills[:limit]:
            # Tìm các kỹ năng liên quan
            related_skills = []
            skill_name = skill["name"]

            # Tìm nhóm kỹ năng chứa kỹ năng này
            for category, similar_skills in self.similar_skills.items():
                if skill_name in similar_skills:
                    related_skills = [s for s in similar_skills if s != skill_name]
                    break

            recommendation = {
                "skill": skill_name,
                "count": skill["count"],
                "related_skills": related_skills,
                "importance": "Cao" if skill["count"] > 5 else "Trung bình",
                "resource_types": ["Khóa học", "Tài liệu", "Thực hành"]
            }

            recommendations.append(recommendation)

        return recommendations

    def _match_skills(self, cv: CVData, job: JobData) -> Tuple[float, List[str], List[str]]:
        """
        Phân tích mức độ phù hợp về kỹ năng

        Args:
            cv: Dữ liệu CV
            job: Dữ liệu công việc

        Returns:
            Tuple[float, List[str], List[str]]: (Điểm số, Kỹ năng khớp, Kỹ năng thiếu)
        """
        # Lấy kỹ năng từ CV
        cv_skills = []
        if cv.skills:
            cv_skills = [skill.name.lower() for skill in cv.skills]

        # Lấy kỹ năng từ mô tả công việc
        job_skills = []
        if job.requirements and job.requirements.skills:
            job_skills = [skill.lower() for skill in job.requirements.skills]

        # Nếu không có yêu cầu kỹ năng, mặc định điểm là 1.0
        if not job_skills:
            return 1.0, [], []

        # Tính toán kỹ năng trùng và kỹ năng thiếu
        matched_skills = []
        missing_skills = []

        for job_skill in job_skills:
            is_matched = False
            for cv_skill in cv_skills:
                if self._is_skill_match(cv_skill, job_skill):
                    matched_skills.append(job_skill)
                    is_matched = True
                    break

            if not is_matched:
                missing_skills.append(job_skill)

        # Tính điểm phù hợp
        score = len(matched_skills) / len(job_skills)

        return score, matched_skills, missing_skills

    def _is_skill_match(self, cv_skill: str, job_skill: str) -> bool:
        """
        Kiểm tra xem hai kỹ năng có khớp nhau không

        Args:
            cv_skill: Kỹ năng từ CV
            job_skill: Kỹ năng từ công việc

        Returns:
            bool: True nếu hai kỹ năng khớp nhau
        """
        # Kiểm tra chính xác
        if cv_skill == job_skill:
            return True

        # Kiểm tra một kỹ năng có chứa kỹ năng kia không
        if cv_skill in job_skill or job_skill in cv_skill:
            return True

        # Kiểm tra hai kỹ năng có thuộc cùng nhóm không
        for similar_skills in self.similar_skills.values():
            if cv_skill in similar_skills and job_skill in similar_skills:
                return True

        return False

    def _match_experience(self, cv: CVData, job: JobData) -> Tuple[float, bool]:
        """
        Phân tích mức độ phù hợp về kinh nghiệm

        Args:
            cv: Dữ liệu CV
            job: Dữ liệu công việc

        Returns:
            Tuple[float, bool]: (Điểm số, Có phù hợp không)
        """
        # Nếu công việc không yêu cầu cấp độ kinh nghiệm, mặc định điểm là 1.0
        if not job.experience_level:
            return 1.0, True

        # Nếu CV không có thông tin về số năm kinh nghiệm, điểm là 0.5
        if cv.years_of_experience is None:
            return 0.5, False

        # Chuyển đổi cấp độ kinh nghiệm thành số năm kinh nghiệm tối thiểu
        exp_level_to_years = {
            "entry": 0,
            "junior": 1,
            "mid": 3,
            "senior": 5,
            "manager": 7,
            "director": 10,
            "executive": 12
        }

        required_years = exp_level_to_years.get(job.experience_level, 0)

        # So sánh số năm kinh nghiệm
        if cv.years_of_experience >= required_years:
            return 1.0, True
        elif cv.years_of_experience >= required_years * 0.8:
            return 0.8, False  # Gần đủ
        elif cv.years_of_experience >= required_years * 0.6:
            return 0.6, False  # Tương đối gần
        else:
            return 0.3, False  # Thiếu nhiều

    def _match_education(self, cv: CVData, job: JobData) -> Tuple[float, bool]:
        """
        Phân tích mức độ phù hợp về học vấn

        Args:
            cv: Dữ liệu CV
            job: Dữ liệu công việc

        Returns:
            Tuple[float, bool]: (Điểm số, Có phù hợp không)
        """
        # Nếu công việc không yêu cầu học vấn, mặc định điểm là 1.0
        if not job.requirements or not job.requirements.education:
            return 1.0, True

        # Nếu CV không có thông tin về học vấn, điểm là 0.5
        if not cv.education:
            return 0.5, False

        # Chuyển đổi yêu cầu học vấn thành cấp độ
        edu_level_mapping = {
            "phổ thông": 1,
            "trung cấp": 2,
            "cao đẳng": 3,
            "đại học": 4,
            "cử nhân": 4,
            "thạc sĩ": 5,
            "tiến sĩ": 6,
            "high school": 1,
            "associate": 3,
            "college": 3,
            "bachelor": 4,
            "bachelor's": 4,
            "master": 5,
            "master's": 5,
            "phd": 6,
            "doctorate": 6
        }

        # Tìm cấp độ học vấn yêu cầu
        required_edu = job.requirements.education.lower()
        required_level = 1  # Mặc định là phổ thông

        for key, level in edu_level_mapping.items():
            if key in required_edu:
                required_level = level
                break

        # Tìm cấp độ học vấn cao nhất từ CV
        cv_level = 1  # Mặc định là phổ thông

        for edu in cv.education:
            if edu.degree:
                degree = edu.degree.lower()
                for key, level in edu_level_mapping.items():
                    if key in degree:
                        cv_level = max(cv_level, level)
                        break

        # So sánh cấp độ học vấn
        if cv_level >= required_level:
            return 1.0, True
        elif cv_level == required_level - 1:
            return 0.7, False  # Gần đủ
        else:
            return 0.4, False  # Thiếu nhiều

    def _match_job_title(self, cv: CVData, job: JobData) -> float:
        """
        Phân tích mức độ phù hợp về tiêu đề công việc

        Args:
            cv: Dữ liệu CV
            job: Dữ liệu công việc

        Returns:
            float: Điểm số
        """
        # Nếu CV không có thông tin về vị trí công việc, điểm là 0.5
        if not cv.job_title:
            return 0.5

        # Tính độ tương đồng giữa tiêu đề công việc trong CV và công việc mới
        cv_title = cv.job_title.lower()
        job_title = job.title.lower()

        # Đơn giản hóa tiêu đề công việc (loại bỏ các từ phổ biến)
        common_words = ["senior", "junior", "mid", "level", "lead", "manager", "director", "executive", "internship", "intern", "experienced"]

        for word in common_words:
            cv_title = cv_title.replace(word, "")
            job_title = job_title.replace(word, "")

        # Tách thành từ
        cv_words = set(re.findall(r'\w+', cv_title))
        job_words = set(re.findall(r'\w+', job_title))

        # Tính độ tương đồng Jaccard
        if cv_words and job_words:
            intersection = len(cv_words.intersection(job_words))
            union = len(cv_words.union(job_words))
            return intersection / union

        return 0.5

    def _match_location(self, cv: CVData, job: JobData) -> Tuple[float, Optional[float]]:
        """
        Phân tích mức độ phù hợp về địa điểm

        Args:
            cv: Dữ liệu CV
            job: Dữ liệu công việc

        Returns:
            Tuple[float, Optional[float]]: (Điểm số, Khoảng cách)
        """
        # Nếu CV không có thông tin về địa điểm mong muốn, điểm là 0.5
        if not cv.preferred_location:
            return 0.5, None

        # Nếu công việc không có thông tin về địa điểm, điểm là 0.7 (không thể xác định khoảng cách)
        if not job.location or not job.location.city:
            return 0.7, None

        # Nếu địa điểm công việc khớp với địa điểm mong muốn, điểm là 1.0 và khoảng cách là 0
        if cv.preferred_location.lower() == job.location.city.lower():
            return 1.0, 0.0

        # Tính khoảng cách giữa hai địa điểm (nếu có thể)
        try:
            distance = self._calculate_distance(cv.preferred_location, job.location.city)

            # Điểm số dựa trên khoảng cách
            if distance <= 5:
                return 0.9, distance  # Rất gần
            elif distance <= 10:
                return 0.8, distance  # Gần
            elif distance <= 20:
                return 0.6, distance  # Khá gần
            elif distance <= 50:
                return 0.4, distance  # Trung bình
            else:
                return 0.2, distance  # Xa
        except:
            # Nếu không thể tính khoảng cách, điểm là 0.5
            return 0.5, None

    def _calculate_distance(self, location1: str, location2: str) -> float:
        """
        Tính khoảng cách giữa hai địa điểm

        Args:
            location1: Địa điểm thứ nhất
            location2: Địa điểm thứ hai

        Returns:
            float: Khoảng cách (km)
        """
        # Đây là một hàm giả định, trong thực tế cần sử dụng API như Google Maps
        # hoặc một cơ sở dữ liệu địa lý

        # Chuẩn hóa địa điểm
        location1 = location1.lower()
        location2 = location2.lower()

        # Xác định khoảng cách giữa các thành phố lớn (đơn vị: km)
        distances = {
            ("hà nội", "hồ chí minh"): 1714,
            ("hà nội", "đà nẵng"): 764,
            ("hồ chí minh", "đà nẵng"): 970,
            ("hà nội", "hải phòng"): 102,
            ("hồ chí minh", "cần thơ"): 169,
            ("hà nội", "vinh"): 300,
            ("hồ chí minh", "nha trang"): 411,
            ("hồ chí minh", "vũng tàu"): 107,
            ("hà nội", "huế"): 654
        }

        # Xác định các biến thể của tên thành phố
        city_variants = {
            "hà nội": ["hà nội", "hanoi", "ha noi"],
            "hồ chí minh": ["hồ chí minh", "tp hcm", "tp. hcm", "hochiminh", "ho chi minh", "sài gòn", "sai gon", "saigon", "thành phố hồ chí minh"],
            "đà nẵng": ["đà nẵng", "da nang", "danang"],
            "hải phòng": ["hải phòng", "hai phong", "haiphong"],
            "cần thơ": ["cần thơ", "can tho", "cantho"],
            "vinh": ["vinh", "nghệ an", "nghe an"],
            "nha trang": ["nha trang", "nhatrang", "khánh hòa", "khanh hoa"],
            "vũng tàu": ["vũng tàu", "vung tau", "bà rịa vũng tàu", "ba ria vung tau"],
            "huế": ["huế", "hue", "thừa thiên huế", "thua thien hue"]
        }

        # Chuẩn hóa tên thành phố
        city1 = None
        city2 = None

        for city, variants in city_variants.items():
            if any(variant in location1 for variant in variants):
                city1 = city
            if any(variant in location2 for variant in variants):
                city2 = city

        if city1 is None or city2 is None:
            raise ValueError("Không thể xác định thành phố")

        # Nếu hai thành phố giống nhau, khoảng cách là 0
        if city1 == city2:
            return 0.0

        # Tìm khoảng cách
        key = (city1, city2)
        reverse_key = (city2, city1)

        if key in distances:
            return distances[key]
        elif reverse_key in distances:
            return distances[reverse_key]
        else:
            # Nếu không có dữ liệu về khoảng cách, trả về giá trị mặc định
            return 1000.0

    def _match_salary(self, cv: CVData, job: JobData) -> Tuple[float, bool]:
        """
        Phân tích mức độ phù hợp về mức lương

        Args:
            cv: Dữ liệu CV
            job: Dữ liệu công việc

        Returns:
            Tuple[float, bool]: (Điểm số, Có phù hợp không)
        """
        # Nếu công việc không có thông tin về mức lương, điểm là 0.5
        if not job.salary or not job.salary.is_disclosed:
            return 0.5, False

        # Nếu CV không có thông tin về mức lương mong muốn, điểm là 0.7
        if not cv.salary_expectation:
            return 0.7, False

        # Phân tích mức lương mong muốn từ CV
        # Mức lương mong muốn thường được lưu dưới dạng chuỗi, cần chuyển đổi thành số
        try:
            expected_salary = self._parse_expected_salary(cv.salary_expectation)
        except:
            return 0.7, False

        # So sánh mức lương
        if job.salary.min is not None and job.salary.max is not None:
            # Trường hợp có cả mức lương tối thiểu và tối đa
            if expected_salary <= job.salary.max:
                if expected_salary >= job.salary.min:
                    return 1.0, True  # Nằm trong khoảng lương
                elif expected_salary >= job.salary.min * 0.8:
                    return 0.8, False  # Gần với mức lương tối thiểu
                else:
                    return 0.6, False  # Thấp hơn mức lương tối thiểu
            else:
                if expected_salary <= job.salary.max * 1.2:
                    return 0.7, False  # Gần với mức lương tối đa
                else:
                    return 0.4, False  # Cao hơn nhiều so với mức lương tối đa
        elif job.salary.min is not None:
            # Trường hợp chỉ có mức lương tối thiểu
            if expected_salary >= job.salary.min:
                return 1.0, True  # Cao hơn hoặc bằng mức lương tối thiểu
            elif expected_salary >= job.salary.min * 0.8:
                return 0.8, False  # Gần với mức lương tối thiểu
            else:
                return 0.6, False  # Thấp hơn mức lương tối thiểu
        elif job.salary.max is not None:
            # Trường hợp chỉ có mức lương tối đa
            if expected_salary <= job.salary.max:
                return 1.0, True  # Thấp hơn hoặc bằng mức lương tối đa
            elif expected_salary <= job.salary.max * 1.2:
                return 0.7, False  # Gần với mức lương tối đa
            else:
                return 0.4, False  # Cao hơn nhiều so với mức lương tối đa
        else:
            return 0.5, False  # Không thể xác định

    def _parse_expected_salary(self, salary_text: str) -> float:
        """
        Phân tích mức lương mong muốn từ chuỗi

        Args:
            salary_text: Chuỗi mức lương

        Returns:
            float: Mức lương (VND)
        """
        # Loại bỏ các ký tự không phải số và đơn vị tiền tệ
        salary_text = salary_text.lower()

        # Xử lý trường hợp đơn vị USD
        if "$" in salary_text or "usd" in salary_text:
            # Chuyển đổi USD sang VND (1 USD = 23,000 VND)
            match = re.search(r'(\d[\d\s.,]*)', salary_text)
            if match:
                usd_amount = float(match.group(1).replace(",", "").replace(".", "").replace(" ", ""))
                return usd_amount * 23000

        # Xử lý trường hợp đơn vị triệu VND
        if "triệu" in salary_text or "trieu" in salary_text or "tr" in salary_text:
            match = re.search(r'(\d[\d\s.,]*)', salary_text)
            if match:
                amount = float(match.group(1).replace(",", "").replace(".", "").replace(" ", ""))
                return amount * 1000000

        # Xử lý trường hợp số tiền VND
        match = re.search(r'(\d[\d\s.,]*)', salary_text)
        if match:
            return float(match.group(1).replace(",", "").replace(".", "").replace(" ", ""))

        # Mặc định
        return 10000000  # Giá trị mặc định

    def _get_cache_key(self, cv: CVData, jobs: List[JobData]) -> str:
        """
        Tạo khóa cache từ CV và danh sách công việc

        Args:
            cv: Dữ liệu CV
            jobs: Danh sách công việc

        Returns:
            str: Khóa cache
        """
        # Tạo chuỗi đại diện cho CV
        cv_str = f"{cv.personal_info.name}_{len(cv.skills)}_{cv.years_of_experience}"

        # Tạo chuỗi đại diện cho danh sách công việc
        jobs_str = "_".join([f"{job.id or ''}_{job.title}" for job in jobs[:5]])

        # Tạo hash từ chuỗi
        import hashlib
        return hashlib.md5((cv_str + jobs_str).encode("utf-8")).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[List[JobMatch]]:
        """
        Lấy kết quả từ cache

        Args:
            cache_key: Khóa cache

        Returns:
            Optional[List[JobMatch]]: Danh sách kết quả phân tích từ cache
        """
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")

        # Kiểm tra nếu tệp cache tồn tại và còn hiệu lực
        if os.path.exists(cache_path):
            try:
                # Kiểm tra thời gian tạo tệp
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
                if datetime.now() - file_time < timedelta(hours=24):  # Cache có hiệu lực trong 24 giờ
                    # Đọc dữ liệu cache
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)

                    # Chuyển đổi thành đối tượng JobMatch
                    results = []
                    for match_dict in cache_data:
                        try:
                            job_dict = match_dict.pop("job")
                            job = JobData.model_validate(job_dict)
                            match_dict["job"] = job
                            match = JobMatch.model_validate(match_dict)
                            results.append(match)
                        except Exception as e:
                            logger.error(f"Lỗi khi chuyển đổi dữ liệu cache: {str(e)}")

                    return results
            except Exception as e:
                logger.error(f"Lỗi khi đọc cache: {str(e)}")

        return None

    def _save_to_cache(self, cache_key: str, results: List[JobMatch]) -> None:
        """
        Lưu kết quả vào cache

        Args:
            cache_key: Khóa cache
            results: Danh sách kết quả phân tích
        """
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")

        try:
            # Chuyển đổi danh sách JobMatch thành JSON
            match_dicts = []
            for match in results:
                match_dict = match.model_dump()
                job_dict = match_dict.pop("job")
                match_dict["job"] = job_dict
                match_dicts.append(match_dict)

            # Ghi vào tệp cache
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(match_dicts, f, ensure_ascii=False, indent=2)

            logger.info(f"Đã lưu {len(results)} kết quả phân tích vào cache: {cache_key}")

        except Exception as e:
            logger.error(f"Lỗi khi lưu cache: {str(e)}")

    def generate_job_insights(self, cv: CVData, job_matches: List[JobMatch]) -> Dict[str, Any]:
        """
        Tạo báo cáo phân tích tổng quan về các công việc phù hợp

        Args:
            cv: Dữ liệu CV
            job_matches: Danh sách kết quả phân tích

        Returns:
            Dict[str, Any]: Báo cáo phân tích
        """
        insights = {
            "total_matches": len(job_matches),
            "average_match_score": 0,
            "high_matches": 0,  # Số lượng công việc có điểm số cao (>= 0.8)
            "medium_matches": 0,  # Số lượng công việc có điểm số trung bình (>= 0.6 và < 0.8)
            "low_matches": 0,  # Số lượng công việc có điểm số thấp (< 0.6)
            "common_missing_skills": {},  # Kỹ năng thiếu phổ biến
            "skill_match_rate": 0,  # Tỷ lệ kỹ năng khớp trung bình
            "location_distribution": {},  # Phân bố địa điểm
            "salary_range": {
                "min": None,
                "max": None,
                "average": None
            },
            "common_job_titles": {},  # Tiêu đề công việc phổ biến
            "experience_level_distribution": {},  # Phân bố cấp độ kinh nghiệm
            "recommendations": []  # Đề xuất cải thiện
        }

        if not job_matches:
            return insights

        # Tính toán các chỉ số
        total_score = 0
        all_missing_skills = []
        location_counts = {}
        salaries = []
        job_titles = {}
        experience_levels = {}

        for match in job_matches:
            # Điểm số
            total_score += match.match_score

            if match.match_score >= 0.8:
                insights["high_matches"] += 1
            elif match.match_score >= 0.6:
                insights["medium_matches"] += 1
            else:
                insights["low_matches"] += 1

            # Kỹ năng thiếu
            all_missing_skills.extend(match.missing_skills)

            # Địa điểm
            if match.job.location and match.job.location.city:
                location = match.job.location.city
                location_counts[location] = location_counts.get(location, 0) + 1

            # Mức lương
            if match.job.salary and match.job.salary.is_disclosed:
                if match.job.salary.min is not None:
                    salaries.append(match.job.salary.min)
                if match.job.salary.max is not None:
                    salaries.append(match.job.salary.max)

            # Tiêu đề công việc
            title = match.job.title
            job_titles[title] = job_titles.get(title, 0) + 1

            # Cấp độ kinh nghiệm
            if match.job.experience_level:
                exp_level = match.job.experience_level.value
                experience_levels[exp_level] = experience_levels.get(exp_level, 0) + 1

        # Cập nhật insights
        insights["average_match_score"] = total_score / len(job_matches)

        # Kỹ năng thiếu phổ biến
        if all_missing_skills:
            from collections import Counter
            missing_skills_counter = Counter(all_missing_skills)
            insights["common_missing_skills"] = {skill: count for skill, count in missing_skills_counter.most_common(5)}

        # Tỷ lệ kỹ năng khớp
        total_skills = sum(len(match.skill_matches) + len(match.missing_skills) for match in job_matches)
        total_matched_skills = sum(len(match.skill_matches) for match in job_matches)
        if total_skills > 0:
            insights["skill_match_rate"] = total_matched_skills / total_skills

        # Phân bố địa điểm
        insights["location_distribution"] = {k: v for k, v in
                                             sorted(location_counts.items(), key=lambda item: item[1], reverse=True)}

        # Mức lương
        if salaries:
            insights["salary_range"]["min"] = min(salaries)
            insights["salary_range"]["max"] = max(salaries)
            insights["salary_range"]["average"] = sum(salaries) / len(salaries)

        # Tiêu đề công việc phổ biến
        insights["common_job_titles"] = {k: v for k, v in
                                         sorted(job_titles.items(), key=lambda item: item[1], reverse=True)[:5]}

        # Phân bố cấp độ kinh nghiệm
        insights["experience_level_distribution"] = experience_levels

        # Đề xuất cải thiện
        recommendations = []

        # Đề xuất dựa trên kỹ năng thiếu
        if insights["common_missing_skills"]:
            recommendations.append({
                "type": "skill",
                "message": "Bổ sung các kỹ năng còn thiếu để tăng cơ hội việc làm",
                "details": list(insights["common_missing_skills"].keys())
            })

        # Đề xuất dựa trên mức lương
        if cv.salary_expectation and insights["salary_range"]["average"]:
            expected_salary = self._parse_expected_salary(cv.salary_expectation)
            if expected_salary > insights["salary_range"]["max"] * 1.2:
                recommendations.append({
                    "type": "salary",
                    "message": "Xem xét điều chỉnh mức lương mong muốn để phù hợp với thị trường",
                    "details": [f"Mức lương trung bình: {insights['salary_range']['average']:,.0f} VND"]
                })

        # Đề xuất dựa trên địa điểm
        if cv.preferred_location and insights["location_distribution"]:
            if cv.preferred_location not in insights["location_distribution"]:
                recommendations.append({
                    "type": "location",
                    "message": "Cân nhắc mở rộng địa điểm làm việc mong muốn",
                    "details": list(insights["location_distribution"].keys())[:3]
                })

        insights["recommendations"] = recommendations

        return insights

