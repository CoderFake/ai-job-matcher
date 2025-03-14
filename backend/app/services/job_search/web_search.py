"""
Module tìm kiếm việc làm trên web
"""

import os
import re
import time
import logging
import tempfile
from typing import Dict, Any, List, Optional, Tuple, Union
from urllib.parse import urlparse, urljoin
import json
import hashlib
from datetime import datetime, timedelta

from app.core.logging import get_logger
from app.core.settings import settings
from app.models.job import JobData, CompanyInfo, Location, SalaryRange, JobRequirement, JobType, ExperienceLevel, \
    JobSource, Benefit

logger = get_logger("job_search")


class WebSearcher:
    """
    Lớp tìm kiếm việc làm trên web
    """

    def __init__(self):
        """
        Khởi tạo
        """
        self.cache_dir = os.path.join(settings.TEMP_DIR, "web_search_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Danh sách các trang việc làm hỗ trợ
        self.supported_sites = {
            "linkedin.com": self._parse_linkedin,
            "topcv.vn": self._parse_topcv,
            "vietnamworks.com": self._parse_vietnamworks,
            "careerbuilder.vn": self._parse_careerbuilder,
            "careerviet.vn": self._parse_careerviet,
            "itviec.com": self._parse_itviec,
            "timviecnhanh.com": self._parse_timviecnhanh,
            "vieclam24h.vn": self._parse_vieclam24h,
            "mywork.com.vn": self._parse_mywork
        }

        # Từ điển chuyển đổi trường
        self.field_mappings = {
            "position": ["job", "title", "position", "role", "chức danh", "vị trí"],
            "company": ["company", "employer", "organization", "công ty", "tổ chức", "doanh nghiệp"],
            "location": ["location", "address", "workplace", "địa điểm", "địa chỉ", "nơi làm việc"],
            "salary": ["salary", "wage", "compensation", "lương", "thu nhập", "mức lương"],
            "experience": ["experience", "exp", "kinh nghiệm", "yêu cầu kinh nghiệm"],
            "education": ["education", "degree", "qualification", "học vấn", "bằng cấp", "trình độ"],
            "skills": ["skills", "requirements", "kỹ năng", "yêu cầu", "technical skills", "kỹ năng chuyên môn"],
            "benefits": ["benefits", "perks", "welfare", "phúc lợi", "chế độ", "đãi ngộ"],
            "job_type": ["job type", "employment type", "loại công việc", "hình thức làm việc"],
            "deadline": ["deadline", "due date", "closing date", "hạn nộp", "hạn ứng tuyển"]
        }

    def search(self, query: Dict[str, Any], max_results: int = 20) -> List[JobData]:
        """
        Tìm kiếm việc làm

        Args:
            query: Câu truy vấn tìm kiếm
            max_results: Số lượng kết quả tối đa

        Returns:
            List[JobData]: Danh sách các công việc tìm được
        """
        try:
            # Tạo chuỗi tìm kiếm từ query
            search_query = self._build_search_query(query)

            # Kiểm tra cache
            cached_results = self._get_from_cache(search_query)
            if cached_results:
                logger.info(f"Đã tìm thấy kết quả trong cache cho query: {search_query}")
                return cached_results

            # Thực hiện tìm kiếm trên web
            search_results = self._search_on_web(search_query, max_results)

            # Phân tích kết quả
            job_results = []

            for result in search_results:
                try:
                    # Lấy thông tin từ URL
                    job_data = self._extract_job_from_url(result["url"], result.get("title", ""),
                                                          result.get("snippet", ""))

                    if job_data:
                        job_results.append(job_data)

                        # Nếu đã đủ số lượng kết quả, dừng
                        if len(job_results) >= max_results:
                            break
                except Exception as e:
                    logger.error(f"Lỗi khi phân tích công việc từ {result['url']}: {str(e)}")

            # Lưu kết quả vào cache
            if job_results:
                self._save_to_cache(search_query, job_results)

            return job_results

        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm việc làm: {str(e)}")
            return []

    def _build_search_query(self, query: Dict[str, Any]) -> str:
        """
        Xây dựng chuỗi tìm kiếm từ query

        Args:
            query: Câu truy vấn tìm kiếm

        Returns:
            str: Chuỗi tìm kiếm
        """
        search_parts = []

        # Thêm tiêu đề công việc
        if "job_title" in query and query["job_title"]:
            search_parts.append(query["job_title"])

        # Thêm kỹ năng
        if "skills" in query and query["skills"]:
            # Chỉ lấy tối đa 3 kỹ năng để tránh quá cụ thể
            skills = query["skills"][:3]
            search_parts.extend(skills)

        # Thêm địa điểm
        if "location" in query and query["location"]:
            search_parts.append(query["location"])

        # Thêm kinh nghiệm
        if "years_of_experience" in query and query["years_of_experience"]:
            exp_years = query["years_of_experience"]
            if isinstance(exp_years, int) and exp_years > 0:
                search_parts.append(f"{exp_years} năm kinh nghiệm")

        # Kết hợp thành chuỗi tìm kiếm
        search_query = " ".join(search_parts)

        # Thêm từ khóa tuyển dụng
        search_query = f"{search_query} tuyển dụng việc làm"

        return search_query

    def _search_on_web(self, query: str, max_results: int = 20) -> List[Dict[str, str]]:
        """
        Thực hiện tìm kiếm trên web

        Args:
            query: Chuỗi tìm kiếm
            max_results: Số lượng kết quả tối đa

        Returns:
            List[Dict[str, str]]: Danh sách kết quả tìm kiếm
        """
        try:
            from duckduckgo_search import DDGS

            # Bổ sung các trang tuyển dụng ưu tiên
            for site in settings.JOB_SITES:
                # Chỉ thêm site: vào query nếu trang đó chưa có trong query
                if site not in query:
                    query = f"{query} site:{site}"

            # Thực hiện tìm kiếm
            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=max_results))

            # Chuẩn hóa kết quả
            normalized_results = []

            for result in results:
                normalized_result = {
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", "")
                }
                normalized_results.append(normalized_result)

            return normalized_results

        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm trên web: {str(e)}")

            # Thử phương pháp khác nếu DuckDuckGo không hoạt động
            return self._search_with_serpapi(query, max_results)

    def _search_with_serpapi(self, query: str, max_results: int = 20) -> List[Dict[str, str]]:
        """
        Thực hiện tìm kiếm với SerpAPI

        Args:
            query: Chuỗi tìm kiếm
            max_results: Số lượng kết quả tối đa

        Returns:
            List[Dict[str, str]]: Danh sách kết quả tìm kiếm
        """
        try:
            import requests

            # SerpAPI cần API key, thử sử dụng requests trực tiếp để tìm kiếm
            # Đây chỉ là giải pháp dự phòng, không đảm bảo hoạt động ổn định

            # Thực hiện tìm kiếm trên Google
            url = "https://www.google.com/search"
            params = {
                "q": query,
                "num": max_results
            }

            headers = {
                "User-Agent": settings.USER_AGENT
            }

            response = requests.get(url, params=params, headers=headers)

            if response.status_code != 200:
                logger.error(f"Không thể tìm kiếm với Google: {response.status_code}")
                return []

            # Phân tích kết quả HTML (cần BeautifulSoup)
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.text, "html.parser")
            search_results = []

            # Tìm các kết quả tìm kiếm
            for result in soup.select(".g"):
                title_element = result.select_one("h3")
                link_element = result.select_one("a")
                snippet_element = result.select_one(".VwiC3b")

                if title_element and link_element and "href" in link_element.attrs:
                    title = title_element.get_text()
                    url = link_element["href"]
                    snippet = snippet_element.get_text() if snippet_element else ""

                    # Kiểm tra URL hợp lệ
                    if url.startswith("http"):
                        search_results.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet
                        })

            return search_results[:max_results]

        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm với phương pháp dự phòng: {str(e)}")
            return []

    def _extract_job_from_url(self, url: str, title: str = "", snippet: str = "") -> Optional[JobData]:
        """
        Trích xuất thông tin công việc từ URL

        Args:
            url: URL của công việc
            title: Tiêu đề kết quả tìm kiếm
            snippet: Đoạn trích kết quả tìm kiếm

        Returns:
            Optional[JobData]: Thông tin công việc
        """
        try:
            # Xác định tên miền
            domain = urlparse(url).netloc

            # Tìm hàm phân tích phù hợp
            parser_func = None

            for site, parser in self.supported_sites.items():
                if site in domain:
                    parser_func = parser
                    break

            if parser_func:
                # Truy cập URL để lấy nội dung
                page_content = self._fetch_url(url)

                if page_content:
                    # Phân tích nội dung
                    return parser_func(url, page_content, title, snippet)
                else:
                    logger.warning(f"Không thể truy cập URL: {url}")
            else:
                # Trang không được hỗ trợ trực tiếp, sử dụng phân tích chung
                page_content = self._fetch_url(url)

                if page_content:
                    return self._parse_generic(url, page_content, title, snippet, domain)
                else:
                    logger.warning(f"Không thể truy cập URL: {url}")

            return None

        except Exception as e:
            logger.error(f"Lỗi khi trích xuất công việc từ URL {url}: {str(e)}")
            return None

    def _fetch_url(self, url: str) -> Optional[str]:
        """
        Truy cập URL để lấy nội dung

        Args:
            url: URL cần truy cập

        Returns:
            Optional[str]: Nội dung trang web
        """
        try:
            import requests
            from bs4 import BeautifulSoup

            # Gửi yêu cầu HTTP
            headers = {
                "User-Agent": settings.USER_AGENT
            }

            response = requests.get(url, headers=headers, timeout=settings.SEARCH_TIMEOUT)

            if response.status_code != 200:
                logger.warning(f"Không thể truy cập URL: {url}, mã trạng thái: {response.status_code}")
                return None

            # Phân tích HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Loại bỏ script và style
            for script in soup(["script", "style"]):
                script.extract()

            # Lấy văn bản
            return soup.get_text(separator="\n")

        except Exception as e:
            logger.error(f"Lỗi khi truy cập URL {url}: {str(e)}")
            return None

    def _parse_linkedin(self, url: str, content: str, title: str, snippet: str) -> Optional[JobData]:
        """
        Phân tích nội dung từ LinkedIn

        Args:
            url: URL công việc
            content: Nội dung trang
            title: Tiêu đề kết quả tìm kiếm
            snippet: Đoạn trích kết quả tìm kiếm

        Returns:
            Optional[JobData]: Thông tin công việc
        """
        # Phân tích tiêu đề và công ty
        job_title = title.split(" | ")[0] if " | " in title else title
        company_name = title.split(" | ")[1].split(" - ")[0] if " | " in title and " - " in title.split(" | ")[
            1] else ""

        # Phân tích địa điểm
        location_text = title.split(" - ")[-1] if " - " in title else ""

        # Phân tích mức lương
        salary_match = re.search(r'(\d[\d\s.,]*\s*(triệu|tr|million|M))', content, re.IGNORECASE)
        salary_text = salary_match.group(1) if salary_match else ""

        # Tạo đối tượng công việc
        job = JobData(
            title=job_title,
            description=snippet,
            company=CompanyInfo(name=company_name),
            location=Location(
                city=location_text,
                country="Việt Nam"
            ),
            job_type=JobType.FULL_TIME,
            requirements=JobRequirement(skills=[]),
            source=JobSource.LINKEDIN,
            source_url=url,
            is_active=True
        )

        # Thêm mức lương nếu có
        if salary_text:
            job.salary = self._parse_salary(salary_text)

        return job

    def _parse_topcv(self, url: str, content: str, title: str, snippet: str) -> Optional[JobData]:
        """
        Phân tích nội dung từ TopCV

        Args:
            url: URL công việc
            content: Nội dung trang
            title: Tiêu đề kết quả tìm kiếm
            snippet: Đoạn trích kết quả tìm kiếm

        Returns:
            Optional[JobData]: Thông tin công việc
        """
        # Phân tích tiêu đề và công ty
        parts = title.split(" - ")
        job_title = parts[0] if len(parts) > 0 else title
        company_name = parts[1] if len(parts) > 1 else ""

        # Phân tích địa điểm
        location_match = re.search(r'Địa điểm:\s*(.+?)[\n\.]', content)
        location_text = location_match.group(1) if location_match else ""

        # Phân tích mức lương
        salary_match = re.search(r'Mức lương:\s*(.+?)[\n\.]', content)
        salary_text = salary_match.group(1) if salary_match else ""

        # Phân tích kinh nghiệm
        exp_match = re.search(r'Kinh nghiệm:\s*(.+?)[\n\.]', content)
        exp_text = exp_match.group(1) if exp_match else ""

        # Phân tích kỹ năng
        skills = []
        skills_match = re.search(r'Kỹ năng:\s*(.+?)[\n\.]', content)
        if skills_match:
            skills_text = skills_match.group(1)
            skills = [skill.strip() for skill in skills_text.split(",")]

        # Tạo đối tượng công việc
        job = JobData(
            title=job_title,
            description=snippet,
            company=CompanyInfo(name=company_name),
            location=Location(
                city=location_text,
                country="Việt Nam"
            ),
            job_type=JobType.FULL_TIME,
            requirements=JobRequirement(skills=skills),
            source=JobSource.TOPCV,
            source_url=url,
            is_active=True
        )

        # Thêm mức lương nếu có
        if salary_text:
            job.salary = self._parse_salary(salary_text)

        # Thêm kinh nghiệm nếu có
        if exp_text:
            job.experience_level = self._parse_experience(exp_text)

        return job

    def _parse_vietnamworks(self, url: str, content: str, title: str, snippet: str) -> Optional[JobData]:
        """
        Phân tích nội dung từ VietnamWorks

        Args:
            url: URL công việc
            content: Nội dung trang
            title: Tiêu đề kết quả tìm kiếm
            snippet: Đoạn trích kết quả tìm kiếm

        Returns:
            Optional[JobData]: Thông tin công việc
        """
        # Phân tích tiêu đề và công ty
        parts = title.split(" - ")
        job_title = parts[0] if len(parts) > 0 else title
        company_name = parts[1] if len(parts) > 1 else ""

        # Phân tích địa điểm
        location_match = re.search(r'Location:\s*(.+?)[\n\.]|Địa điểm:\s*(.+?)[\n\.]', content)
        location_text = location_match.group(1) or location_match.group(2) if location_match else ""

        # Phân tích mức lương
        salary_match = re.search(r'Salary:\s*(.+?)[\n\.]|Mức lương:\s*(.+?)[\n\.]', content)
        salary_text = salary_match.group(1) or salary_match.group(2) if salary_match else ""

        # Phân tích kỹ năng
        skills = []
        skills_section = re.search(r'(Skills|Kỹ năng):(.*?)(Mô tả công việc|Job Description)', content,
                                   re.DOTALL | re.IGNORECASE)
        if skills_section:
            skills_text = skills_section.group(2)
            # Phân tách kỹ năng theo dấu phẩy hoặc dấu bullet
            skills_raw = re.split(r',|\n', skills_text)
            for skill in skills_raw:
                skill = skill.strip()
                if skill and len(skill) > 2:  # Loại bỏ các chuỗi quá ngắn
                    skills.append(skill)

        # Phân tích phúc lợi
        benefits = []
        benefits_section = re.search(r'(Benefits|Phúc lợi):(.*?)(Yêu cầu công việc|Job Requirement)', content,
                                     re.DOTALL | re.IGNORECASE)
        if benefits_section:
            benefits_text = benefits_section.group(2)
            # Phân tách phúc lợi theo dấu phẩy hoặc dấu bullet
            benefits_raw = re.split(r',|\n', benefits_text)
            for benefit in benefits_raw:
                benefit = benefit.strip()
                if benefit and len(benefit) > 3:  # Loại bỏ các chuỗi quá ngắn
                    benefits.append(Benefit(name=benefit))

        # Tạo đối tượng công việc
        job = JobData(
            title=job_title,
            description=snippet,
            company=CompanyInfo(name=company_name),
            location=Location(
                city=location_text,
                country="Việt Nam"
            ),
            job_type=JobType.FULL_TIME,
            requirements=JobRequirement(skills=skills),
            benefits=benefits,
            source=JobSource.VIETNAMWORKS,
            source_url=url,
            is_active=True
        )

        # Thêm mức lương nếu có
        if salary_text:
            job.salary = self._parse_salary(salary_text)

        return job

    def _parse_careerbuilder(self, url: str, content: str, title: str, snippet: str) -> Optional[JobData]:
        """
        Phân tích nội dung từ CareerBuilder

        Args:
            url: URL công việc
            content: Nội dung trang
            title: Tiêu đề kết quả tìm kiếm
            snippet: Đoạn trích kết quả tìm kiếm

        Returns:
            Optional[JobData]: Thông tin công việc
        """
        # Phân tích tiêu đề và công ty
        parts = title.split(" - ")
        job_title = parts[0] if len(parts) > 0 else title
        company_name = parts[1] if len(parts) > 1 else ""

        # Phân tích địa điểm
        location_match = re.search(r'Nơi làm việc:\s*(.+?)[\n\.]|Địa điểm:\s*(.+?)[\n\.]', content)
        location_text = location_match.group(1) or location_match.group(2) if location_match else ""

        # Phân tích mức lương
        salary_match = re.search(r'Lương:\s*(.+?)[\n\.]|Mức lương:\s*(.+?)[\n\.]', content)
        salary_text = salary_match.group(1) or salary_match.group(2) if salary_match else ""

        # Phân tích kinh nghiệm
        exp_match = re.search(r'Kinh nghiệm:\s*(.+?)[\n\.]', content)
        exp_text = exp_match.group(1) if exp_match else ""

        # Phân tích hạn nộp hồ sơ
        deadline_match = re.search(r'Hạn nộp hồ sơ:\s*(\d{1,2}/\d{1,2}/\d{4})', content)
        deadline_text = deadline_match.group(1) if deadline_match else None

        # Chuyển đổi deadline thành datetime
        deadline = None
        if deadline_text:
            try:
                deadline = datetime.strptime(deadline_text, "%d/%m/%Y")
            except ValueError:
                pass

        # Tạo đối tượng công việc
        job = JobData(
            title=job_title,
            description=snippet,
            company=CompanyInfo(name=company_name),
            location=Location(
                city=location_text,
                country="Việt Nam"
            ),
            job_type=JobType.FULL_TIME,
            requirements=JobRequirement(skills=[]),
            source=JobSource.CAREERBUILDER,
            source_url=url,
            deadline=deadline,
            is_active=True
        )

        # Thêm mức lương nếu có
        if salary_text:
            job.salary = self._parse_salary(salary_text)

        # Thêm kinh nghiệm nếu có
        if exp_text:
            job.experience_level = self._parse_experience(exp_text)

        return job

    def _parse_careerviet(self, url: str, content: str, title: str, snippet: str) -> Optional[JobData]:
        """
        Phân tích nội dung từ CareerViet

        Args:
            url: URL công việc
            content: Nội dung trang
            title: Tiêu đề kết quả tìm kiếm
            snippet: Đoạn trích kết quả tìm kiếm

        Returns:
            Optional[JobData]: Thông tin công việc
        """
        # Dùng giống cấu trúc của CareerBuilder
        return self._parse_careerbuilder(url, content, title, snippet)

    def _parse_itviec(self, url: str, content: str, title: str, snippet: str) -> Optional[JobData]:
        """
        Phân tích nội dung từ ITViec

        Args:
            url: URL công việc
            content: Nội dung trang
            title: Tiêu đề kết quả tìm kiếm
            snippet: Đoạn trích kết quả tìm kiếm

        Returns:
            Optional[JobData]: Thông tin công việc
        """
        # Phân tích tiêu đề và công ty
        parts = title.split(" - ")
        job_title = parts[0] if len(parts) > 0 else title
        company_name = parts[1] if len(parts) > 1 else ""

        # Phân tích địa điểm
        location_match = re.search(r'Địa điểm:\s*(.+?)[\n\.]|Location:\s*(.+?)[\n\.]', content)
        location_text = location_match.group(1) or location_match.group(2) if location_match else ""

        # Phân tích mức lương
        salary_text = ""
        for pattern in [r'Lương:\s*(.+?)[\n\.]', r'Salary:\s*(.+?)[\n\.]', r'(Up to \$[\d,]+)',
                        r'(\$[\d,]+ - \$[\d,]+)']:
            salary_match = re.search(pattern, content)
            if salary_match:
                salary_text = salary_match.group(1)
                break

        # Phân tích kỹ năng
        skills = []
        skill_patterns = [
            r'Kỹ năng:\s*(.+?)[\n\.]',
            r'Skills:\s*(.+?)[\n\.]',
            r'Tech stack:(.*?)(?=Benefits|Phúc lợi)',
            r'Tech stack và công cụ:(.*?)(?=Benefits|Phúc lợi)'
        ]

        for pattern in skill_patterns:
            skills_match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if skills_match:
                skills_text = skills_match.group(1)
                # Phân tách kỹ năng
                skills_raw = re.split(r',|\n', skills_text)
                for skill in skills_raw:
                    skill = skill.strip()
                    if skill and len(skill) > 2:
                        skills.append(skill)
                break

        # Tạo đối tượng công việc
        job = JobData(
            title=job_title,
            description=snippet,
            company=CompanyInfo(name=company_name),
            location=Location(
                city=location_text,
                country="Việt Nam"
            ),
            job_type=JobType.FULL_TIME,
            requirements=JobRequirement(skills=skills),
            source=JobSource.OTHER,
            source_url=url,
            is_active=True
        )

        # Thêm mức lương nếu có
        if salary_text:
            job.salary = self._parse_salary(salary_text)

        return job

    def _parse_timviecnhanh(self, url: str, content: str, title: str, snippet: str) -> Optional[JobData]:
        """
        Phân tích nội dung từ TimViecNhanh

        Args:
            url: URL công việc
            content: Nội dung trang
            title: Tiêu đề kết quả tìm kiếm
            snippet: Đoạn trích kết quả tìm kiếm

        Returns:
            Optional[JobData]: Thông tin công việc
        """
        # Phân tích tương tự các trang khác
        return self._parse_generic(url, content, title, snippet, "timviecnhanh.com")

    def _parse_vieclam24h(self, url: str, content: str, title: str, snippet: str) -> Optional[JobData]:
        """
        Phân tích nội dung từ ViecLam24h

        Args:
            url: URL công việc
            content: Nội dung trang
            title: Tiêu đề kết quả tìm kiếm
            snippet: Đoạn trích kết quả tìm kiếm

        Returns:
            Optional[JobData]: Thông tin công việc
        """
        # Phân tích tương tự các trang khác
        return self._parse_generic(url, content, title, snippet, "vieclam24h.vn")

    def _parse_mywork(self, url: str, content: str, title: str, snippet: str) -> Optional[JobData]:
        """
        Phân tích nội dung từ MyWork

        Args:
            url: URL công việc
            content: Nội dung trang
            title: Tiêu đề kết quả tìm kiếm
            snippet: Đoạn trích kết quả tìm kiếm

        Returns:
            Optional[JobData]: Thông tin công việc
        """
        # Phân tích tương tự các trang khác
        return self._parse_generic(url, content, title, snippet, "mywork.com.vn")

    def _parse_generic(self, url: str, content: str, title: str, snippet: str, domain: str) -> Optional[JobData]:
        """
        Phân tích nội dung từ trang web không được hỗ trợ cụ thể

        Args:
            url: URL công việc
            content: Nội dung trang
            title: Tiêu đề kết quả tìm kiếm
            snippet: Đoạn trích kết quả tìm kiếm
            domain: Tên miền trang web

        Returns:
            Optional[JobData]: Thông tin công việc
        """
        # Phân tích tiêu đề và công ty
        parts = title.split(" - ")
        job_title = parts[0] if len(parts) > 0 else title
        company_name = parts[1] if len(parts) > 1 else ""

        # Phân tích địa điểm
        location_patterns = [
            r'Địa điểm:?\s*(.+?)[\n\.]',
            r'Nơi làm việc:?\s*(.+?)[\n\.]',
            r'Location:?\s*(.+?)[\n\.]',
            r'Workplace:?\s*(.+?)[\n\.]'
        ]

        location_text = ""
        for pattern in location_patterns:
            location_match = re.search(pattern, content)
            if location_match:
                location_text = location_match.group(1).strip()
                break

        # Nếu không tìm thấy địa điểm theo pattern, tìm các thành phố lớn trong nội dung
        if not location_text:
            cities = ["Hà Nội", "TP HCM", "TP. HCM", "Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ", "Biên Hòa",
                      "Nha Trang"]
            for city in cities:
                if city in content:
                    location_text = city
                    break

        # Phân tích mức lương
        salary_patterns = [
            r'Lương:?\s*(.+?)[\n\.]',
            r'Mức lương:?\s*(.+?)[\n\.]',
            r'Salary:?\s*(.+?)[\n\.]',
            r'([\d,\.]+\s*-\s*[\d,\.]+\s*(triệu|tr|million|M))',
            r'(Lên đến|Up to) (\d[\d\s.,]*\s*(triệu|tr|million|M))'
        ]

        salary_text = ""
        for pattern in salary_patterns:
            salary_match = re.search(pattern, content)
            if salary_match:
                salary_text = salary_match.group(1).strip()
                break

        # Phân tích kinh nghiệm
        exp_patterns = [
            r'Kinh nghiệm:?\s*(.+?)[\n\.]',
            r'Yêu cầu kinh nghiệm:?\s*(.+?)[\n\.]',
            r'Experience:?\s*(.+?)[\n\.]'
        ]

        exp_text = ""
        for pattern in exp_patterns:
            exp_match = re.search(pattern, content)
            if exp_match:
                exp_text = exp_match.group(1).strip()
                break

        # Phân tích kỹ năng bằng cách tìm các đoạn văn bản phù hợp
        skills = []
        skill_sections = [
            r'Kỹ năng:?\s*(.*?)(?=Phúc lợi|Quyền lợi|Yêu cầu)',
            r'Skills:?\s*(.*?)(?=Benefits|Requirements)',
            r'Yêu cầu:?\s*(.*?)(?=Phúc lợi|Quyền lợi|Mô tả)',
            r'Requirements:?\s*(.*?)(?=Benefits|Description)'
        ]

        for pattern in skill_sections:
            skills_match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if skills_match:
                skills_text = skills_match.group(1)
                # Phân tách kỹ năng
                skills_raw = re.split(r',|\n|•|\*|-|\.', skills_text)
                for skill in skills_raw:
                    skill = skill.strip()
                    if skill and len(skill) > 2 and not any(
                            common in skill.lower() for common in ["yêu cầu", "requirement", "looking for"]):
                        skills.append(skill)
                break

        # Xác định job source
        job_source = JobSource.OTHER
        if "linkedin.com" in domain:
            job_source = JobSource.LINKEDIN
        elif "topcv.vn" in domain:
            job_source = JobSource.TOPCV
        elif "vietnamworks.com" in domain:
            job_source = JobSource.VIETNAMWORKS
        elif "careerbuilder.vn" in domain:
            job_source = JobSource.CAREERBUILDER
        elif "careerviet.vn" in domain:
            job_source = JobSource.CAREERVIET

        # Tạo đối tượng công việc
        job = JobData(
            title=job_title,
            description=snippet or content[:500],  # Sử dụng đoạn trích hoặc 500 ký tự đầu tiên của nội dung
            company=CompanyInfo(name=company_name),
            location=Location(
                city=location_text,
                country="Việt Nam"
            ),
            job_type=JobType.FULL_TIME,
            requirements=JobRequirement(skills=skills),
            source=job_source,
            source_url=url,
            is_active=True,
            raw_text=content[:2000]  # Lưu 2000 ký tự đầu tiên của nội dung gốc
        )

        # Thêm mức lương nếu có
        if salary_text:
            job.salary = self._parse_salary(salary_text)

        # Thêm kinh nghiệm nếu có
        if exp_text:
            job.experience_level = self._parse_experience(exp_text)

        return job

    def _parse_salary(self, salary_text: str) -> SalaryRange:
        """
        Phân tích chuỗi lương thành đối tượng SalaryRange

        Args:
            salary_text: Chuỗi lương

        Returns:
            SalaryRange: Đối tượng phạm vi lương
        """
        salary = SalaryRange(
            is_disclosed=True,
            is_negotiable=False,
            currency="VND",
            period="monthly"
        )

        # Kiểm tra lương thỏa thuận hoặc cạnh tranh
        if re.search(r'(thỏa thuận|cạnh tranh|thoa thuan|canh tranh|trao đổi|trao doi|negotiable|competitive)',
                     salary_text, re.IGNORECASE):
            salary.is_disclosed = False
            salary.is_negotiable = True
            return salary

        # Xử lý lương USD
        if "$" in salary_text:
            salary.currency = "USD"
            # Phân tích phạm vi lương USD
            if "-" in salary_text:
                # Dạng $X - $Y
                match = re.search(r'\$\s*([\d,\.]+)\s*-\s*\$\s*([\d,\.]+)', salary_text)
                if match:
                    min_val = float(match.group(1).replace(",", "").replace(".", ""))
                    max_val = float(match.group(2).replace(",", "").replace(".", ""))
                    salary.min = min_val
                    salary.max = max_val
            else:
                # Dạng $X hoặc Up to $X
                match = re.search(r'\$\s*([\d,\.]+)', salary_text)
                if match:
                    val = float(match.group(1).replace(",", "").replace(".", ""))

                    if "up to" in salary_text.lower():
                        salary.max = val
                    else:
                        salary.min = val

        # Xử lý lương VND
        else:
            # Phân tích phạm vi lương VND
            match = re.search(r'([\d\.,]+)\s*-\s*([\d\.,]+)\s*(tr|triệu|trieu|m|million)', salary_text, re.IGNORECASE)
            if match:
                # Dạng X - Y triệu
                min_val = float(match.group(1).replace(".", "").replace(",", "."))
                max_val = float(match.group(2).replace(".", "").replace(",", "."))

                # Đơn vị triệu
                salary.min = min_val * 1000000
                salary.max = max_val * 1000000
            else:
                # Dạng X triệu hoặc Lên đến X triệu
                match = re.search(r'([\d\.,]+)\s*(tr|triệu|trieu|m|million)', salary_text, re.IGNORECASE)
                if match:
                    val = float(match.group(1).replace(".", "").replace(",", "."))

                    # Đơn vị triệu
                    val = val * 1000000

                    if re.search(r'(lên đến|up to|đến|tối đa|toi da)', salary_text, re.IGNORECASE):
                        salary.max = val
                    else:
                        salary.min = val

        return salary

    def _parse_experience(self, exp_text: str) -> ExperienceLevel:
        """
        Phân tích chuỗi kinh nghiệm thành đối tượng ExperienceLevel

        Args:
            exp_text: Chuỗi kinh nghiệm

        Returns:
            ExperienceLevel: Cấp độ kinh nghiệm
        """
        exp_text = exp_text.lower()

        # Kiểm tra fresh graduate / không yêu cầu kinh nghiệm
        if re.search(r'(không yêu cầu|khong yeu cau|fresh|mới tốt nghiệp|new graduate|no experience|0 năm|0 nam)',
                     exp_text):
            return ExperienceLevel.ENTRY

        # Tìm số năm kinh nghiệm
        year_match = re.search(r'(\d+)\s*(năm|nam|year)', exp_text)
        if year_match:
            years = int(year_match.group(1))

            if years < 1:
                return ExperienceLevel.ENTRY
            elif years < 3:
                return ExperienceLevel.JUNIOR
            elif years < 5:
                return ExperienceLevel.MID
            elif years < 8:
                return ExperienceLevel.SENIOR
            else:
                return ExperienceLevel.MANAGER

        # Phân loại theo từ khóa
        if re.search(r'(junior|fresher|entry|sơ cấp|so cap)', exp_text):
            return ExperienceLevel.JUNIOR
        elif re.search(r'(middle|mid|intermediate|trung cấp|trung cap)', exp_text):
            return ExperienceLevel.MID
        elif re.search(r'(senior|lead|cao cấp|cao cap)', exp_text):
            return ExperienceLevel.SENIOR
        elif re.search(r'(manager|quản lý|quan ly|giám đốc|giam doc)', exp_text):
            return ExperienceLevel.MANAGER
        elif re.search(r'(director|cto|vp|phó chủ tịch|pho chu tich)', exp_text):
            return ExperienceLevel.DIRECTOR

        # Mặc định
        return ExperienceLevel.MID

    def _get_from_cache(self, query: str) -> Optional[List[JobData]]:
        """
        Lấy kết quả từ cache

        Args:
            query: Chuỗi tìm kiếm

        Returns:
            Optional[List[JobData]]: Danh sách công việc từ cache hoặc None nếu không có
        """
        # Tạo key cache từ query
        cache_key = self._get_cache_key(query)
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

                    # Chuyển đổi thành đối tượng JobData
                    jobs = []
                    for job_dict in cache_data:
                        try:
                            job = JobData.model_validate(job_dict)
                            jobs.append(job)
                        except Exception as e:
                            """