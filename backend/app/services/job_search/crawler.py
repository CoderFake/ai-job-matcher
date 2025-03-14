"""
Module thu thập thông tin chi tiết từ các trang việc làm
"""

import os
import re
import time
import logging
import tempfile
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from urllib.parse import urlparse, urljoin
import json
import hashlib
from datetime import datetime, timedelta
import random
import asyncio
from pathlib import Path

from app.core.logging import get_logger
from app.core.settings import settings
from app.models.job import JobData, CompanyInfo, Location, SalaryRange

logger = get_logger("job_search")


class WebCrawler:
    """
    Lớp thu thập thông tin chi tiết từ các trang việc làm
    """

    def __init__(self):
        """
        Khởi tạo crawler
        """
        self.cache_dir = os.path.join(settings.TEMP_DIR, "crawler_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Danh sách các trang hỗ trợ
        self.supported_domains = {
            "linkedin.com": self._crawl_linkedin,
            "topcv.vn": self._crawl_topcv,
            "vietnamworks.com": self._crawl_vietnamworks,
            "careerbuilder.vn": self._crawl_careerbuilder,
            "careerviet.vn": self._crawl_careerviet,
            "itviec.com": self._crawl_itviec
        }

        # Thiết lập thời gian chờ giữa các yêu cầu
        self.delay = 2  # giây

        # Danh sách User-Agent để thay đổi
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59"
        ]

        # Khởi tạo session
        self.last_request_time = {}

    async def crawl_jobs(self, urls: List[str]) -> List[JobData]:
        """
        Thu thập thông tin chi tiết từ danh sách URL việc làm

        Args:
            urls: Danh sách URL cần thu thập

        Returns:
            List[JobData]: Danh sách thông tin việc làm
        """
        jobs = []
        tasks = []

        # Tạo các task để crawl bất đồng bộ
        for url in urls:
            task = asyncio.create_task(self.crawl_job(url))
            tasks.append(task)

        # Chờ tất cả các task hoàn thành
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Lỗi khi crawl: {str(result)}")
            elif result:
                jobs.append(result)

        return jobs

    async def crawl_job(self, url: str) -> Optional[JobData]:
        """
        Thu thập thông tin chi tiết từ URL việc làm

        Args:
            url: URL cần thu thập

        Returns:
            Optional[JobData]: Thông tin việc làm
        """
        # Kiểm tra cache
        cached_job = self._get_from_cache(url)
        if cached_job:
            logger.info(f"Đã tìm thấy thông tin trong cache cho URL: {url}")
            return cached_job

        try:
            # Xác định tên miền
            domain = urlparse(url).netloc

            # Tìm hàm crawl phù hợp
            crawler_func = None
            for supported_domain, func in self.supported_domains.items():
                if supported_domain in domain:
                    crawler_func = func
                    break

            # Sử dụng crawler chung nếu không có hàm crawl cụ thể
            if crawler_func is None:
                crawler_func = self._crawl_generic

            # Lấy HTML từ URL
            html = await self._fetch_url(url)
            if not html:
                logger.warning(f"Không thể lấy HTML từ URL: {url}")
                return None

            # Crawl thông tin
            job = await crawler_func(url, html)

            # Lưu vào cache nếu crawl thành công
            if job:
                self._save_to_cache(url, job)

            return job

        except Exception as e:
            logger.error(f"Lỗi khi crawl URL {url}: {str(e)}")
            return None

    async def _fetch_url(self, url: str) -> Optional[str]:
        """
        Lấy nội dung HTML từ URL

        Args:
            url: URL cần lấy

        Returns:
            Optional[str]: Nội dung HTML
        """
        try:
            # Kiểm tra và thực hiện delay để tránh bị chặn
            domain = urlparse(url).netloc
            if domain in self.last_request_time:
                time_since_last_request = time.time() - self.last_request_time[domain]
                if time_since_last_request < self.delay:
                    await asyncio.sleep(self.delay - time_since_last_request)

            # Chọn ngẫu nhiên User-Agent
            user_agent = random.choice(self.user_agents)

            # Thực hiện yêu cầu HTTP
            import aiohttp

            headers = {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=30) as response:
                    # Cập nhật thời gian yêu cầu cuối cùng
                    self.last_request_time[domain] = time.time()

                    if response.status != 200:
                        logger.warning(f"Không thể truy cập URL: {url}, mã trạng thái: {response.status}")
                        return None

                    # Lấy nội dung HTML
                    content_type = response.headers.get("Content-Type", "")

                    # Kiểm tra loại nội dung
                    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                        logger.warning(f"URL không trả về HTML: {url}, Content-Type: {content_type}")
                        return None

                    return await response.text()

        except Exception as e:
            logger.error(f"Lỗi khi lấy HTML từ URL {url}: {str(e)}")
            return None

    async def _crawl_linkedin(self, url: str, html: str) -> Optional[JobData]:
        """
        Crawl thông tin từ LinkedIn

        Args:
            url: URL việc làm
            html: Nội dung HTML

        Returns:
            Optional[JobData]: Thông tin việc làm
        """
        try:
            # Sử dụng BeautifulSoup để phân tích HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # Trích xuất tiêu đề công việc
            title_element = soup.select_one(".top-card-layout__title")
            title = title_element.text.strip() if title_element else ""

            # Trích xuất tên công ty
            company_element = soup.select_one(".topcard__org-name-link")
            company_name = company_element.text.strip() if company_element else ""

            # Trích xuất địa điểm
            location_element = soup.select_one(".topcard__flavor.topcard__flavor--bullet")
            location_text = location_element.text.strip() if location_element else ""

            # Trích xuất mô tả công việc
            description_element = soup.select_one(".description__text")
            description = description_element.text.strip() if description_element else ""

            # Trích xuất các thông tin khác
            criteria_elements = soup.select(".description__job-criteria-item")
            experience_level = ""
            job_type = ""

            for element in criteria_elements:
                header = element.select_one(".description__job-criteria-subheader")
                value = element.select_one(".description__job-criteria-text")

                if header and value:
                    header_text = header.text.strip()
                    value_text = value.text.strip()

                    if "Kinh nghiệm" in header_text or "Experience" in header_text:
                        experience_level = value_text
                    elif "Loại công việc" in header_text or "Employment type" in header_text:
                        job_type = value_text

            # Tạo đối tượng công việc
            from app.models.job import JobData, CompanyInfo, Location, JobType, ExperienceLevel, JobSource, \
                JobRequirement

            # Xác định loại công việc
            job_type_enum = JobType.FULL_TIME
            if job_type:
                if "part" in job_type.lower() or "bán thời gian" in job_type.lower():
                    job_type_enum = JobType.PART_TIME
                elif "contract" in job_type.lower() or "hợp đồng" in job_type.lower():
                    job_type_enum = JobType.CONTRACT
                elif "temporary" in job_type.lower() or "tạm thời" in job_type.lower():
                    job_type_enum = JobType.TEMPORARY
                elif "intern" in job_type.lower() or "thực tập" in job_type.lower():
                    job_type_enum = JobType.INTERNSHIP
                elif "freelance" in job_type.lower():
                    job_type_enum = JobType.FREELANCE

            # Xác định cấp độ kinh nghiệm
            experience_level_enum = None
            if experience_level:
                if "entry" in experience_level.lower() or "fresher" in experience_level.lower():
                    experience_level_enum = ExperienceLevel.ENTRY
                elif "mid" in experience_level.lower() or "trung cấp" in experience_level.lower():
                    experience_level_enum = ExperienceLevel.MID
                elif "senior" in experience_level.lower() or "cao cấp" in experience_level.lower():
                    experience_level_enum = ExperienceLevel.SENIOR
                elif "manager" in experience_level.lower() or "quản lý" in experience_level.lower():
                    experience_level_enum = ExperienceLevel.MANAGER
                elif "director" in experience_level.lower() or "giám đốc" in experience_level.lower():
                    experience_level_enum = ExperienceLevel.DIRECTOR

            # Trích xuất kỹ năng
            skills = []
            skills_section = soup.select_one(".skills-section")
            if skills_section:
                skill_elements = skills_section.select(".skill-pill")
                for skill_element in skill_elements:
                    skill = skill_element.text.strip()
                    if skill:
                        skills.append(skill)

            job = JobData(
                title=title,
                description=description,
                company=CompanyInfo(name=company_name),
                location=Location(
                    city=location_text,
                    country="Việt Nam" if "việt nam" in location_text.lower() else None
                ),
                job_type=job_type_enum,
                experience_level=experience_level_enum,
                requirements=JobRequirement(skills=skills),
                source=JobSource.LINKEDIN,
                source_url=url,
                is_active=True
            )

            return job

        except Exception as e:
            logger.error(f"Lỗi khi crawl LinkedIn {url}: {str(e)}")
            return None

    async def _crawl_topcv(self, url: str, html: str) -> Optional[JobData]:
        """
        Crawl thông tin từ TopCV

        Args:
            url: URL việc làm
            html: Nội dung HTML

        Returns:
            Optional[JobData]: Thông tin việc làm
        """
        try:
            # Sử dụng BeautifulSoup để phân tích HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # Trích xuất tiêu đề công việc
            title_element = soup.select_one("h1.job-title")
            title = title_element.text.strip() if title_element else ""

            # Trích xuất tên công ty
            company_element = soup.select_one(".company-name")
            company_name = company_element.text.strip() if company_element else ""

            # Trích xuất địa điểm
            location_element = soup.select_one(".address")
            location_text = location_element.text.strip() if location_element else ""

            # Trích xuất mô tả công việc
            description_element = soup.select_one(".job-description")
            description = description_element.text.strip() if description_element else ""

            # Trích xuất mức lương
            salary_element = soup.select_one(".salary")
            salary_text = salary_element.text.strip() if salary_element else ""

            # Trích xuất các thông tin khác
            job_info_elements = soup.select(".job-info .item")
            job_type = ""
            deadline = None

            for element in job_info_elements:
                label = element.select_one(".label")
                value = element.select_one(".value")

                if label and value:
                    label_text = label.text.strip()
                    value_text = value.text.strip()

                    if "Hình thức" in label_text:
                        job_type = value_text
                    elif "Hạn nộp" in label_text:
                        try:
                            deadline = datetime.strptime(value_text, "%d/%m/%Y")
                        except ValueError:
                            pass

            # Trích xuất kỹ năng
            skills = []
            skills_section = soup.select_one(".skills-list")
            if skills_section:
                skill_elements = skills_section.select(".skill-tag")
                for skill_element in skill_elements:
                    skill = skill_element.text.strip()
                    if skill:
                        skills.append(skill)

            # Trích xuất phúc lợi
            benefits = []
            benefits_section = soup.select_one(".benefit-list")
            if benefits_section:
                benefit_elements = benefits_section.select(".benefit-item")
                for benefit_element in benefit_elements:
                    benefit_text = benefit_element.text.strip()
                    if benefit_text:
                        from app.models.job import Benefit
                        benefits.append(Benefit(name=benefit_text))

            # Tạo đối tượng công việc
            from app.models.job import JobData, CompanyInfo, Location, JobType, JobSource, JobRequirement, SalaryRange

            # Xác định loại công việc
            job_type_enum = JobType.FULL_TIME
            if job_type:
                if "part" in job_type.lower() or "bán thời gian" in job_type.lower():
                    job_type_enum = JobType.PART_TIME
                elif "contract" in job_type.lower() or "hợp đồng" in job_type.lower():
                    job_type_enum = JobType.CONTRACT
                elif "temporary" in job_type.lower() or "tạm thời" in job_type.lower():
                    job_type_enum = JobType.TEMPORARY
                elif "intern" in job_type.lower() or "thực tập" in job_type.lower():
                    job_type_enum = JobType.INTERNSHIP
                elif "freelance" in job_type.lower():
                    job_type_enum = JobType.FREELANCE

            # Tạo đối tượng công việc
            job = JobData(
                title=title,
                description=description,
                company=CompanyInfo(name=company_name),
                location=Location(
                    city=location_text,
                    country="Việt Nam"
                ),
                job_type=job_type_enum,
                requirements=JobRequirement(skills=skills),
                benefits=benefits,
                source=JobSource.TOPCV,
                source_url=url,
                deadline=deadline,
                is_active=True
            )

            # Thêm mức lương nếu có
            if salary_text:
                from app.services.job_search.web_search import WebSearcher
                web_searcher = WebSearcher()
                job.salary = web_searcher._parse_salary(salary_text)

            return job

        except Exception as e:
            logger.error(f"Lỗi khi crawl TopCV {url}: {str(e)}")
            return None

    async def _crawl_vietnamworks(self, url: str, html: str) -> Optional[JobData]:
        """
        Crawl thông tin từ VietnamWorks

        Args:
            url: URL việc làm
            html: Nội dung HTML

        Returns:
            Optional[JobData]: Thông tin việc làm
        """
        try:
            # Sử dụng BeautifulSoup để phân tích HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # Trích xuất tiêu đề công việc
            title_element = soup.select_one("h1.job-title")
            title = title_element.text.strip() if title_element else ""

            # Trích xuất tên công ty
            company_element = soup.select_one(".company-name")
            company_name = company_element.text.strip() if company_element else ""

            # Trích xuất địa điểm
            location_element = soup.select_one(".location-name")
            location_text = location_element.text.strip() if location_element else ""

            # Trích xuất mô tả công việc
            description_element = soup.select_one(".job-description")
            description = description_element.text.strip() if description_element else ""

            # Trích xuất mức lương
            salary_element = soup.select_one(".salary-text")
            salary_text = salary_element.text.strip() if salary_element else ""

            # Trích xuất kỹ năng
            skills = []
            skills_elements = soup.select(".skill-tag")
            for skill_element in skills_elements:
                skill = skill_element.text.strip()
                if skill:
                    skills.append(skill)

            # Trích xuất phúc lợi
            benefits = []
            benefits_elements = soup.select(".benefits-list .benefit-item")
            for benefit_element in benefits_elements:
                benefit_text = benefit_element.text.strip()
                if benefit_text:
                    from app.models.job import Benefit
                    benefits.append(Benefit(name=benefit_text))

            # Tạo đối tượng công việc
            from app.models.job import JobData, CompanyInfo, Location, JobType, JobSource, JobRequirement

            job = JobData(
                title=title,
                description=description,
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
                from app.services.job_search.web_search import WebSearcher
                web_searcher = WebSearcher()
                job.salary = web_searcher._parse_salary(salary_text)

            return job

        except Exception as e:
            logger.error(f"Lỗi khi crawl VietnamWorks {url}: {str(e)}")
            return None

    async def _crawl_careerbuilder(self, url: str, html: str) -> Optional[JobData]:
        """
        Crawl thông tin từ CareerBuilder

        Args:
            url: URL việc làm
            html: Nội dung HTML

        Returns:
            Optional[JobData]: Thông tin việc làm
        """
        try:
            # Sử dụng BeautifulSoup để phân tích HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # Trích xuất tiêu đề công việc
            title_element = soup.select_one("h1.title")
            title = title_element.text.strip() if title_element else ""

            # Trích xuất tên công ty
            company_element = soup.select_one(".company-name")
            company_name = company_element.text.strip() if company_element else ""

            # Trích xuất địa điểm
            location_element = soup.select_one(".map")
            location_text = location_element.text.strip() if location_element else ""

            # Trích xuất mô tả công việc
            description_element = soup.select_one(".detail-content")
            description = description_element.text.strip() if description_element else ""

            # Trích xuất mức lương
            salary_element = soup.select_one(".salary")
            salary_text = salary_element.text.strip() if salary_element else ""

            # Trích xuất hạn nộp hồ sơ
            deadline = None
            deadline_element = soup.select_one(".deadline")
            if deadline_element:
                deadline_text = deadline_element.text.strip()
                match = re.search(r'(\d{2}/\d{2}/\d{4})', deadline_text)
                if match:
                    try:
                        deadline = datetime.strptime(match.group(1), "%d/%m/%Y")
                    except ValueError:
                        pass

            # Tạo đối tượng công việc
            from app.models.job import JobData, CompanyInfo, Location, JobType, JobSource, JobRequirement

            job = JobData(
                title=title,
                description=description,
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
                from app.services.job_search.web_search import WebSearcher
                web_searcher = WebSearcher()
                job.salary = web_searcher._parse_salary(salary_text)

            return job

        except Exception as e:
            logger.error(f"Lỗi khi crawl CareerBuilder {url}: {str(e)}")
            return None

    async def _crawl_careerviet(self, url: str, html: str) -> Optional[JobData]:
        """
        Crawl thông tin từ CareerViet

        Args:
            url: URL việc làm
            html: Nội dung HTML

        Returns:
            Optional[JobData]: Thông tin việc làm
        """
        # Sử dụng cấu trúc tương tự CareerBuilder
        return await self._crawl_careerbuilder(url, html)

    async def _crawl_itviec(self, url: str, html: str) -> Optional[JobData]:
        """
        Crawl thông tin từ ITViec

        Args:
            url: URL việc làm
            html: Nội dung HTML

        Returns:
            Optional[JobData]: Thông tin việc làm
        """
        try:
            # Sử dụng BeautifulSoup để phân tích HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # Trích xuất tiêu đề công việc
            title_element = soup.select_one("h1.job-title")
            title = title_element.text.strip() if title_element else ""

            # Trích xuất tên công ty
            company_element = soup.select_one(".company-name")
            company_name = company_element.text.strip() if company_element else ""

            # Trích xuất địa điểm
            location_element = soup.select_one(".location")
            location_text = location_element.text.strip() if location_element else ""

            # Trích xuất mô tả công việc
            description_element = soup.select_one(".job-description")
            description = description_element.text.strip() if description_element else ""

            # Trích xuất mức lương
            salary_element = soup.select_one(".salary-text")
            salary_text = salary_element.text.strip() if salary_element else ""

            # Trích xuất kỹ năng
            skills = []
            skills_elements = soup.select(".tag-list .tag")
            for skill_element in skills_elements:
                skill = skill_element.text.strip()
                if skill:
                    skills.append(skill)

            # Tạo đối tượng công việc
            from app.models.job import JobData, CompanyInfo, Location, JobType, JobSource, JobRequirement

            job = JobData(
                title=title,
                description=description,
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
                from app.services.job_search.web_search import WebSearcher
                web_searcher = WebSearcher()
                job.salary = web_searcher._parse_salary(salary_text)

            return job

        except Exception as e:
            logger.error(f"Lỗi khi crawl ITViec {url}: {str(e)}")
            return None

    async def _crawl_generic(self, url: str, html: str) -> Optional[JobData]:
        """
        Crawl thông tin từ trang web chung

        Args:
            url: URL việc làm
            html: Nội dung HTML

        Returns:
            Optional[JobData]: Thông tin việc làm
        """
        try:
            # Sử dụng BeautifulSoup để phân tích HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # Trích xuất tiêu đề từ thẻ title
            title_element = soup.title
            title = title_element.text.strip() if title_element else ""

            # Nếu tiêu đề có dạng "Tiêu đề - Công ty", tách thành tiêu đề và tên công ty
            title_parts = title.split(" - ", 1)
            job_title = title_parts[0].strip() if title_parts else title
            company_name = title_parts[1].strip() if len(title_parts) > 1 else ""

            # Trích xuất mô tả
            description = ""

            # Thử tìm phần tử có id hoặc class chứa "description"
            desc_selectors = ["#job-description", ".job-description", "#description", ".description",
                              "#job-detail", ".job-detail", "#content", ".content"]

            for selector in desc_selectors:
                desc_element = soup.select_one(selector)
                if desc_element:
                    description = desc_element.text.strip()
                    break

            # Nếu không tìm thấy, sử dụng nội dung của thẻ body
            if not description:
                body_element = soup.body
                if body_element:
                    # Loại bỏ script, style và các thẻ chứa nội dung không liên quan
                    for script in body_element(["script", "style", "nav", "header", "footer"]):
                        script.extract()

                    description = body_element.text.strip()

            # Giới hạn độ dài mô tả
            description = description[:2000] if description else ""

            # Tìm các công ty phổ biến trong nội dung
            companies = ["FPT", "Viettel", "VNG", "VinFast", "Momo", "Zalo", "Tiki", "Google",
                         "Microsoft", "Amazon", "IBM", "Samsung", "LG", "Intel", "TMA", "KMS"]

            if not company_name:
                for company in companies:
                    if company in description or company in title:
                        company_name = company
                        break

            # Tìm các thành phố phổ biến trong nội dung
            cities = ["Hà Nội", "TP HCM", "TP. HCM", "Hồ Chí Minh", "Đà Nẵng", "Hải Phòng",
                      "Cần Thơ", "Biên Hòa", "Nha Trang", "Quy Nhơn", "Huế"]

            location_text = ""
            for city in cities:
                if city in description or city in title:
                    location_text = city
                    break

            # Trích xuất mức lương
            salary_text = ""
            salary_patterns = [
                r'(?:mức lương|lương|salary|income)(?:\s*:)?\s*([\d\s.,]+\s*-\s*[\d\s.,]+)\s*(?:tr|triệu|VND|USD|\$)',
                r'(?:mức lương|lương|salary|income)(?:\s*:)?\s*([\d\s.,]+)\s*(?:tr|triệu|VND|USD|\$)',
                r'([\d\s.,]+\s*-\s*[\d\s.,]+)\s*(?:tr|triệu|VND|USD|\$)'
            ]

            for pattern in salary_patterns:
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    salary_text = match.group(1)
                    break

            # Trích xuất kỹ năng phổ biến
            common_skills = ["Java", "Python", "JavaScript", "C#", "C++", "PHP", "Ruby", "Go", "Swift",
                             "Kotlin", "React", "Angular", "Vue", "Node.js", "Django", "Flask", "Laravel",
                             "ASP.NET", "Spring", "SQL", "MongoDB", "AWS", "Azure", "GCP", "Docker",
                             "Kubernetes", "Linux", "Git", "Agile", "Scrum", "DevOps", "CI/CD", "TDD",
                             "UI/UX", "HTML", "CSS", "Figma", "Sketch", "Photoshop", "Power BI", "Excel",
                             "Data Analysis", "Machine Learning", "AI", "Deep Learning", "NLP", "Marketing",
                             "SEO", "Content", "Social Media", "Google Analytics", "Accounting", "Finance"]

            skills = []
            for skill in common_skills:
                if re.search(r'\b' + re.escape(skill) + r'\b', description, re.IGNORECASE):
                    skills.append(skill)

            # Tạo đối tượng công việc
            from app.models.job import JobData, CompanyInfo, Location, JobType, JobSource, JobRequirement

            job = JobData(
                title=job_title,
                description=description,
                company=CompanyInfo(name=company_name),
                location=Location(
                    city=location_text,
                    country="Việt Nam" if location_text else None
                ),
                job_type=JobType.FULL_TIME,
                requirements=JobRequirement(skills=skills),
                source=JobSource.OTHER,
                source_url=url,
                is_active=True,
                raw_text=html[:5000]  # Lưu 5000 ký tự đầu tiên của HTML gốc
            )

            # Thêm mức lương nếu có
            if salary_text:
                from app.services.job_search.web_search import WebSearcher
                web_searcher = WebSearcher()
                job.salary = web_searcher._parse_salary(salary_text)

            return job

        except Exception as e:
            logger.error(f"Lỗi khi crawl URL {url}: {str(e)}")
            return None

    def _get_from_cache(self, url: str) -> Optional[JobData]:
        """
        Lấy thông tin từ cache

        Args:
            url: URL công việc

        Returns:
            Optional[JobData]: Thông tin công việc từ cache
        """
        # Tạo key cache từ URL
        cache_key = self._get_cache_key(url)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")

        # Kiểm tra nếu tệp cache tồn tại và còn hiệu lực
        if os.path.exists(cache_path):
            try:
                # Kiểm tra thời gian tạo tệp
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
                if datetime.now() - file_time < timedelta(days=1):  # Cache có hiệu lực trong 1 ngày
                    # Đọc dữ liệu cache
                    with open(cache_path, "r", encoding="utf-8") as f:
                        job_dict = json.load(f)

                    # Chuyển đổi thành đối tượng JobData
                    from app.models.job import JobData
                    job = JobData.model_validate(job_dict)
                    return job
            except Exception as e:
                logger.error(f"Lỗi khi đọc cache: {str(e)}")

        return None

    def _save_to_cache(self, url: str, job: JobData) -> None:
        """
        Lưu thông tin vào cache

        Args:
            url: URL công việc
            job: Thông tin công việc
        """
        # Tạo key cache từ URL
        cache_key = self._get_cache_key(url)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")

        try:
            # Chuyển đổi JobData thành JSON
            job_dict = job.model_dump()

            # Ghi vào tệp cache
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(job_dict, f, ensure_ascii=False, indent=2)

            logger.info(f"Đã lưu thông tin vào cache: {cache_key}")

        except Exception as e:
            logger.error(f"Lỗi khi lưu cache: {str(e)}")

    def _get_cache_key(self, url: str) -> str:
        """
        Tạo key cache từ URL

        Args:
            url: URL công việc

        Returns:
            str: Key cache
        """
        # Tạo hash từ URL
        return hashlib.md5(url.encode("utf-8")).hexdigest()

    async def get_company_info(self, company_name: str) -> Optional[CompanyInfo]:
        """
        Lấy thông tin chi tiết về công ty

        Args:
            company_name: Tên công ty

        Returns:
            Optional[CompanyInfo]: Thông tin công ty
        """
        # Chức năng này có thể được mở rộng trong tương lai
        from app.models.job import CompanyInfo

        # Tạo thông tin công ty cơ bản
        company = CompanyInfo(
            name=company_name
        )

        # Thử tìm kiếm thông tin công ty trên web
        try:
            # Tạo URL tìm kiếm
            search_query = f"{company_name} company vietnam"

            # Thực hiện tìm kiếm
            import requests

            # Chọn ngẫu nhiên User-Agent
            user_agent = random.choice(self.user_agents)

            headers = {
                "User-Agent": user_agent
            }

            # Tìm kiếm trên Google
            search_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
            response = requests.get(search_url, headers=headers, timeout=10)

            if response.status_code == 200:
                # Phân tích HTML
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, "html.parser")

                # Tìm các thẻ meta có chứa thông tin về công ty
                description_element = soup.select_one('div.VwiC3b')
                if description_element:
                    company.description = description_element.text.strip()

                # Tìm website
                for link in soup.select('a'):
                    href = link.get('href', '')
                    if 'url=' in href and 'google.com' not in href and company_name.lower() in href.lower():
                        company.website = href.split('url=')[1].split('&')[0]
                        break
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin công ty {company_name}: {str(e)}")

        return company

    async def crawl_company_details(self, company_website: str) -> Dict[str, Any]:
        """
        Thu thập thông tin chi tiết về công ty từ trang web của công ty

        Args:
            company_website: URL trang web công ty

        Returns:
            Dict[str, Any]: Thông tin chi tiết về công ty
        """
        try:
            # Truy cập trang web công ty
            html = await self._fetch_url(company_website)
            if not html:
                return {}

            # Phân tích HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # Trích xuất thông tin
            company_info = {}

            # Tiêu đề trang (thường là tên công ty)
            title = soup.title.text.strip() if soup.title else ""
            company_info["title"] = title

            # Mô tả từ thẻ meta
            meta_description = soup.find("meta", {"name": "description"})
            if meta_description and "content" in meta_description.attrs:
                company_info["meta_description"] = meta_description["content"]

            # Thông tin liên hệ
            contact_info = {}

            # Tìm số điện thoại
            phone_pattern = re.compile(r'(?:\+84|0)(?:\d[ -.]?){9,10}')
            phone_matches = phone_pattern.findall(html)
            if phone_matches:
                contact_info["phone"] = phone_matches[0]

            # Tìm email
            email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
            email_matches = email_pattern.findall(html)
            if email_matches:
                contact_info["email"] = email_matches[0]

            # Tìm địa chỉ
            address_elements = soup.select(".address, .contact-address, footer .address")
            if address_elements:
                contact_info["address"] = address_elements[0].text.strip()

            company_info["contact"] = contact_info

            # Tìm các trang mạng xã hội
            social_media = {}
            social_platforms = {
                "facebook": ["facebook.com"],
                "linkedin": ["linkedin.com"],
                "twitter": ["twitter.com", "x.com"],
                "instagram": ["instagram.com"],
                "youtube": ["youtube.com"]
            }

            for platform, domains in social_platforms.items():
                for link in soup.select("a[href]"):
                    href = link.get("href", "")
                    if any(domain in href for domain in domains):
                        social_media[platform] = href
                        break

            company_info["social_media"] = social_media

            # Tìm thông tin về quy mô công ty
            company_size = None
            size_patterns = [
                r'(?:công ty|nhân viên|nhân sự|quy mô).{0,30}(\d+\s*-\s*\d+|\d+\+|\d+)',
                r'(?:company|employees|team|size).{0,30}(\d+\s*-\s*\d+|\d+\+|\d+)'
            ]

            for pattern in size_patterns:
                size_match = re.search(pattern, html, re.IGNORECASE)
                if size_match:
                    company_size = size_match.group(1)
                    break

            if company_size:
                company_info["company_size"] = company_size

            # Tìm thông tin về năm thành lập
            founded_year = None
            year_patterns = [
                r'(?:thành lập|thành lập năm|năm \d{4}|since|est\.?|established).{0,10}(\d{4})',
                r'(?:©|\(c\)|\&copy;).{0,10}(\d{4})'
            ]

            for pattern in year_patterns:
                year_match = re.search(pattern, html, re.IGNORECASE)
                if year_match:
                    founded_year = year_match.group(1)
                    break

            if founded_year:
                company_info["founded_year"] = founded_year

            return company_info

        except Exception as e:
            logger.error(f"Lỗi khi crawl thông tin công ty từ {company_website}: {str(e)}")
            return {}

    async def crawl_job_count(self, company_name: str) -> int:
        """
        Ước tính số lượng công việc đang tuyển dụng của công ty

        Args:
            company_name: Tên công ty

        Returns:
            int: Số lượng công việc ước tính
        """
        try:
            # Tạo URL tìm kiếm
            search_query = f'"{company_name}" tuyển dụng'

            # Thực hiện tìm kiếm
            from duckduckgo_search import DDGS

            # Thực hiện tìm kiếm
            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=30))

            # Đếm số lượng kết quả có liên quan đến tuyển dụng
            count = 0
            for result in results:
                if "tuyển dụng" in result.get("body", "").lower() and company_name.lower() in result.get("body",
                                                                                                         "").lower():
                    count += 1

            return count

        except Exception as e:
            logger.error(f"Lỗi khi ước tính số lượng công việc của {company_name}: {str(e)}")
            return 0

    async def crawl_salary_data(self, job_title: str) -> Dict[str, Any]:
        """
        Thu thập thông tin về mức lương trung bình cho vị trí công việc

        Args:
            job_title: Tên vị trí công việc

        Returns:
            Dict[str, Any]: Thông tin về mức lương
        """
        try:
            # Tạo URL tìm kiếm
            search_query = f"{job_title} mức lương trung bình"

            # Thực hiện tìm kiếm
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=10))

            # Phân tích kết quả
            salary_data = {
                "min": None,
                "max": None,
                "average": None,
                "sources": []
            }

            # Mẫu để tìm mức lương
            salary_patterns = [
                r'(\d[\d\s,.]+)(\s*-\s*)(\d[\d\s,.]+)(\s*)(triệu|tr|million)',
                r'(lương trung bình|mức lương|average salary|salary range)(\s*)(\d[\d\s,.]+)(\s*)(triệu|tr|million|VND)',
                r'(\d[\d\s,.]+)(\s*)(triệu|tr|million)(\s*)/(\s*)(tháng|month)'
            ]

            for result in results:
                body = result.get("body", "")
                for pattern in salary_patterns:
                    match = re.search(pattern, body, re.IGNORECASE)
                    if match:
                        if "-" in match.group(0):
                            # Trường hợp có khoảng lương
                            parts = re.split(r'\s*-\s*', match.group(0))
                            min_part = re.search(r'(\d[\d\s,.]+)', parts[0])
                            max_part = re.search(r'(\d[\d\s,.]+)', parts[1])

                            if min_part and max_part:
                                min_salary = float(min_part.group(1).replace(" ", "").replace(",", "").replace(".", ""))
                                max_salary = float(max_part.group(1).replace(" ", "").replace(",", "").replace(".", ""))

                                if "triệu" in match.group(0) or "tr" in match.group(0) or "million" in match.group(0):
                                    min_salary *= 1000000
                                    max_salary *= 1000000

                                if salary_data["min"] is None or min_salary < salary_data["min"]:
                                    salary_data["min"] = min_salary

                                if salary_data["max"] is None or max_salary > salary_data["max"]:
                                    salary_data["max"] = max_salary
                        else:
                            # Trường hợp có một mức lương
                            amount_match = re.search(r'(\d[\d\s,.]+)', match.group(0))
                            if amount_match:
                                amount = float(amount_match.group(1).replace(" ", "").replace(",", "").replace(".", ""))

                                if "triệu" in match.group(0) or "tr" in match.group(0) or "million" in match.group(0):
                                    amount *= 1000000

                                if "trung bình" in body.lower() or "average" in body.lower():
                                    salary_data["average"] = amount

                        # Thêm nguồn
                        source = {
                            "title": result.get("title", ""),
                            "url": result.get("href", ""),
                            "snippet": body[:200]
                        }

                        if source not in salary_data["sources"]:
                            salary_data["sources"].append(source)

            # Tính mức lương trung bình nếu chưa có
            if salary_data["average"] is None and salary_data["min"] is not None and salary_data["max"] is not None:
                salary_data["average"] = (salary_data["min"] + salary_data["max"]) / 2

            return salary_data

        except Exception as e:
            logger.error(f"Lỗi khi thu thập thông tin lương cho {job_title}: {str(e)}")
            return {"min": None, "max": None, "average": None, "sources": []}

    async def crawl_job_trends(self, job_title: str) -> Dict[str, Any]:
        """
        Thu thập thông tin về xu hướng việc làm

        Args:
            job_title: Tên vị trí công việc

        Returns:
            Dict[str, Any]: Thông tin về xu hướng việc làm
        """
        try:
            # Tạo URL tìm kiếm
            search_query = f"{job_title} xu hướng thị trường việc làm"

            # Thực hiện tìm kiếm
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=10))

            # Phân tích kết quả
            trends = {
                "demand": None,  # Nhu cầu (cao, trung bình, thấp)
                "growth": None,  # Tăng trưởng (tăng, ổn định, giảm)
                "competition": None,  # Mức độ cạnh tranh (cao, trung bình, thấp)
                "keywords": [],  # Các từ khóa xu hướng
                "sources": []  # Nguồn tham khảo
            }

            # Các từ khóa để xác định nhu cầu
            demand_keywords = {
                "high": ["nhu cầu cao", "nhu cầu lớn", "high demand", "tăng mạnh", "tăng nhanh", "hot", "khan hiếm"],
                "medium": ["nhu cầu ổn định", "stable demand", "vừa phải"],
                "low": ["nhu cầu thấp", "low demand", "giảm", "suy giảm", "khó khăn"]
            }

            # Các từ khóa để xác định tăng trưởng
            growth_keywords = {
                "increase": ["tăng trưởng", "tăng", "growth", "increase", "phát triển", "mở rộng"],
                "stable": ["ổn định", "stable", "không đổi", "duy trì"],
                "decrease": ["giảm", "decrease", "suy giảm", "thu hẹp", "cắt giảm"]
            }

            # Các từ khóa để xác định mức độ cạnh tranh
            competition_keywords = {
                "high": ["cạnh tranh cao", "high competition", "khốc liệt", "gay gắt"],
                "medium": ["cạnh tranh vừa phải", "medium competition"],
                "low": ["cạnh tranh thấp", "low competition", "dễ dàng", "ít cạnh tranh"]
            }

            # Từ khóa xu hướng
            trend_keywords = [
                "AI", "Machine Learning", "Data Science", "Cloud", "DevOps", "Remote Work", "Hybrid Work",
                "Blockchain", "Cybersecurity", "Digital Transformation", "Agile", "Scrum", "Microservices",
                "Docker", "Kubernetes", "AWS", "Azure", "GCP", "React", "Angular", "Vue", "Node.js",
                "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust", "Mobile Development", "IoT",
                "AR/VR", "5G", "Automation", "Big Data", "Analytics", "UX/UI", "Product Management"
            ]

            # Phân tích kết quả
            for result in results:
                body = result.get("body", "").lower()

                # Xác định nhu cầu
                if trends["demand"] is None:
                    for level, keywords in demand_keywords.items():
                        if any(keyword.lower() in body for keyword in keywords):
                            trends["demand"] = level
                            break

                # Xác định tăng trưởng
                if trends["growth"] is None:
                    for trend, keywords in growth_keywords.items():
                        if any(keyword.lower() in body for keyword in keywords):
                            trends["growth"] = trend
                            break

                # Xác định mức độ cạnh tranh
                if trends["competition"] is None:
                    for level, keywords in competition_keywords.items():
                        if any(keyword.lower() in body for keyword in keywords):
                            trends["competition"] = level
                            break

                # Tìm từ khóa xu hướng
                for keyword in trend_keywords:
                    if keyword.lower() in body and keyword not in trends["keywords"]:
                        trends["keywords"].append(keyword)

                # Thêm nguồn
                source = {
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", "")[:200]
                }

                if source not in trends["sources"]:
                    trends["sources"].append(source)

            # Giới hạn số lượng từ khóa
            trends["keywords"] = trends["keywords"][:10]

            return trends

        except Exception as e:
            logger.error(f"Lỗi khi thu thập xu hướng việc làm cho {job_title}: {str(e)}")
            return {"demand": None, "growth": None, "competition": None, "keywords": [], "sources": []}

    async def crawl_similar_jobs(self, job_title: str) -> List[str]:
        """
        Thu thập danh sách các vị trí công việc tương tự

        Args:
            job_title: Tên vị trí công việc

        Returns:
            List[str]: Danh sách các vị trí công việc tương tự
        """
        try:
            # Tạo URL tìm kiếm
            search_query = f"{job_title} vị trí tương tự công việc"

            # Thực hiện tìm kiếm
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=5))

            # Phân tích kết quả
            similar_jobs = []

            for result in results:
                body = result.get("body", "")

                # Tìm các vị trí công việc trong nội dung
                job_patterns = [
                    r'(?:vị trí|công việc|job titles?|roles?|positions?)(?:\s*:|\s+tương tự\s+)(?:[^.]+)',
                    r'(?:như|giống như|similar to|such as)(?:[^.]+)'
                ]

                for pattern in job_patterns:
                    match = re.search(pattern, body, re.IGNORECASE)
                    if match:
                        job_text = match.group(0)

                        # Tách các vị trí công việc
                        jobs = re.split(r',|\bvà\b|\band\b|\bor\b', job_text)

                        for job in jobs:
                            # Loại bỏ các từ không liên quan
                            job = re.sub(
                                r'(?:vị trí|công việc|job titles?|roles?|positions?|như|giống như|similar to|such as|:|tương tự)',
                                '', job, flags=re.IGNORECASE)

                            # Làm sạch chuỗi
                            job = job.strip()

                            if job and len(job) > 3 and job not in similar_jobs:
                                similar_jobs.append(job)

            # Nếu không tìm thấy kết quả, sử dụng danh sách mặc định
            if not similar_jobs:
                # Các vị trí tương tự mặc định cho một số ngành nghề phổ biến
                default_similar_jobs = {
                    "developer": ["Software Engineer", "Programmer", "Coder", "Full-stack Developer",
                                  "Backend Developer", "Frontend Developer"],
                    "designer": ["UI Designer", "UX Designer", "Graphic Designer", "Web Designer", "Product Designer"],
                    "manager": ["Team Lead", "Project Manager", "Product Manager", "Supervisor", "Director"],
                    "marketing": ["Marketing Specialist", "Digital Marketer", "Content Creator", "SEO Specialist",
                                  "Social Media Manager"],
                    "sales": ["Sales Executive", "Business Development", "Account Manager", "Sales Representative",
                              "Sales Consultant"],
                    "accountant": ["Financial Analyst", "Bookkeeper", "Financial Controller", "Auditor",
                                   "Tax Specialist"],
                    "hr": ["HR Specialist", "Recruiter", "Talent Acquisition", "HR Manager", "People Operations"],
                    "customer service": ["Customer Support", "Client Success", "Technical Support", "Help Desk",
                                         "Customer Relations"]
                }

                # Tìm danh sách phù hợp
                for key, jobs in default_similar_jobs.items():
                    if key in job_title.lower():
                        similar_jobs = jobs
                        break

            # Giới hạn số lượng kết quả
            return similar_jobs[:10]

        except Exception as e:
            logger.error(f"Lỗi khi thu thập công việc tương tự cho {job_title}: {str(e)}")
            return []

    async def crawl_required_skills(self, job_title: str) -> Dict[str, List[str]]:
        """
        Thu thập danh sách kỹ năng yêu cầu phổ biến cho vị trí công việc

        Args:
            job_title: Tên vị trí công việc

        Returns:
            Dict[str, List[str]]: Danh sách kỹ năng theo danh mục
        """
        try:
            # Tạo URL tìm kiếm
            search_query = f"{job_title} kỹ năng yêu cầu skills required"

            # Thực hiện tìm kiếm
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=7))

            # Phân tích kết quả
            skills = {
                "technical": [],  # Kỹ năng kỹ thuật
                "soft": [],  # Kỹ năng mềm
                "languages": [],  # Ngôn ngữ
                "tools": [],  # Công cụ
                "certifications": []  # Chứng chỉ
            }

            # Danh sách kỹ năng mềm phổ biến
            common_soft_skills = [
                "Communication", "Teamwork", "Problem-solving", "Critical thinking", "Adaptability",
                "Time management", "Leadership", "Creativity", "Work ethic", "Interpersonal skills",
                "Giao tiếp", "Làm việc nhóm", "Giải quyết vấn đề", "Tư duy phản biện", "Thích nghi",
                "Quản lý thời gian", "Lãnh đạo", "Sáng tạo", "Đạo đức làm việc", "Kỹ năng giao tiếp"
            ]

            # Danh sách ngôn ngữ phổ biến
            common_languages = [
                "English", "Chinese", "Japanese", "Korean", "French", "German", "Spanish", "Russian",
                "Tiếng Anh", "Tiếng Trung", "Tiếng Nhật", "Tiếng Hàn", "Tiếng Pháp", "Tiếng Đức", "Tiếng Tây Ban Nha",
                "Tiếng Nga"
            ]

            # Danh sách chứng chỉ phổ biến
            common_certifications = [
                "AWS", "Microsoft", "Google", "Cisco", "CompTIA", "PMI", "ITIL", "Agile", "Scrum",
                "PMP", "CISSP", "CISA", "CISM", "CFA", "ACCA", "CPA", "IELTS", "TOEIC", "TOEFL"
            ]

            for result in results:
                body = result.get("body", "")

                # Tìm danh sách kỹ năng
                skill_patterns = [
                    r'(?:kỹ năng|skills|required skills|yêu cầu)(?:\s*:|\s+yêu cầu\s+)(?:[^.]+)',
                    r'(?:technical skills|kỹ năng kỹ thuật)(?:\s*:)(?:[^.]+)',
                    r'(?:soft skills|kỹ năng mềm)(?:\s*:)(?:[^.]+)',
                    r'(?:tools|công cụ)(?:\s*:)(?:[^.]+)',
                    r'(?:certifications|chứng chỉ)(?:\s*:)(?:[^.]+)',
                    r'(?:languages|ngôn ngữ)(?:\s*:)(?:[^.]+)'
                ]

                for pattern in skill_patterns:
                    match = re.search(pattern, body, re.IGNORECASE)
                    if match:
                        skill_text = match.group(0)

                        # Tách các kỹ năng
                        skill_items = re.split(r',|\bvà\b|\band\b|\bor\b|\n|\s{2,}', skill_text)

                        for skill in skill_items:
                            # Loại bỏ các từ không liên quan
                            skill = re.sub(
                                r'(?:kỹ năng|skills|required skills|yêu cầu|technical skills|kỹ năng kỹ thuật|soft skills|kỹ năng mềm|tools|công cụ|certifications|chứng chỉ|languages|ngôn ngữ|:|yêu cầu)',
                                '', skill, flags=re.IGNORECASE)

                            # Làm sạch chuỗi
                            skill = skill.strip()

                            if skill and len(skill) > 2:
                                # Phân loại kỹ năng
                                if any(soft_skill.lower() in skill.lower() for soft_skill in common_soft_skills):
                                    if skill not in skills["soft"]:
                                        skills["soft"].append(skill)
                                elif any(language.lower() in skill.lower() for language in common_languages):
                                    if skill not in skills["languages"]:
                                        skills["languages"].append(skill)
                                elif any(cert.lower() in skill.lower() for cert in common_certifications):
                                    if skill not in skills["certifications"]:
                                        skills["certifications"].append(skill)
                                elif "software" in skill.lower() or "tool" in skill.lower() or "công cụ" in skill.lower() or "phần mềm" in skill.lower():
                                    if skill not in skills["tools"]:
                                        skills["tools"].append(skill)
                                else:
                                    if skill not in skills["technical"]:
                                        skills["technical"].append(skill)

            # Nếu không tìm thấy đủ kỹ năng, sử dụng danh sách mặc định
            if len(skills["technical"]) < 3:
                # Các kỹ năng kỹ thuật mặc định cho một số ngành nghề phổ biến
                default_technical_skills = {
                    "developer": ["Programming", "Coding", "Database", "Web Development", "Software Engineering"],
                    "designer": ["UI/UX Design", "Graphic Design", "Visual Design", "Layout Design", "Typography"],
                    "manager": ["Project Management", "Team Management", "Resource Planning", "Budgeting",
                                "Stakeholder Management"],
                    "marketing": ["Digital Marketing", "Content Marketing", "SEO", "SEM", "Social Media Marketing"],
                    "sales": ["Sales Techniques", "Negotiation", "Customer Relationship Management", "Lead Generation",
                              "Market Research"],
                    "accountant": ["Financial Analysis", "Bookkeeping", "Tax Planning", "Auditing",
                                   "Financial Reporting"],
                    "hr": ["Recruitment", "Employee Relations", "Talent Management", "Performance Management",
                           "Compensation and Benefits"],
                    "customer service": ["Customer Support", "Problem Resolution", "Product Knowledge",
                                         "Conflict Management", "Patience"]
                }

                # Tìm danh sách phù hợp
                for key, tech_skills in default_technical_skills.items():
                    if key in job_title.lower():
                        skills["technical"] = tech_skills
                        break

            # Giới hạn số lượng kỹ năng mỗi danh mục
            for category in skills:
                skills[category] = skills[category][:10]

            return skills

        except Exception as e:
            logger.error(f"Lỗi khi thu thập kỹ năng yêu cầu cho {job_title}: {str(e)}")
            return {"technical": [], "soft": [], "languages": [], "tools": [], "certifications": []}

    async def get_company_reviews(self, company_name: str) -> Dict[str, Any]:
        """
        Thu thập đánh giá về công ty

        Args:
            company_name: Tên công ty

        Returns:
            Dict[str, Any]: Thông tin đánh giá về công ty
        """
        try:
            # Tạo URL tìm kiếm
            search_query = f"{company_name} đánh giá review"

            # Thực hiện tìm kiếm
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=10))

            # Phân tích kết quả
            reviews = {
                "rating": None,  # Điểm đánh giá trung bình
                "positive": [],  # Đánh giá tích cực
                "negative": [],  # Đánh giá tiêu cực
                "sources": []  # Nguồn tham khảo
            }

            # Các từ khóa tích cực
            positive_keywords = [
                "tốt", "tuyệt vời", "excellent", "great", "good", "positive", "thú vị", "hài lòng",
                "satisfied", "recommend", "khuyến khích", "phúc lợi tốt", "lương cao", "cơ hội",
                "opportunity", "growth", "phát triển", "học hỏi", "learn", "flexible", "linh hoạt",
                "work-life balance", "cân bằng"
            ]

            # Các từ khóa tiêu cực
            negative_keywords = [
                "kém", "tồi", "bad", "poor", "negative", "không hài lòng", "disappointed",
                "stress", "áp lực", "overtime", "tăng ca", "low salary", "lương thấp",
                "no growth", "không phát triển", "toxic", "độc hại", "management issues",
                "vấn đề quản lý", "high turnover", "nghỉ việc nhiều"
            ]

            # Phân tích kết quả
            for result in results:
                body = result.get("body", "").lower()
                title = result.get("title", "").lower()

                # Tìm điểm đánh giá
                rating_pattern = r'(\d(?:[.,]\d)?)(?:\s*\/\s*\d+|\s*stars|\s*sao|\s*điểm)'
                rating_match = re.search(rating_pattern, body) or re.search(rating_pattern, title)

                if rating_match:
                    rating_text = rating_match.group(1).replace(",", ".")
                    rating = float(rating_text)

                    # Chuẩn hóa điểm về thang 5
                    if "/" in rating_match.group(0):
                        scale_match = re.search(r'\/\s*(\d+)', rating_match.group(0))
                        if scale_match:
                            scale = float(scale_match.group(1))
                            rating = rating / scale * 5

                    # Cập nhật điểm đánh giá
                    if reviews["rating"] is None:
                        reviews["rating"] = rating
                    else:
                        reviews["rating"] = (reviews["rating"] + rating) / 2

                # Tìm đánh giá tích cực
                for keyword in positive_keywords:
                    pattern = rf'(?:[^.!?]*{keyword}[^.!?]*[.!?])'
                    matches = re.findall(pattern, body)
                    for match in matches:
                        if match and match not in reviews["positive"] and len(reviews["positive"]) < 5:
                            reviews["positive"].append(match.strip())

                # Tìm đánh giá tiêu cực
                for keyword in negative_keywords:
                    pattern = rf'(?:[^.!?]*{keyword}[^.!?]*[.!?])'
                    matches = re.findall(pattern, body)
                    for match in matches:
                        if match and match not in reviews["negative"] and len(reviews["negative"]) < 5:
                            reviews["negative"].append(match.strip())

                # Thêm nguồn
                source = {
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", "")[:200]
                }

                if source not in reviews["sources"]:
                    reviews["sources"].append(source)

            return reviews

        except Exception as e:
            logger.error(f"Lỗi khi thu thập đánh giá về công ty {company_name}: {str(e)}")
            return {"rating": None, "positive": [], "negative": [], "sources": []}

    async def get_location_info(self, location: str) -> Dict[str, Any]:
        """
        Thu thập thông tin về địa điểm làm việc

        Args:
            location: Tên địa điểm

        Returns:
            Dict[str, Any]: Thông tin về địa điểm
        """
        try:
            # Tạo URL tìm kiếm
            search_query = f"{location} thông tin"

            # Thực hiện tìm kiếm
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=5))

            # Phân tích kết quả
            location_info = {
                "name": location,
                "description": "",
                "district": None,
                "city": None,
                "living_cost": None,  # Chi phí sinh hoạt
                "transportation": [],  # Thông tin giao thông
                "amenities": []  # Tiện ích xung quanh
            }

            # Danh sách các thành phố lớn
            major_cities = ["Hà Nội", "TP HCM", "TP. HCM", "Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ"]

            # Phân tích kết quả
            for result in results:
                body = result.get("body", "")

                # Tìm mô tả
                if not location_info["description"] and len(body) > 100:
                    location_info["description"] = body[:300]

                # Tìm quận/huyện và thành phố
                if not location_info["district"] or not location_info["city"]:
                    district_pattern = r'(?:quận|huyện|district)\s+([^.,;]+)'
                    city_pattern = r'(?:thành phố|tp\.?|tp|city)\s+([^.,;]+)'

                    district_match = re.search(district_pattern, body, re.IGNORECASE)
                    city_match = re.search(city_pattern, body, re.IGNORECASE)

                    if district_match:
                        location_info["district"] = district_match.group(1).strip()

                    if city_match:
                        location_info["city"] = city_match.group(1).strip()
                    elif any(city in body for city in major_cities):
                        for city in major_cities:
                            if city in body:
                                location_info["city"] = city
                                break

                # Tìm thông tin về chi phí sinh hoạt
                if not location_info["living_cost"]:
                    cost_pattern = r'(?:chi phí sinh hoạt|living cost|cost of living)(?:[^.]+)'
                    cost_match = re.search(cost_pattern, body, re.IGNORECASE)

                    if cost_match:
                        location_info["living_cost"] = cost_match.group(0).strip()

                # Tìm thông tin về giao thông
                transport_patterns = [
                    r'(?:giao thông|transportation|đi lại|phương tiện)(?:[^.]+)',
                    r'(?:bus|xe buýt|tàu điện|metro|subway|taxi|grab)(?:[^.]+)'
                ]

                for pattern in transport_patterns:
                    matches = re.findall(pattern, body, re.IGNORECASE)
                    for match in matches:
                        if match and match not in location_info["transportation"] and len(
                                location_info["transportation"]) < 3:
                            location_info["transportation"].append(match.strip())

                # Tìm thông tin về tiện ích xung quanh
                amenities_patterns = [
                    r'(?:tiện ích|amenities|facilities|dịch vụ|services)(?:[^.]+)',
                    r'(?:trường học|schools|bệnh viện|hospitals|công viên|parks|trung tâm thương mại|shopping centers|nhà hàng|restaurants)(?:[^.]+)'
                ]

                for pattern in amenities_patterns:
                    matches = re.findall(pattern, body, re.IGNORECASE)
                    for match in matches:
                        if match and match not in location_info["amenities"] and len(location_info["amenities"]) < 5:
                            location_info["amenities"].append(match.strip())

            return location_info

        except Exception as e:
            logger.error(f"Lỗi khi thu thập thông tin về địa điểm {location}: {str(e)}")
            return {"name": location, "description": "", "district": None, "city": None, "living_cost": None,
                    "transportation": [], "amenities": []}