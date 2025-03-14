import os
import logging
import tempfile
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, BinaryIO

from app.core.logging import get_logger
from app.models.cv import CVData, PersonalInfo, Education, WorkExperience, Skill, Language, Project, Certificate
from app.services.cv_parser.pdf import PDFParser
from app.services.cv_parser.word import WordParser
from app.services.cv_parser.excel import ExcelParser
from app.services.cv_parser.image import ImageParser

logger = get_logger("cv_parser")


class CVExtractor:
    """
    Lớp trích xuất dữ liệu từ tất cả các loại tệp CV
    """

    def __init__(self):
        """
        Khởi tạo parser và bộ trích xuất thông tin
        """
        self.pdf_parser = PDFParser()
        self.word_parser = WordParser()
        self.excel_parser = ExcelParser()
        self.image_parser = ImageParser()

        # Danh sách các phần mở rộng hỗ trợ
        self.supported_extensions = {
            'pdf': ['.pdf'],
            'word': ['.doc', '.docx', '.rtf'],
            'excel': ['.xls', '.xlsx', '.xlsm', '.csv'],
            'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        }

    def get_parser_for_file(self, file_path: str) -> Tuple[Any, str]:
        """
        Lấy parser phù hợp cho loại tệp

        Args:
            file_path: Đường dẫn đến tệp

        Returns:
            Tuple[Any, str]: Tuple gồm (parser, loại_tệp)
        """
        ext = Path(file_path).suffix.lower()

        for file_type, extensions in self.supported_extensions.items():
            if ext in extensions:
                if file_type == 'pdf':
                    return self.pdf_parser, 'pdf'
                elif file_type == 'word':
                    return self.word_parser, 'word'
                elif file_type == 'excel':
                    return self.excel_parser, 'excel'
                elif file_type == 'image':
                    return self.image_parser, 'image'

        raise ValueError(f"Không hỗ trợ loại tệp {ext}")

    def extract_text_from_file(self, file_path: str) -> Tuple[str, str]:
        """
        Trích xuất văn bản từ tệp

        Args:
            file_path: Đường dẫn đến tệp

        Returns:
            Tuple[str, str]: Tuple gồm (văn_bản, loại_tệp)
        """
        try:
            parser, file_type = self.get_parser_for_file(file_path)

            # Đối với hình ảnh, kiểm tra nếu nó là CV
            if file_type == 'image' and not self.image_parser.is_cv_image(file_path):
                logger.warning(f"Tệp {file_path} có vẻ không phải là CV")

            text = parser.extract_text(file_path)

            return text, file_type
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất văn bản từ {file_path}: {str(e)}")
            return "", "unknown"

    def extract_from_upload(self, file: BinaryIO, filename: str) -> Tuple[str, str]:
        """
        Trích xuất văn bản từ tệp tải lên

        Args:
            file: Đối tượng tệp
            filename: Tên tệp

        Returns:
            Tuple[str, str]: Tuple gồm (văn_bản, loại_tệp)
        """
        try:
            # Lưu tệp tạm thời
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_path = temp_file.name

            # Trích xuất văn bản
            text, file_type = self.extract_text_from_file(temp_path)

            # Xóa tệp tạm thời
            os.unlink(temp_path)

            return text, file_type
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất từ tệp tải lên {filename}: {str(e)}")
            return "", "unknown"

    def parse_cv_from_text(self, text: str, file_source: str = None, use_llm: bool = True) -> CVData:
        """
        Phân tích CV từ văn bản

        Args:
            text: Văn bản CV
            file_source: Tên tệp nguồn
            use_llm: Sử dụng LLM để phân tích

        Returns:
            CVData: Đối tượng dữ liệu CV
        """
        try:
            if not text or len(text.strip()) < 50:
                logger.error("Văn bản CV quá ngắn hoặc rỗng")
                return self._create_empty_cv_data(file_source)

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """
        Phân tích chuỗi ngày tháng thành đối tượng datetime

        Args:
            date_str: Chuỗi ngày tháng

        Returns:
            Optional[datetime]: Đối tượng datetime hoặc None nếu không thể phân tích
        """
        if not date_str:
            return None

        date_str = str(date_str).strip()

        try:
            # Thử với các định dạng phổ biến
            formats = [
                '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
                '%d-%m-%Y', '%m-%d-%Y',
                '%b %Y', '%B %Y',  # Tháng và năm (Jan 2020, January 2020)
                '%Y'  # Chỉ năm
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue

            # Nếu chỉ là năm (4 chữ số)
            if date_str.isdigit() and len(date_str) == 4:
                return datetime(int(date_str), 1, 1)

            return None
        except Exception:
            return None

    def _parse_int(self, value: Any) -> Optional[int]:
        """
        Chuyển đổi giá trị thành số nguyên

        Args:
            value: Giá trị cần chuyển đổi

        Returns:
            Optional[int]: Số nguyên hoặc None nếu không thể chuyển đổi
        """
        if value is None:
            return None

        try:
            if isinstance(value, str):
                # Loại bỏ các ký tự không phải số
                value = ''.join(c for c in value if c.isdigit())

            return int(value) if value else None
        except Exception:
            return None

    def _parse_float(self, value: Any) -> Optional[float]:
        """
        Chuyển đổi giá trị thành số thực

        Args:
            value: Giá trị cần chuyển đổi

        Returns:
            Optional[float]: Số thực hoặc None nếu không thể chuyển đổi
        """
        if value is None:
            return None

        try:
            if isinstance(value, str):
                # Thay thế dấu phẩy bằng dấu chấm
                value = value.replace(',', '.')

                # Loại bỏ các ký tự không phải số hoặc dấu chấm
                value = ''.join(c for c in value if c.isdigit() or c == '.')

            return float(value) if value else None
        except Exception:
            return None

    def _create_empty_cv_data(self, file_source: str = None) -> CVData:
        """
        Tạo đối tượng CVData trống

        Args:
            file_source: Tên tệp nguồn

        Returns:
            CVData: Đối tượng dữ liệu CV trống
        """
        return CVData(
            personal_info=PersonalInfo(name=""),
            extracted_from_file=file_source,
            confidence_score=0.0
        )

    def process_cv_file(self, file_path: str, use_llm: bool = True) -> CVData:
        """
        Xử lý tệp CV và trích xuất thông tin

        Args:
            file_path: Đường dẫn đến tệp CV
            use_llm: Sử dụng LLM để phân tích

        Returns:
            CVData: Đối tượng dữ liệu CV
        """
        try:
            # Trích xuất văn bản từ tệp
            text, file_type = self.extract_text_from_file(file_path)

            if not text:
                logger.error(f"Không thể trích xuất văn bản từ {file_path}")
                return self._create_empty_cv_data(os.path.basename(file_path))

            # Xử lý case đặc biệt cho ảnh
            if file_type == 'image':
                # Nếu là ảnh, có thể sử dụng AI Vision để trích xuất trực tiếp
                if use_llm:
                    cv_data = self.image_parser.extract_info_with_ai(file_path)
                    if cv_data:
                        result = self._convert_json_to_cv_data(cv_data, os.path.basename(file_path))
                        result.extracted_from_image = True
                        result.raw_text = text
                        return result

            # Phân tích văn bản
            cv_data = self.parse_cv_from_text(text, os.path.basename(file_path), use_llm)

            return cv_data
        except Exception as e:
            logger.error(f"Lỗi khi xử lý tệp CV {file_path}: {str(e)}")
            return self._create_empty_cv_data(os.path.basename(file_path))

    def enhance_cv_data(self, cv_data: CVData) -> CVData:
        """
        Nâng cao dữ liệu CV bằng cách suy luận thông tin thiếu

        Args:
            cv_data: Đối tượng dữ liệu CV

        Returns:
            CVData: Đối tượng dữ liệu CV đã được nâng cao
        """
        try:
            # Ước tính tổng số năm kinh nghiệm nếu không có
            if cv_data.years_of_experience is None and cv_data.work_experience:
                total_years = 0
                for exp in cv_data.work_experience:
                    if exp.start_date and (exp.end_date or exp.current):
                        end_date = datetime.now() if exp.current else exp.end_date
                        years = (end_date.year - exp.start_date.year) + \
                                (1 if end_date.month >= exp.start_date.month else 0)
                        total_years += years

                if total_years > 0:
                    cv_data.years_of_experience = total_years

            # Đoán job title từ kinh nghiệm gần đây nếu không có
            if not cv_data.job_title and cv_data.work_experience:
                # Sắp xếp theo thời gian giảm dần
                recent_jobs = sorted(
                    [job for job in cv_data.work_experience if job.start_date],
                    key=lambda x: x.start_date,
                    reverse=True
                )

                if recent_jobs:
                    cv_data.job_title = recent_jobs[0].position

            # Đoán preferred location từ vị trí hiện tại
            if not cv_data.preferred_location:
                if cv_data.personal_info.city:
                    cv_data.preferred_location = cv_data.personal_info.city
                elif cv_data.work_experience and cv_data.work_experience[0].location:
                    cv_data.preferred_location = cv_data.work_experience[0].location

            return cv_data
        except Exception as e:
            logger.error(f"Lỗi khi nâng cao dữ liệu CV: {str(e)}")
            return cv_data

            if use_llm:
                # Sử dụng LLM để phân tích
                cv_data = self._extract_with_llm(text, file_source)
            else:
                # Sử dụng phương pháp truyền thống (regex, rules)
                cv_data = self._extract_with_rules(text, file_source)

            # Thêm văn bản gốc
            cv_data.raw_text = text

            return cv_data
        except Exception as e:
            logger.error(f"Lỗi khi phân tích CV từ văn bản: {str(e)}")
            return self._create_empty_cv_data(file_source)

    def _extract_with_llm(self, text: str, file_source: str = None) -> CVData:
        """
        Sử dụng LLM để trích xuất thông tin từ văn bản CV

        Args:
            text: Văn bản CV
            file_source: Tên tệp nguồn

        Returns:
            CVData: Đối tượng dữ liệu CV
        """
        try:
            from app.services.llm.local_model import LocalLLM

            llm = LocalLLM()

            prompt = f"""
            Dưới đây là văn bản trích xuất từ một CV/Resume. Hãy phân tích và trích xuất các thông tin quan trọng dưới dạng JSON.

            Văn bản CV:
            {text[:5000]}  # Giới hạn độ dài để tránh quá tải LLM

            Hãy trích xuất và trả về thông tin dưới dạng JSON với các trường sau:
            {{
                "personal_info": {{
                    "name": "Tên người",
                    "email": "email@example.com",
                    "phone": "Số điện thoại",
                    "address": "Địa chỉ",
                    "city": "Thành phố",
                    "country": "Quốc gia",
                    "linkedin": "URL LinkedIn",
                    "github": "URL Github",
                    "website": "URL website cá nhân",
                    "summary": "Tóm tắt bản thân"
                }},
                "education": [
                    {{
                        "institution": "Tên trường",
                        "degree": "Bằng cấp",
                        "field_of_study": "Ngành học",
                        "start_date": "Ngày bắt đầu (YYYY-MM-DD hoặc chỉ năm)",
                        "end_date": "Ngày kết thúc (YYYY-MM-DD hoặc chỉ năm)",
                        "description": "Mô tả",
                        "gpa": "Điểm trung bình"
                    }}
                ],
                "work_experience": [
                    {{
                        "company": "Tên công ty",
                        "position": "Vị trí",
                        "start_date": "Ngày bắt đầu (YYYY-MM-DD hoặc chỉ năm)",
                        "end_date": "Ngày kết thúc (YYYY-MM-DD hoặc chỉ năm)",
                        "current": true/false,
                        "description": "Mô tả công việc",
                        "achievements": ["Thành tựu 1", "Thành tựu 2"],
                        "location": "Địa điểm"
                    }}
                ],
                "skills": [
                    {{
                        "name": "Tên kỹ năng",
                        "level": "Cấp độ (nếu có)",
                        "years": "Số năm kinh nghiệm (nếu có)",
                        "category": "Danh mục kỹ năng"
                    }}
                ],
                "languages": [
                    {{
                        "name": "Tên ngôn ngữ",
                        "proficiency": "Mức độ thành thạo"
                    }}
                ],
                "projects": [
                    {{
                        "name": "Tên dự án",
                        "description": "Mô tả",
                        "technologies": ["Công nghệ 1", "Công nghệ 2"],
                        "url": "URL dự án",
                        "role": "Vai trò"
                    }}
                ],
                "certificates": [
                    {{
                        "name": "Tên chứng chỉ",
                        "issuer": "Tổ chức cấp",
                        "date_issued": "Ngày cấp (YYYY-MM-DD)",
                        "expiration_date": "Ngày hết hạn (YYYY-MM-DD)",
                        "credential_id": "ID chứng chỉ",
                        "url": "URL chứng chỉ"
                    }}
                ],
                "job_title": "Vị trí công việc",
                "years_of_experience": "Số năm kinh nghiệm tổng cộng",
                "salary_expectation": "Mức lương mong muốn",
                "preferred_location": "Địa điểm làm việc mong muốn"
            }}

            Chỉ trả về dữ liệu JSON hợp lệ, không thêm bất kỳ chú thích nào.
            """

            result = llm.generate(prompt)

            # Tìm và trích xuất JSON từ kết quả
            json_result = self._extract_json_from_text(result)

            # Chuyển đổi dữ liệu JSON thành đối tượng CVData
            return self._convert_json_to_cv_data(json_result, file_source)

        except Exception as e:
            logger.error(f"Lỗi khi trích xuất thông tin với LLM: {str(e)}")
            return self._create_empty_cv_data(file_source)

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Trích xuất JSON từ văn bản

        Args:
            text: Văn bản có chứa JSON

        Returns:
            Dict[str, Any]: Dữ liệu JSON
        """
        try:
            # Tìm chuỗi JSON trong văn bản
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # Thử tải trực tiếp
                return json.loads(text)
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất JSON: {str(e)}")
            return {}

    def _extract_with_rules(self, text: str, file_source: str = None) -> CVData:
        """
        Sử dụng các quy tắc và biểu thức chính quy để trích xuất thông tin từ văn bản CV

        Args:
            text: Văn bản CV
            file_source: Tên tệp nguồn

        Returns:
            CVData: Đối tượng dữ liệu CV
        """
        # Tạo dữ liệu CV trống
        cv_data = self._create_empty_cv_data(file_source)

        try:
            # Sử dụng regex để trích xuất thông tin
            import re

            # Trích xuất email
            email_regex = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            email_matches = re.findall(email_regex, text)
            if email_matches:
                cv_data.personal_info.email = email_matches[0]

            # Trích xuất số điện thoại
            phone_regex = r'(?:\+84|0)(?:\d[ -.]?){9,10}'
            phone_matches = re.findall(phone_regex, text)
            if phone_matches:
                cv_data.personal_info.phone = phone_matches[0]

            # Trích xuất tên (khó vì đa dạng, đây chỉ là hàm đơn giản)
            # Giả định tên có thể ở đầu CV hoặc gần địa chỉ email/điện thoại
            lines = text.split('\n')
            for i, line in enumerate(lines[:10]):  # Chỉ xem 10 dòng đầu tiên
                line = line.strip()
                if len(line) > 3 and len(line.split()) <= 5 and not re.search(r'[0-9@.:]', line):
                    cv_data.personal_info.name = line
                    break

            # Tìm kỹ năng
            skill_sections = re.split(r'(?i)kỹ năng|skills|chuyên môn', text)
            if len(skill_sections) > 1:
                skill_text = skill_sections[1].split('\n\n')[0]
                skill_items = re.split(r'[,•\n-]', skill_text)
                for item in skill_items:
                    item = item.strip()
                    if item and len(item) > 2:
                        cv_data.skills.append(Skill(name=item))

            # Trích xuất học vấn
            edu_regex = r'(?i)(university|trường|học viện|cao đẳng|college|institute)\s+of\s+(.+?)[\n,.]'
            edu_matches = re.findall(edu_regex, text)
            for match in edu_matches:
                edu = Education(institution=' '.join(match).strip())
                cv_data.education.append(edu)

            # TODO: Trích xuất thêm thông tin khác (kinh nghiệm, dự án, v.v.)

            return cv_data

        except Exception as e:
            logger.error(f"Lỗi khi trích xuất thông tin với quy tắc: {str(e)}")
            return cv_data

    def _convert_json_to_cv_data(self, json_data: Dict[str, Any], file_source: str = None) -> CVData:
        """
        Chuyển đổi dữ liệu JSON thành đối tượng CVData

        Args:
            json_data: Dữ liệu JSON
            file_source: Tên tệp nguồn

        Returns:
            CVData: Đối tượng dữ liệu CV
        """
        try:
            # Xử lý personal_info
            personal_info_data = json_data.get('personal_info', {})
            personal_info = PersonalInfo(
                name=personal_info_data.get('name', ''),
                email=personal_info_data.get('email'),
                phone=personal_info_data.get('phone'),
                address=personal_info_data.get('address'),
                city=personal_info_data.get('city'),
                country=personal_info_data.get('country'),
                linkedin=personal_info_data.get('linkedin'),
                github=personal_info_data.get('github'),
                website=personal_info_data.get('website'),
                summary=personal_info_data.get('summary')
            )

            # Xử lý education
            education_list = []
            for edu_data in json_data.get('education', []):
                try:
                    # Chuyển đổi ngày tháng nếu có
                    start_date = self._parse_date(edu_data.get('start_date'))
                    end_date = self._parse_date(edu_data.get('end_date'))

                    education = Education(
                        institution=edu_data.get('institution', ''),
                        degree=edu_data.get('degree'),
                        field_of_study=edu_data.get('field_of_study'),
                        start_date=start_date,
                        end_date=end_date,
                        description=edu_data.get('description'),
                        gpa=self._parse_float(edu_data.get('gpa'))
                    )
                    education_list.append(education)
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý education: {str(e)}")

            # Xử lý work_experience
            work_exp_list = []
            for work_data in json_data.get('work_experience', []):
                try:
                    # Chuyển đổi ngày tháng
                    start_date = self._parse_date(work_data.get('start_date'))
                    end_date = self._parse_date(work_data.get('end_date'))

                    work_exp = WorkExperience(
                        company=work_data.get('company', ''),
                        position=work_data.get('position', ''),
                        start_date=start_date,
                        end_date=end_date,
                        current=work_data.get('current', False),
                        description=work_data.get('description'),
                        achievements=work_data.get('achievements', []),
                        location=work_data.get('location')
                    )
                    work_exp_list.append(work_exp)
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý work experience: {str(e)}")

            # Xử lý skills
            skill_list = []
            for skill_data in json_data.get('skills', []):
                try:
                    skill = Skill(
                        name=skill_data.get('name', ''),
                        level=skill_data.get('level'),
                        years=self._parse_int(skill_data.get('years')),
                        category=skill_data.get('category')
                    )
                    skill_list.append(skill)
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý skill: {str(e)}")

            # Xử lý languages
            language_list = []
            for lang_data in json_data.get('languages', []):
                try:
                    language = Language(
                        name=lang_data.get('name', ''),
                        proficiency=lang_data.get('proficiency')
                    )
                    language_list.append(language)
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý language: {str(e)}")

            # Xử lý projects
            project_list = []
            for proj_data in json_data.get('projects', []):
                try:
                    # Chuyển đổi ngày tháng
                    start_date = self._parse_date(proj_data.get('start_date'))
                    end_date = self._parse_date(proj_data.get('end_date'))

                    project = Project(
                        name=proj_data.get('name', ''),
                        description=proj_data.get('description'),
                        start_date=start_date,
                        end_date=end_date,
                        technologies=proj_data.get('technologies', []),
                        url=proj_data.get('url'),
                        role=proj_data.get('role')
                    )
                    project_list.append(project)
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý project: {str(e)}")

            # Xử lý certificates
            certificate_list = []
            for cert_data in json_data.get('certificates', []):
                try:
                    # Chuyển đổi ngày tháng
                    date_issued = self._parse_date(cert_data.get('date_issued'))
                    expiration_date = self._parse_date(cert_data.get('expiration_date'))

                    certificate = Certificate(
                        name=cert_data.get('name', ''),
                        issuer=cert_data.get('issuer'),
                        date_issued=date_issued,
                        expiration_date=expiration_date,
                        credential_id=cert_data.get('credential_id'),
                        url=cert_data.get('url')
                    )
                    certificate_list.append(certificate)
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý certificate: {str(e)}")

            # Tạo dữ liệu CV
            cv_data = CVData(
                personal_info=personal_info,
                education=education_list,
                work_experience=work_exp_list,
                skills=skill_list,
                languages=language_list,
                projects=project_list,
                certificates=certificate_list,
                job_title=json_data.get('job_title'),
                years_of_experience=self._parse_int(json_data.get('years_of_experience')),
                salary_expectation=json_data.get('salary_expectation'),
                preferred_location=json_data.get('preferred_location'),
                extracted_from_file=file_source,
                confidence_score=0.8
            )

            return cv_data

        except Exception as e:
            logger.error(f"Lỗi khi chuyển đổi JSON thành CV data: {str(e)}")
            return self._create_empty_cv_data(file_source)