import os
import logging
import re
import tempfile
import json
import time
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
        Xử lý tệp CV và trích xuất thông tin với cơ chế retry và xử lý lỗi tốt hơn

        Args:
            file_path: Đường dẫn đến tệp CV
            use_llm: Sử dụng LLM để phân tích

        Returns:
            CVData: Đối tượng dữ liệu CV
        """
        start_time = time.time()
        logger.info(f"Bắt đầu xử lý tệp CV: {file_path}")

        # Kiểm tra tệp tồn tại và có thể đọc được
        if not os.path.exists(file_path):
            logger.error(f"Tệp CV không tồn tại: {file_path}")
            return self._create_empty_cv_data(os.path.basename(file_path))

        if not os.path.isfile(file_path) or not os.access(file_path, os.R_OK):
            logger.error(f"Không thể đọc tệp CV: {file_path}")
            return self._create_empty_cv_data(os.path.basename(file_path))

        try:
            # Xác định loại tệp
            file_extension = os.path.splitext(file_path)[1].lower()

            # Đảm bảo tệp có kích thước hợp lý
            file_size = os.path.getsize(file_path)
            logger.info(f"Kích thước tệp: {file_size / 1024:.2f} KB")

            if file_size == 0:
                logger.error(f"Tệp CV rỗng: {file_path}")
                return self._create_empty_cv_data(os.path.basename(file_path))

            if file_size > 10 * 1024 * 1024:  # Giới hạn 10MB
                logger.warning(f"Tệp CV quá lớn ({file_size / 1024 / 1024:.2f} MB): {file_path}")

            # Trích xuất văn bản từ tệp
            text, file_type = None, None

            # Thử lại tối đa 3 lần
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    text, file_type = self.extract_text_from_file(file_path)
                    if text:
                        break

                    logger.warning(f"Không trích xuất được văn bản, thử lại ({attempt + 1}/{max_retries})")
                    time.sleep(1)  # Chờ 1 giây trước khi thử lại
                except Exception as e:
                    logger.error(f"Lỗi khi trích xuất văn bản (lần {attempt + 1}): {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                    else:
                        raise

            if not text:
                logger.error(f"Không thể trích xuất văn bản từ {file_path} sau {max_retries} lần thử")

                # Nếu là hình ảnh, thử biện pháp dự phòng với OCR nâng cao
                if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']:
                    logger.info(f"Thử dùng OCR nâng cao cho hình ảnh {file_path}")
                    from app.services.cv_parser.image import ImageParser
                    image_parser = ImageParser()

                    # Cải thiện chất lượng hình ảnh
                    improved_image = image_parser.improve_image_for_ocr(file_path)
                    text = image_parser.extract_text(improved_image)

                    if improved_image != file_path and os.path.exists(improved_image):
                        os.unlink(improved_image)  # Xóa tệp cải thiện tạm thời

                    if not text:
                        # Nếu vẫn không trích xuất được văn bản, sử dụng AI Vision
                        if use_llm:
                            logger.info(f"Thử dùng AI Vision cho hình ảnh {file_path}")
                            cv_data = image_parser.extract_info_with_ai(file_path)
                            if cv_data:
                                result = self._convert_json_to_cv_data(cv_data, os.path.basename(file_path))
                                result.extracted_from_image = True

                                # Thêm thông tin thêm
                                processing_time = time.time() - start_time
                                logger.info(f"Đã xử lý tệp CV trong {processing_time:.2f} giây với AI Vision")
                                result.confidence_score = 0.7  # AI Vision thường có độ chính xác khá

                                return result

                # Trả về CV rỗng nếu không thể trích xuất văn bản
                return self._create_empty_cv_data(os.path.basename(file_path))

            # Thông tin về văn bản đã trích xuất
            logger.info(f"Đã trích xuất {len(text)} ký tự từ {file_path}")

            # Giới hạn kích thước văn bản để tránh quá tải LLM
            max_text_size = 15000  # 15K ký tự
            if len(text) > max_text_size:
                logger.warning(f"Văn bản quá dài ({len(text)} ký tự), cắt bớt xuống {max_text_size} ký tự")
                text = text[:max_text_size]

            # Phân tích CV từ văn bản
            cv_data = self.parse_cv_from_text(text, os.path.basename(file_path), use_llm)

            # Nâng cao dữ liệu CV
            cv_data = self.enhance_cv_data(cv_data)

            # Đánh giá độ tin cậy của kết quả
            confidence_score = self._evaluate_confidence(cv_data)
            cv_data.confidence_score = confidence_score

            # Ghi lại thời gian xử lý
            processing_time = time.time() - start_time
            logger.info(f"Đã xử lý tệp CV trong {processing_time:.2f} giây (độ tin cậy: {confidence_score:.2f})")

            return cv_data

        except Exception as e:
            logger.error(f"Lỗi không xử lý được khi xử lý tệp CV {file_path}: {str(e)}", exc_info=True)
            return self._create_empty_cv_data(os.path.basename(file_path))

    def _evaluate_confidence(self, cv_data: CVData) -> float:
        """
        Đánh giá độ tin cậy của dữ liệu CV đã trích xuất

        Args:
            cv_data: Đối tượng dữ liệu CV

        Returns:
            float: Điểm độ tin cậy (0.0 - 1.0)
        """
        score = 0.0
        total_weight = 0.0

        # Kiểm tra các trường thông tin cá nhân
        personal_weight = 0.3
        personal_score = 0.0

        if cv_data.personal_info.name and len(cv_data.personal_info.name) > 3:
            personal_score += 0.4

        if cv_data.personal_info.email:
            personal_score += 0.3

        if cv_data.personal_info.phone:
            personal_score += 0.3

        score += personal_score * personal_weight
        total_weight += personal_weight

        # Kiểm tra học vấn
        education_weight = 0.2
        education_score = 0.0

        if cv_data.education and len(cv_data.education) > 0:
            edu_items = min(len(cv_data.education), 3)  # Tối đa 3 mục
            education_score = edu_items / 3.0

        score += education_score * education_weight
        total_weight += education_weight

        # Kiểm tra kinh nghiệm làm việc
        experience_weight = 0.3
        experience_score = 0.0

        if cv_data.work_experience and len(cv_data.work_experience) > 0:
            exp_items = min(len(cv_data.work_experience), 5)  # Tối đa 5 mục
            experience_score = exp_items / 5.0

        score += experience_score * experience_weight
        total_weight += experience_weight

        # Kiểm tra kỹ năng
        skills_weight = 0.2
        skills_score = 0.0

        if cv_data.skills and len(cv_data.skills) > 0:
            skills_items = min(len(cv_data.skills), 10)  # Tối đa 10 mục
            skills_score = skills_items / 10.0

        score += skills_score * skills_weight
        total_weight += skills_weight

        # Tính điểm cuối cùng
        final_score = score / total_weight if total_weight > 0 else 0.0

        # Giới hạn trong khoảng [0.0, 1.0]
        return max(0.0, min(1.0, final_score))

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

    def _extract_with_llm(self, text: str, file_source: str = None) -> CVData:
        """
        Sử dụng LLM để trích xuất thông tin từ văn bản CV với xử lý lỗi tốt hơn

        Args:
            text: Văn bản CV
            file_source: Tên tệp nguồn

        Returns:
            CVData: Đối tượng dữ liệu CV
        """
        try:
            from app.services.llm.local_model import LocalLLM
            import json
            import re
            from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

            # Chuẩn bị văn bản đầu vào
            # Làm sạch văn bản
            cleaned_text = self._clean_text(text)

            # Giới hạn độ dài văn bản
            max_length = 5000
            if len(cleaned_text) > max_length:
                logger.warning(f"Văn bản quá dài ({len(cleaned_text)} ký tự), cắt bớt xuống {max_length} ký tự")
                cleaned_text = cleaned_text[:max_length]

            # Khởi tạo LLM với retry
            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                retry=retry_if_exception_type(Exception)
            )
            def generate_with_llm(prompt):
                llm = LocalLLM()
                return llm.generate(prompt, max_tokens=2000, temperature=0.2)  # Giảm temperature để kết quả ổn định hơn

            # Tạo prompt chi tiết hơn
            prompt = f"""
            Dưới đây là văn bản trích xuất từ một CV/Resume. Hãy phân tích cẩn thận và trích xuất các thông tin quan trọng dưới dạng JSON.

            Văn bản CV:
            ```
            {cleaned_text}
            ```

            Hãy trích xuất và trả về thông tin dưới dạng JSON với các trường sau:
            {{
                "personal_info": {{
                    "name": "Tên người (trích xuất từ văn bản)",
                    "email": "email nếu có (phải đúng định dạng email)",
                    "phone": "số điện thoại nếu có (chỉ gồm chữ số và dấu +)",
                    "address": "địa chỉ nếu có",
                    "city": "thành phố nếu có",
                    "country": "quốc gia nếu có",
                    "linkedin": "URL LinkedIn nếu có",
                    "github": "URL Github nếu có",
                    "website": "URL website cá nhân nếu có",
                    "summary": "tóm tắt bản thân nếu có"
                }},
                "education": [
                    {{
                        "institution": "tên trường",
                        "degree": "bằng cấp nếu có",
                        "field_of_study": "ngành học nếu có",
                        "start_date": "ngày bắt đầu (YYYY-MM-DD hoặc YYYY) nếu có",
                        "end_date": "ngày kết thúc (YYYY-MM-DD hoặc YYYY) nếu có",
                        "description": "mô tả nếu có",
                        "gpa": "điểm trung bình nếu có"
                    }}
                ],
                "work_experience": [
                    {{
                        "company": "tên công ty",
                        "position": "vị trí",
                        "start_date": "ngày bắt đầu (YYYY-MM-DD hoặc YYYY) nếu có",
                        "end_date": "ngày kết thúc (YYYY-MM-DD hoặc YYYY) nếu có",
                        "current": true/false (đang làm việc tại đây hay không),
                        "description": "mô tả công việc nếu có",
                        "achievements": ["thành tựu 1", "thành tựu 2"] (nếu có),
                        "location": "địa điểm làm việc nếu có"
                    }}
                ],
                "skills": [
                    {{
                        "name": "tên kỹ năng",
                        "level": "cấp độ nếu có",
                        "years": "số năm kinh nghiệm nếu có",
                        "category": "danh mục kỹ năng nếu có"
                    }}
                ],
                "languages": [
                    {{
                        "name": "tên ngôn ngữ",
                        "proficiency": "mức độ thành thạo nếu có"
                    }}
                ],
                "projects": [
                    {{
                        "name": "tên dự án",
                        "description": "mô tả nếu có",
                        "technologies": ["công nghệ 1", "công nghệ 2"] (nếu có),
                        "url": "URL dự án nếu có",
                        "role": "vai trò nếu có"
                    }}
                ],
                "certificates": [
                    {{
                        "name": "tên chứng chỉ",
                        "issuer": "tổ chức cấp nếu có",
                        "date_issued": "ngày cấp (YYYY-MM-DD) nếu có",
                        "expiration_date": "ngày hết hạn (YYYY-MM-DD) nếu có",
                        "credential_id": "ID chứng chỉ nếu có",
                        "url": "URL chứng chỉ nếu có"
                    }}
                ],
                "job_title": "vị trí công việc mong muốn (ví dụ: Software Engineer)",
                "years_of_experience": số năm kinh nghiệm tổng cộng (số nguyên, chỉ điền nếu có thông tin rõ ràng),
                "salary_expectation": "mức lương mong muốn nếu có",
                "preferred_location": "địa điểm làm việc mong muốn nếu có"
            }}

            Chỉ trả về dữ liệu JSON hợp lệ, không thêm bất kỳ chú thích nào.
            Nếu không thấy thông tin nào, hãy để trống hoặc null.
            Đảm bảo JSON hợp lệ, đặc biệt là dấu ngoặc, dấu phẩy và chuỗi.
            """

            # Thực hiện gọi LLM
            result = generate_with_llm(prompt)

            # Tìm và trích xuất JSON từ kết quả
            json_result = self._extract_json_from_text(result)

            if not json_result:
                # Nếu không tìm thấy JSON, thử lại với prompt đơn giản hơn
                logger.warning("Không tìm thấy JSON trong kết quả LLM, thử lại với prompt đơn giản hơn")
                simple_prompt = f"""
                Dưới đây là văn bản trích xuất từ một CV. Hãy phân tích và trả về thông tin dưới dạng JSON đơn giản:

                ```
                {cleaned_text[:3000]}
                ```

                Trả về dạng JSON:
                {{
                    "personal_info": {{ "name": "", "email": "", "phone": "" }},
                    "education": [{{ "institution": "", "degree": "", "field_of_study": "" }}],
                    "work_experience": [{{ "company": "", "position": "" }}],
                    "skills": [{{ "name": "" }}]
                }}
                """

                result = generate_with_llm(simple_prompt)
                json_result = self._extract_json_from_text(result)

                if not json_result:
                    logger.error("Không thể trích xuất JSON từ kết quả LLM sau nhiều lần thử")
                    return self._create_empty_cv_data(file_source)

            # Chuyển đổi dữ liệu JSON thành đối tượng CVData
            return self._convert_json_to_cv_data(json_result, file_source)

        except Exception as e:
            logger.error(f"Lỗi không xử lý được khi trích xuất với LLM: {str(e)}", exc_info=True)
            return self._create_empty_cv_data(file_source)

    def _clean_text(self, text: str) -> str:
        """
        Làm sạch văn bản đầu vào

        Args:
            text: Văn bản cần làm sạch

        Returns:
            str: Văn bản đã làm sạch
        """
        if not text:
            return ""

        # Loại bỏ các ký tự đặc biệt không cần thiết
        text = re.sub(r'[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]', '', text)

        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text)

        # Loại bỏ các dòng trống liên tiếp
        text = re.sub(r'\n\s*\n+', '\n\n', text)

        # Loại bỏ khoảng trắng đầu dòng
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)

        return text.strip()

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