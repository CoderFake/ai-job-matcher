"""
Module xử lý trích xuất thông tin từ ảnh CV
"""

import os
import logging
import tempfile
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import base64

from app.core.logging import get_logger
from app.core.config import get_model_config

logger = get_logger("cv_parser")

class ImageParser:
    """
    Lớp trích xuất dữ liệu từ ảnh CV
    """

    def __init__(self, use_gpu: Optional[bool] = None):
        """
        Khởi tạo parser

        Args:
            use_gpu: Sử dụng GPU hay không. Nếu None, sẽ sử dụng cấu hình từ settings
        """
        self.model_config = get_model_config()
        self.use_gpu = use_gpu if use_gpu is not None else self.model_config.get('device') == 'cuda'

        # Các extension hình ảnh hỗ trợ
        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']

    def extract_text(self, image_path: str) -> str:
        """
        Trích xuất văn bản từ ảnh CV

        Args:
            image_path: Đường dẫn đến file ảnh

        Returns:
            str: Văn bản đã trích xuất
        """
        try:
            # Thử sử dụng pytesseract cho OCR
            import pytesseract
            from PIL import Image

            # Mở và tiền xử lý ảnh
            image = Image.open(image_path)

            # Tối ưu hóa ảnh cho OCR
            image = self._preprocess_image(image)

            # Thực hiện OCR
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(image, lang='vie+eng', config=custom_config)

            return text
        except ImportError:
            logger.warning("Không thể sử dụng pytesseract. Thử sử dụng phương pháp khác.")
            return self._extract_text_with_easyocr(image_path)
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất văn bản từ ảnh {image_path}: {str(e)}")
            return ""

    def _extract_text_with_easyocr(self, image_path: str) -> str:
        """
        Trích xuất văn bản từ ảnh sử dụng EasyOCR

        Args:
            image_path: Đường dẫn đến file ảnh

        Returns:
            str: Văn bản đã trích xuất
        """
        try:
            import easyocr

            # Khởi tạo reader với ngôn ngữ Tiếng Việt và Tiếng Anh
            reader = easyocr.Reader(['vi', 'en'], gpu=self.use_gpu)

            # Thực hiện OCR
            results = reader.readtext(image_path)

            # Tổng hợp kết quả
            texts = [result[1] for result in results]

            return '\n'.join(texts)
        except ImportError:
            logger.warning("Không thể sử dụng easyocr. Thử sử dụng phương pháp khác.")
            return self._extract_text_with_llm_vision(image_path)
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất văn bản với EasyOCR: {str(e)}")
            return ""

    def _extract_text_with_llm_vision(self, image_path: str) -> str:
        """
        Trích xuất văn bản từ ảnh sử dụng mô hình LLM có khả năng nhìn

        Args:
            image_path: Đường dẫn đến file ảnh

        Returns:
            str: Văn bản đã trích xuất
        """
        try:
            # Thử sử dụng model local với khả năng vision
            from app.services.llm.vision_model import VisionModel

            vision_model = VisionModel()

            # Tạo prompt
            prompt = """
            Đây là một hình ảnh CV/Resume. Hãy trích xuất TẤT CẢ văn bản từ hình ảnh này.
            Chỉ trả về VĂN BẢN, không thêm bất kỳ chú thích hoặc mô tả nào.
            Giữ nguyên định dạng bố cục càng nhiều càng tốt.
            """

            # Trích xuất văn bản
            result = vision_model.generate_from_image(image_path, prompt)

            return result
        except ImportError:
            logger.warning("Không thể sử dụng mô hình vision. Quay lại phương pháp cơ bản.")
            return "Không thể trích xuất văn bản từ ảnh. Vui lòng cài đặt pytesseract hoặc easyocr."
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất văn bản với mô hình vision: {str(e)}")
            return ""

    def _preprocess_image(self, image):
        """
        Tiền xử lý ảnh để cải thiện kết quả OCR

        Args:
            image: Đối tượng ảnh PIL

        Returns:
            Đối tượng ảnh PIL đã được xử lý
        """
        try:
            from PIL import ImageEnhance, ImageFilter

            # Chuyển đổi sang ảnh xám nếu có màu
            if image.mode != 'L':
                image = image.convert('L')

            # Tăng độ tương phản
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)

            # Làm sắc nét
            image = image.filter(ImageFilter.SHARPEN)

            # Lọc nhiễu
            image = image.filter(ImageFilter.MedianFilter())

            return image
        except Exception as e:
            logger.error(f"Lỗi khi tiền xử lý ảnh: {str(e)}")
            return image

    def extract_info_with_ai(self, image_path: str) -> Dict[str, Any]:
        """
        Sử dụng AI để trích xuất thông tin có cấu trúc từ ảnh CV

        Args:
            image_path: Đường dẫn đến file ảnh

        Returns:
            Dict[str, Any]: Thông tin đã trích xuất
        """
        try:
            # Tạo tệp tạm thời nếu image_path không phải là tệp
            if not os.path.isfile(image_path):
                # Giả sử image_path có thể là dữ liệu base64
                if image_path.startswith('data:image'):
                    image_path = self._save_base64_to_file(image_path)
                else:
                    logger.error(f"Không thể nhận dạng định dạng ảnh: {image_path}")
                    return {}

            # Trích xuất văn bản từ ảnh
            text = self.extract_text(image_path)

            if not text:
                logger.warning(f"Không thể trích xuất văn bản từ ảnh {image_path}")
                return {}

            # Sử dụng LLM để phân tích văn bản
            from app.services.llm.local_model import LocalLLM

            llm = LocalLLM()

            prompt = f"""
            Dưới đây là văn bản trích xuất từ một CV/Resume. Hãy phân tích và trích xuất các thông tin quan trọng dưới dạng JSON.
            
            Văn bản CV:
            {text}
            
            Hãy trích xuất và trả về thông tin dưới dạng JSON với các trường sau:
            - personal_info: thông tin cá nhân (tên, email, điện thoại, địa chỉ)
            - education: học vấn (trường, chuyên ngành, thời gian học, bằng cấp)
            - work_experience: kinh nghiệm làm việc (công ty, vị trí, thời gian, mô tả công việc)
            - skills: kỹ năng (tên kỹ năng, mức độ thành thạo nếu có)
            - languages: ngôn ngữ (tên ngôn ngữ, mức độ thành thạo)
            - projects: dự án (tên dự án, mô tả, công nghệ sử dụng)
            - certificates: chứng chỉ (tên chứng chỉ, tổ chức cấp, thời gian)
            
            Chỉ trả về dữ liệu JSON hợp lệ, không thêm bất kỳ chú thích nào.
            """

            try:
                result = llm.generate(prompt)

                # Phân tích JSON
                import json
                return json.loads(result)
            except json.JSONDecodeError:
                logger.error(f"Không thể phân tích kết quả JSON: {result}")
                return {"raw_text": text}

        except Exception as e:
            logger.error(f"Lỗi khi trích xuất thông tin từ ảnh: {str(e)}")
            return {}

    def is_cv_image(self, image_path: str, confidence_threshold: float = 0.7) -> bool:
        """
        Kiểm tra xem một ảnh có phải là CV hay không

        Args:
            image_path: Đường dẫn đến file ảnh
            confidence_threshold: Ngưỡng độ tin cậy

        Returns:
            bool: True nếu ảnh là CV, False nếu không
        """
        try:
            # Trích xuất văn bản từ ảnh
            text = self.extract_text(image_path)

            if not text:
                return False

            # Xác định các từ khóa phổ biến trong CV
            cv_keywords = [
                "resume", "curriculum vitae", "cv", "sơ yếu lý lịch", "hồ sơ",
                "education", "học vấn", "experience", "kinh nghiệm",
                "skills", "kỹ năng", "employment", "việc làm", "contact", "liên hệ",
                "objective", "mục tiêu", "professional", "chuyên nghiệp",
                "qualification", "bằng cấp", "reference", "tham khảo",
                "project", "dự án", "achievement", "thành tựu"
            ]

            # Đếm số lượng từ khóa xuất hiện
            keyword_count = sum(1 for keyword in cv_keywords if keyword.lower() in text.lower())

            # Tính tỷ lệ từ khóa
            confidence = keyword_count / len(cv_keywords)

            return confidence >= confidence_threshold

        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra ảnh CV: {str(e)}")
            return False

    def _save_base64_to_file(self, base64_str: str) -> str:
        """
        Lưu chuỗi base64 thành tệp tin

        Args:
            base64_str: Chuỗi base64 của hình ảnh

        Returns:
            str: Đường dẫn đến tệp tin đã lưu
        """
        try:
            # Tạo tệp tạm thời
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_path = temp_file.name

            # Cắt bỏ phần header base64 nếu có
            if ',' in base64_str:
                base64_str = base64_str.split(',', 1)[1]

            # Giải mã base64 và lưu vào tệp
            with open(temp_path, 'wb') as f:
                f.write(base64.b64decode(base64_str))

            return temp_path
        except Exception as e:
            logger.error(f"Lỗi khi lưu chuỗi base64: {str(e)}")
            return ""

    def crop_image(self, image_path: str, output_path: str = None) -> str:
        """
        Cắt ảnh để chỉ giữ phần văn bản

        Args:
            image_path: Đường dẫn đến file ảnh
            output_path: Đường dẫn đầu ra. Nếu None, sẽ tạo tệp tạm thời

        Returns:
            str: Đường dẫn đến ảnh đã cắt
        """
        try:
            from PIL import Image
            import cv2
            import numpy as np

            # Đọc ảnh
            image = cv2.imread(image_path)

            # Chuyển sang ảnh xám
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Áp dụng ngưỡng
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Tìm contour
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Tìm contour lớn nhất
            areas = [cv2.contourArea(c) for c in contours]
            if not areas:
                return image_path

            max_index = np.argmax(areas)
            max_contour = contours[max_index]

            # Tạo bounding box
            x, y, w, h = cv2.boundingRect(max_contour)

            # Mở rộng vùng cắt để đảm bảo không mất thông tin
            padding = 50
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)

            # Cắt ảnh
            cropped = image[y:y+h, x:x+w]

            # Lưu ảnh
            if output_path is None:
                output_path = tempfile.mktemp(suffix='.png')

            cv2.imwrite(output_path, cropped)

            return output_path
        except Exception as e:
            logger.error(f"Lỗi khi cắt ảnh: {str(e)}")
            return image_path

    def get_image_quality(self, image_path: str) -> Dict[str, Any]:
        """
        Đánh giá chất lượng ảnh

        Args:
            image_path: Đường dẫn đến file ảnh

        Returns:
            Dict[str, Any]: Thông tin về chất lượng ảnh
        """
        try:
            from PIL import Image
            import cv2
            import numpy as np

            # Đọc ảnh với PIL
            pil_image = Image.open(image_path)

            # Kích thước ảnh
            width, height = pil_image.size

            # Đọc ảnh với OpenCV
            cv_image = cv2.imread(image_path)

            # Độ tương phản
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            contrast = gray.std()

            # Độ nhiễu
            noise = np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray)

            # Độ sắc nét
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian)

            # Đánh giá chất lượng OCR
            # Nếu ảnh quá nhỏ, độ tương phản thấp hoặc có nhiều nhiễu, OCR sẽ khó khăn
            ocr_quality = "good"
            if width < 1000 or height < 1000:
                ocr_quality = "low"
            elif contrast < 40:
                ocr_quality = "medium"
            elif noise > 10:
                ocr_quality = "medium"

            return {
                "width": width,
                "height": height,
                "contrast": float(contrast),
                "noise": float(noise),
                "sharpness": float(sharpness),
                "ocr_quality": ocr_quality,
                "format": pil_image.format,
                "mode": pil_image.mode,
                "dpi": pil_image.info.get('dpi', (72, 72))
            }
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá chất lượng ảnh: {str(e)}")
            return {
                "width": 0,
                "height": 0,
                "ocr_quality": "unknown",
                "error": str(e)
            }