"""
Module tương tác với mô hình vision để phân tích hình ảnh
"""

import os
import base64
import logging
import tempfile
from typing import Dict, Any, List, Optional, Union
import time
import json
from pathlib import Path

from app.core.logging import get_logger
from app.core.config import get_model_config

logger = get_logger("models")


class VisionModel:
    """
    Lớp tương tác với mô hình vision để phân tích hình ảnh
    """

    def __init__(self):
        """
        Khởi tạo mô hình vision
        """
        self.model_config = get_model_config()
        self.backend = None
        self._initialize_model()

    def _initialize_model(self):
        """
        Khởi tạo mô hình vision
        """
        try:
            # Thử sử dụng Ollama nếu có
            if self._is_ollama_vision_available():
                logger.info("Sử dụng Ollama để tương tác với mô hình vision")
                self.backend = "ollama"
                return

            # Thử sử dụng transformers và torch
            if self._is_transformers_vision_available():
                logger.info("Sử dụng transformers để tương tác với mô hình vision")
                self.backend = "transformers"
                return

            logger.warning("Không thể khởi tạo bất kỳ model vision nào. Sẽ sử dụng OCR thay thế.")
            self.backend = "ocr"

        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo mô hình vision: {str(e)}")
            self.backend = "ocr"

    def _is_ollama_vision_available(self) -> bool:
        """
        Kiểm tra xem Ollama có hỗ trợ mô hình vision không

        Returns:
            bool: True nếu có thể sử dụng Ollama vision
        """
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")

            if response.status_code == 200:
                # Lấy danh sách mô hình có sẵn
                available_models = response.json().get("models", [])

                # Kiểm tra mô hình có khả năng vision
                vision_models = ["llava", "bakllava", "llava-llama3"]

                for model in available_models:
                    model_name = model["name"].lower()
                    for vision_model in vision_models:
                        if vision_model in model_name:
                            self.model_name = model["name"]
                            logger.info(f"Tìm thấy mô hình vision: {self.model_name}")
                            return True

            return False
        except Exception as e:
            logger.warning(f"Không thể kết nối đến Ollama hoặc không tìm thấy mô hình vision: {str(e)}")
            return False

    def _is_transformers_vision_available(self) -> bool:
        """
        Kiểm tra xem có thể sử dụng transformers cho vision model không

        Returns:
            bool: True nếu có thể sử dụng transformers vision
        """
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM

            # Chỉ khởi tạo nếu có GPU
            if not torch.cuda.is_available():
                logger.warning("Không sử dụng transformers vision vì không có GPU")
                return False

            # Mô hình vision nhỏ
            vision_models = ["llava-hf/llava-1.5-7b-hf", "Efficient-Large-Model/VILA-7B"]

            for model_name in vision_models:
                try:
                    # Chỉ kiểm tra, không tải mô hình thực tế
                    self.processor = AutoProcessor.from_pretrained(model_name)
                    self.vision_model_name = model_name
                    logger.info(f"Tìm thấy mô hình vision transformers: {model_name}")
                    return True
                except Exception:
                    continue

            return False
        except ImportError:
            logger.warning("Không thể import transformers hoặc torch")
            return False
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra transformers vision: {str(e)}")
            return False

    def generate_from_image(self, image_path: str, prompt: str) -> str:
        """
        Tạo văn bản dựa trên hình ảnh và prompt

        Args:
            image_path: Đường dẫn đến hình ảnh
            prompt: Lời nhắc

        Returns:
            str: Văn bản được tạo
        """
        try:
            if self.backend == "ollama":
                return self._generate_with_ollama_vision(image_path, prompt)
            elif self.backend == "transformers":
                return self._generate_with_transformers_vision(image_path, prompt)
            else:
                # Fallback sang OCR
                return self._generate_with_ocr(image_path)

        except Exception as e:
            logger.error(f"Lỗi khi tạo văn bản từ hình ảnh: {str(e)}")
            return f"Lỗi khi phân tích hình ảnh: {str(e)}"

    def _generate_with_ollama_vision(self, image_path: str, prompt: str) -> str:
        """
        Tạo văn bản sử dụng Ollama vision

        Args:
            image_path: Đường dẫn đến hình ảnh
            prompt: Lời nhắc

        Returns:
            str: Văn bản được tạo
        """
        try:
            import requests

            # Đọc hình ảnh dưới dạng base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # Chuẩn bị tin nhắn
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image",
                            "image": image_data
                        }
                    ]
                }
            ]

            # Chuẩn bị dữ liệu yêu cầu
            data = {
                "model": self.model_name,
                "messages": messages,
                "stream": False
            }

            # Gửi yêu cầu
            response = requests.post("http://localhost:11434/api/chat", json=data)

            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                logger.error(f"Lỗi khi gọi Ollama vision API: {response.status_code} - {response.text}")
                return f"Lỗi: {response.status_code} - {response.text}"

        except Exception as e:
            logger.error(f"Lỗi khi tạo văn bản với Ollama vision: {str(e)}")
            return f"Lỗi khi tạo văn bản với Ollama vision: {str(e)}"

    def _generate_with_transformers_vision(self, image_path: str, prompt: str) -> str:
        """
        Tạo văn bản sử dụng transformers vision

        Args:
            image_path: Đường dẫn đến hình ảnh
            prompt: Lời nhắc

        Returns:
            str: Văn bản được tạo
        """
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM
            from PIL import Image

            # Tải mô hình nếu chưa tải
            if not hasattr(self, "model"):
                # Sử dụng 4-bit quantization để tiết kiệm VRAM
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.vision_model_name,
                    device_map="auto",
                    quantization_config=quantization_config
                )

            # Đọc hình ảnh
            image = Image.open(image_path)

            # Chuẩn bị đầu vào
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(device=torch.device("cuda"))

            # Tạo văn bản
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False
                )

            # Giải mã văn bản
            generated_text = self.processor.decode(output[0], skip_special_tokens=True)

            # Lấy phần phản hồi (loại bỏ prompt)
            response = generated_text.split(prompt)[-1].strip()

            return response

        except Exception as e:
            logger.error(f"Lỗi khi tạo văn bản với transformers vision: {str(e)}")
            return self._generate_with_ocr(image_path)

    def _generate_with_ocr(self, image_path: str) -> str:
        """
        Trích xuất văn bản từ hình ảnh sử dụng OCR

        Args:
            image_path: Đường dẫn đến hình ảnh

        Returns:
            str: Văn bản được trích xuất
        """
        try:
            # Thử sử dụng pytesseract trước
            import pytesseract
            from PIL import Image
            from PIL import ImageEnhance

            # Đọc và tiền xử lý ảnh
            image = Image.open(image_path)

            # Chuyển sang ảnh xám
            if image.mode != 'L':
                image = image.convert('L')

            # Tăng độ tương phản
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)

            # Trích xuất văn bản
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(image, lang='vie+eng', config=custom_config)

            if text.strip():
                return text

            # Nếu pytesseract không hoạt động, thử EasyOCR
            return self._ocr_with_easyocr(image_path)

        except ImportError:
            logger.warning("Không thể sử dụng pytesseract. Thử EasyOCR.")
            return self._ocr_with_easyocr(image_path)
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất văn bản với OCR: {str(e)}")
            return f"Không thể trích xuất văn bản từ hình ảnh: {str(e)}"

    def _ocr_with_easyocr(self, image_path: str) -> str:
        """
        Trích xuất văn bản từ hình ảnh sử dụng EasyOCR

        Args:
            image_path: Đường dẫn đến hình ảnh

        Returns:
            str: Văn bản được trích xuất
        """
        try:
            import easyocr

            # Khởi tạo reader với ngôn ngữ Tiếng Việt và Tiếng Anh
            reader = easyocr.Reader(['vi', 'en'], gpu=self.model_config.get('device') == 'cuda')

            # Trích xuất văn bản
            results = reader.readtext(image_path)

            # Tổng hợp kết quả
            texts = [result[1] for result in results]

            return '\n'.join(texts)

        except ImportError:
            logger.warning("Không thể sử dụng EasyOCR.")
            return "Không thể trích xuất văn bản từ hình ảnh vì thiếu các thư viện OCR cần thiết."
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất văn bản với EasyOCR: {str(e)}")
            return f"Không thể trích xuất văn bản từ hình ảnh: {str(e)}"

    def extract_structured_info(self, image_path: str) -> Dict[str, Any]:
        """
        Trích xuất thông tin có cấu trúc từ hình ảnh CV

        Args:
            image_path: Đường dẫn đến hình ảnh

        Returns:
            Dict[str, Any]: Thông tin được trích xuất
        """
        try:
            # Trích xuất văn bản từ hình ảnh
            if self.backend in ["ollama", "transformers"]:
                prompt = """
                Đây là một hình ảnh CV/Resume. Hãy trích xuất các thông tin sau dưới định dạng JSON:
                - personal_info: thông tin cá nhân (tên, email, điện thoại, địa chỉ)
                - education: học vấn (trường, chuyên ngành, thời gian học)
                - work_experience: kinh nghiệm làm việc (công ty, vị trí, thời gian)
                - skills: kỹ năng
                - languages: ngôn ngữ

                Chỉ trả về dữ liệu JSON.
                """

                text = self.generate_from_image(image_path, prompt)
            else:
                text = self._generate_with_ocr(image_path)

            # Sử dụng LLM để phân tích văn bản
            from app.services.llm.local_model import LocalLLM

            llm = LocalLLM()

            prompt = f"""
            Dưới đây là văn bản trích xuất từ một CV/Resume. Hãy phân tích và trích xuất các thông tin quan trọng dưới dạng JSON.

            Văn bản CV:
            {text}

            Hãy trích xuất và trả về thông tin dưới dạng JSON với các trường sau:
            {
            "personal_info": {
            "name": "Tên người",
                    "email": "email@example.com",
                    "phone": "Số điện thoại",
                    "address": "Địa chỉ"
                },
                "education": [
                    {
            "institution": "Tên trường",
                        "degree": "Bằng cấp",
                        "field_of_study": "Ngành học",
                        "start_date": "Ngày bắt đầu",
                        "end_date": "Ngày kết thúc"
                    }
                ],
                "work_experience": [
                    {
            "company": "Tên công ty",
                        "position": "Vị trí",
                        "start_date": "Ngày bắt đầu",
                        "end_date": "Ngày kết thúc",
                        "description": "Mô tả công việc"
                    }
                ],
                "skills": [
                    {"name": "Tên kỹ năng"}
                ],
                "languages": [
                    {"name": "Tên ngôn ngữ", "proficiency": "Mức độ thành thạo"}
                ]
            }

            Chỉ trả về dữ liệu JSON hợp lệ, không thêm bất kỳ chú thích nào.
            """

            result = llm.generate(prompt)

            # Tìm và trích xuất JSON từ kết quả
            import re
            import json

            # Tìm chuỗi JSON trong văn bản
            json_match = re.search(r'\{.*\}', result, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
                try:
                    data = json.loads(json_str)
                    return data
                except json.JSONDecodeError:
                    logger.error(f"Không thể phân tích JSON: {json_str}")

            # Nếu không tìm thấy JSON, trả về thông tin cơ bản
            return {
                "personal_info": {"name": ""},
                "raw_text": text
            }

        except Exception as e:
            logger.error(f"Lỗi khi trích xuất thông tin có cấu trúc từ ảnh: {str(e)}")
            return {"error": str(e)}

    def detect_document_type(self, image_path: str) -> str:
        """
        Phát hiện loại tài liệu từ hình ảnh

        Args:
            image_path: Đường dẫn đến hình ảnh

        Returns:
            str: Loại tài liệu (cv, id_card, passport, certificate, other)
        """
        try:
            if self.backend in ["ollama", "transformers"]:
                prompt = """
                Đây là hình ảnh của tài liệu gì? Hãy phân loại vào một trong các loại sau:
                - cv (nếu là CV/Resume)
                - id_card (nếu là thẻ căn cước/CMND)
                - passport (nếu là hộ chiếu)
                - certificate (nếu là chứng chỉ/bằng cấp)
                - other (nếu là loại tài liệu khác)

                Chỉ trả về một từ duy nhất từ danh sách trên.
                """

                result = self.generate_from_image(image_path, prompt).strip().lower()

                # Chuẩn hóa kết quả
                document_types = {"cv", "id_card", "passport", "certificate", "other"}

                if result in document_types:
                    return result
                elif "cv" in result or "resume" in result:
                    return "cv"
                elif "id" in result or "card" in result:
                    return "id_card"
                elif "passport" in result or "hộ chiếu" in result:
                    return "passport"
                elif "certificate" in result or "chứng chỉ" in result or "bằng" in result:
                    return "certificate"
                else:
                    return "other"
            else:
                # Nếu không có khả năng vision, thử phân tích bằng OCR + LLM
                text = self._generate_with_ocr(image_path)

                from app.services.llm.local_model import LocalLLM
                llm = LocalLLM()

                prompt = f"""
                Dưới đây là văn bản trích xuất từ một hình ảnh. Hãy phân loại văn bản này vào một trong các loại tài liệu sau:
                - cv (nếu là CV/Resume)
                - id_card (nếu là thẻ căn cước/CMND)
                - passport (nếu là hộ chiếu)
                - certificate (nếu là chứng chỉ/bằng cấp)
                - other (nếu là loại tài liệu khác)

                Văn bản:
                {text}

                Chỉ trả về một từ duy nhất từ danh sách trên.
                """

                result = llm.generate(prompt).strip().lower()

                # Chuẩn hóa kết quả
                document_types = {"cv", "id_card", "passport", "certificate", "other"}

                if result in document_types:
                    return result
                elif "cv" in result or "resume" in result:
                    return "cv"
                elif "id" in result or "card" in result:
                    return "id_card"
                elif "passport" in result or "hộ chiếu" in result:
                    return "passport"
                elif "certificate" in result or "chứng chỉ" in result or "bằng" in result:
                    return "certificate"
                else:
                    return "other"

        except Exception as e:
            logger.error(f"Lỗi khi phát hiện loại tài liệu: {str(e)}")
            return "other"

    def improve_image_for_ocr(self, image_path: str, output_path: Optional[str] = None) -> str:
        """
        Cải thiện chất lượng hình ảnh cho OCR

        Args:
            image_path: Đường dẫn đến hình ảnh
            output_path: Đường dẫn đầu ra. Nếu None, sẽ tạo tệp tạm thời

        Returns:
            str: Đường dẫn đến hình ảnh đã cải thiện
        """
        try:
            from PIL import Image, ImageEnhance, ImageFilter

            # Tạo đường dẫn đầu ra nếu chưa có
            if output_path is None:
                output_path = tempfile.mktemp(suffix='.png')

            # Đọc hình ảnh
            image = Image.open(image_path)

            # Chuyển sang ảnh xám
            image = image.convert('L')

            # Tăng độ tương phản
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)

            # Làm sắc nét
            image = image.filter(ImageFilter.SHARPEN)

            # Lọc nhiễu
            image = image.filter(ImageFilter.MedianFilter())

            # Lưu hình ảnh
            image.save(output_path)

            return output_path

        except Exception as e:
            logger.error(f"Lỗi khi cải thiện hình ảnh: {str(e)}")
            return image_path

    def extract_tables_from_image(self, image_path: str) -> List[List[List[str]]]:
        """
        Trích xuất bảng từ hình ảnh

        Args:
            image_path: Đường dẫn đến hình ảnh

        Returns:
            List[List[List[str]]]: Danh sách các bảng, mỗi bảng là một danh sách các hàng, mỗi hàng là một danh sách các ô
        """
        try:
            # Thử sử dụng OpenCV để phát hiện bảng
            import cv2
            import numpy as np

            # Đọc hình ảnh
            image = cv2.imread(image_path)

            # Chuyển sang ảnh xám
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Threshold
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

            # Tìm contour
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Lọc contour có hình dạng giống bảng
            tables = []

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 10000:  # Lọc theo diện tích
                    x, y, w, h = cv2.boundingRect(contour)

                    # Cắt vùng bảng
                    table_image = image[y:y + h, x:x + w]

                    # Lưu vùng bảng vào tệp tạm thời
                    table_path = tempfile.mktemp(suffix='.png')
                    cv2.imwrite(table_path, table_image)

                    # Trích xuất văn bản từ bảng
                    table_text = self._generate_with_ocr(table_path)

                    # Phân tích cấu trúc bảng
                    table_data = self._parse_table_structure(table_text)

                    if table_data:
                        tables.append(table_data)

                    # Xóa tệp tạm thời
                    os.unlink(table_path)

            if tables:
                return tables

            # Nếu không phát hiện được bảng, thử phương pháp khác
            return self._extract_tables_with_vision(image_path)

        except Exception as e:
            logger.error(f"Lỗi khi trích xuất bảng: {str(e)}")
            return []

    def _parse_table_structure(self, table_text: str) -> List[List[str]]:
        """
        Phân tích cấu trúc bảng từ văn bản

        Args:
            table_text: Văn bản bảng

        Returns:
            List[List[str]]: Bảng đã được phân tích
        """
        try:
            # Tách các dòng
            lines = [line.strip() for line in table_text.split('\n') if line.strip()]

            # Tạo cấu trúc bảng
            table = []

            for line in lines:
                # Phân tách ô trong dòng
                # Đây là phương pháp đơn giản, có thể cần cải tiến tùy vào định dạng bảng
                cells = [cell.strip() for cell in line.split('|')]

                # Lọc bỏ ô trống
                cells = [cell for cell in cells if cell]

                if cells:
                    table.append(cells)

            return table

        except Exception as e:
            logger.error(f"Lỗi khi phân tích cấu trúc bảng: {str(e)}")
            return []

    def _extract_tables_with_vision(self, image_path: str) -> List[List[List[str]]]:
        """
        Trích xuất bảng từ hình ảnh sử dụng mô hình vision

        Args:
            image_path: Đường dẫn đến hình ảnh

        Returns:
            List[List[List[str]]]: Danh sách các bảng
        """
        try:
            if self.backend in ["ollama", "transformers"]:
                prompt = """
                Đây là hình ảnh có chứa bảng. Hãy trích xuất nội dung của bảng và trả về dưới dạng JSON.
                Mỗi bảng là một mảng các hàng, mỗi hàng là một mảng các ô.
                Ví dụ:
                [
                    [["Tiêu đề 1", "Tiêu đề 2"], ["Giá trị 1", "Giá trị 2"]]
                ]
                """

                result = self.generate_from_image(image_path, prompt)

                # Tìm và trích xuất JSON từ kết quả
                import re
                import json

                # Tìm chuỗi JSON trong văn bản
                json_match = re.search(r'\[.*\]', result, re.DOTALL)

                if json_match:
                    json_str = json_match.group(0)
                    try:
                        data = json.loads(json_str)
                        return data
                    except json.JSONDecodeError:
                        logger.error(f"Không thể phân tích JSON: {json_str}")

            # Nếu không thành công, trả về danh sách trống
            return []

        except Exception as e:
            logger.error(f"Lỗi khi trích xuất bảng với vision: {str(e)}")
            return []