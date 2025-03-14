"""
Module xử lý trích xuất thông tin từ file PDF
"""

import os
import re
import logging
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import PyPDF2
import pdfplumber
from app.core.logging import get_logger

logger = get_logger("cv_parser")


class PDFParser:
    """
    Lớp trích xuất dữ liệu từ file PDF
    """

    def __init__(self, optimize_ocr: bool = False):
        """
        Khởi tạo parser

        Args:
            optimize_ocr: Tối ưu hóa OCR nếu cần thiết
        """
        self.optimize_ocr = optimize_ocr

    def extract_text(self, file_path: str) -> str:
        """
        Trích xuất văn bản từ file PDF

        Args:
            file_path: Đường dẫn đến file PDF

        Returns:
            str: Văn bản đã trích xuất
        """
        try:
            # Thử sử dụng PyPDF2 trước
            text = self._extract_with_pypdf2(file_path)

            # Nếu text quá ngắn hoặc trống, thử dùng pdfplumber
            if len(text.strip()) < 100:
                logger.info(f"PyPDF2 không trích xuất được đủ text từ {file_path}, thử dùng pdfplumber")
                text = self._extract_with_pdfplumber(file_path)

            # Nếu vẫn không có text, thử OCR
            if len(text.strip()) < 100 and self.optimize_ocr:
                logger.info(f"Thử dùng OCR cho {file_path}")
                text = self._extract_with_ocr(file_path)

            return text
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất text từ PDF {file_path}: {str(e)}")
            return ""

    def _extract_with_pypdf2(self, file_path: str) -> str:
        """
        Trích xuất văn bản sử dụng PyPDF2

        Args:
            file_path: Đường dẫn đến file PDF

        Returns:
            str: Văn bản đã trích xuất
        """
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất với PyPDF2: {str(e)}")
            return ""

    def _extract_with_pdfplumber(self, file_path: str) -> str:
        """
        Trích xuất văn bản sử dụng pdfplumber

        Args:
            file_path: Đường dẫn đến file PDF

        Returns:
            str: Văn bản đã trích xuất
        """
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất với pdfplumber: {str(e)}")
            return ""

    def _extract_with_ocr(self, file_path: str) -> str:
        """
        Trích xuất văn bản sử dụng OCR (Optical Character Recognition)

        Args:
            file_path: Đường dẫn đến file PDF

        Returns:
            str: Văn bản đã trích xuất
        """
        try:
            # Thử import pytesseract và PIL nếu cần
            import pytesseract
            from PIL import Image
            from pdf2image import convert_from_path

            # Thiết lập thư mục tạm thời
            with tempfile.TemporaryDirectory() as temp_dir:
                # Chuyển đổi các trang PDF thành hình ảnh
                images = convert_from_path(file_path)

                text = ""
                for i, image in enumerate(images):
                    # Lưu hình ảnh tạm thời
                    temp_image_path = os.path.join(temp_dir, f'page_{i}.png')
                    image.save(temp_image_path, 'PNG')

                    # Nhận diện chữ từ hình ảnh
                    page_text = pytesseract.image_to_string(Image.open(temp_image_path), lang='vie+eng')
                    text += page_text + "\n"

                    # Xóa tệp tạm thời
                    os.remove(temp_image_path)

                return text
        except ImportError:
            logger.warning("Không thể sử dụng OCR vì thiếu các thư viện cần thiết")
            return ""
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất với OCR: {str(e)}")
            return ""

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Trích xuất metadata từ file PDF

        Args:
            file_path: Đường dẫn đến file PDF

        Returns:
            Dict[str, Any]: Metadata từ file PDF
        """
        metadata = {}
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                if reader.metadata:
                    for key, value in reader.metadata.items():
                        # Loại bỏ '/' từ key nếu có
                        clean_key = key.strip('/') if key.startswith('/') else key
                        metadata[clean_key] = value
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất metadata: {str(e)}")

        return metadata

    def extract_images(self, file_path: str, output_dir: Optional[str] = None) -> List[str]:
        """
        Trích xuất hình ảnh từ file PDF

        Args:
            file_path: Đường dẫn đến file PDF
            output_dir: Thư mục đầu ra cho các hình ảnh

        Returns:
            List[str]: Danh sách đường dẫn đến các hình ảnh đã trích xuất
        """
        image_paths = []

        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        else:
            os.makedirs(output_dir, exist_ok=True)

        try:
            # Sử dụng pdf2image để chuyển đổi các trang thành hình ảnh
            from pdf2image import convert_from_path

            images = convert_from_path(file_path)

            for i, image in enumerate(images):
                image_path = os.path.join(output_dir, f'page_{i}.png')
                image.save(image_path, 'PNG')
                image_paths.append(image_path)

        except ImportError:
            logger.warning("Không thể trích xuất hình ảnh vì thiếu thư viện pdf2image")
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất hình ảnh: {str(e)}")

        return image_paths

    def is_scanned_pdf(self, file_path: str) -> bool:
        """
        Kiểm tra nếu PDF là quét (scanned) hay sinh ra từ máy tính

        Args:
            file_path: Đường dẫn đến file PDF

        Returns:
            bool: True nếu PDF là quét, False nếu không
        """
        text = self._extract_with_pypdf2(file_path)

        # Nếu không có text hoặc rất ít text, có thể là PDF quét
        if len(text.strip()) < 100:
            return True

        return False

    def get_page_count(self, file_path: str) -> int:
        """
        Lấy số trang của file PDF

        Args:
            file_path: Đường dẫn đến file PDF

        Returns:
            int: Số trang
        """
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return len(reader.pages)
        except Exception as e:
            logger.error(f"Lỗi khi đếm số trang PDF: {str(e)}")
            return 0