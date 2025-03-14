import os
import sys
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import subprocess

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("download_models")

# Thư mục lưu trữ mô hình
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Danh sách mô hình cần tải
MODELS = {
    "embedding": {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2",
        "type": "transformers",
        "use_case": "Embedding văn bản đa ngôn ngữ",
        "size_mb": 120,
        "required": True
    },
    "spacy_vi": {
        "name": "vi_core_news_md",
        "type": "spacy",
        "use_case": "Xử lý ngôn ngữ tự nhiên tiếng Việt",
        "size_mb": 40,
        "required": True
    },
    "onnx_runtime": {
        "name": "onnxruntime-gpu" if os.environ.get("CUDA_VISIBLE_DEVICES") else "onnxruntime",
        "type": "pip",
        "use_case": "Tối ưu hóa việc chạy mô hình",
        "size_mb": 80,
        "required": False
    },
    "ollama": {
        "name": "ollama/gemma:2b-instruct-q4_K_M",
        "type": "ollama",
        "use_case": "Mô hình ngôn ngữ nhỏ cho máy tính yếu",
        "size_mb": 1300,
        "required": False
    },
    "llama_cpp": {
        "name": "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "type": "file",
        "use_case": "Mô hình ngôn ngữ siêu nhẹ",
        "size_mb": 700,
        "required": False
    }
}


def check_gpu() -> bool:
    try:
        # Kiểm tra GPU với torch
        import torch
        return torch.cuda.is_available()
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra GPU: {str(e)}")
        return False


def download_transformers_model(model_name: str) -> bool:
    try:
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Đang tải mô hình {model_name} từ Hugging Face...")

        # Tải tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(os.path.join(MODEL_DIR, model_name.split('/')[-1]))

        # Tải mô hình
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(os.path.join(MODEL_DIR, model_name.split('/')[-1]))

        logger.info(f"Đã tải mô hình {model_name} thành công!")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình {model_name}: {str(e)}")
        return False


def download_spacy_model(model_name: str) -> bool:
    """
    Tải mô hình spaCy

    Args:
        model_name: Tên mô hình

    Returns:
        bool: True nếu tải thành công, False nếu không
    """
    try:
        import spacy
        from spacy.cli import download

        logger.info(f"Đang tải mô hình spaCy {model_name}...")

        # Kiểm tra xem mô hình đã được tải chưa
        try:
            spacy.load(model_name)
            logger.info(f"Mô hình {model_name} đã được tải!")
            return True
        except OSError:
            # Nếu mô hình chưa được tải, tải xuống
            download(model_name)
            logger.info(f"Đã tải mô hình {model_name} thành công!")
            return True
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình spaCy {model_name}: {str(e)}")
        return False


def install_pip_package(package_name: str) -> bool:
    """
    Cài đặt gói pip

    Args:
        package_name: Tên gói

    Returns:
        bool: True nếu cài đặt thành công, False nếu không
    """
    try:
        logger.info(f"Đang cài đặt gói {package_name}...")

        # Cài đặt gói
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

        logger.info(f"Đã cài đặt gói {package_name} thành công!")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi cài đặt gói {package_name}: {str(e)}")
        return False


def download_ollama_model(model_name: str) -> bool:
    """
    Tải mô hình Ollama

    Args:
        model_name: Tên mô hình Ollama

    Returns:
        bool: True nếu tải thành công, False nếu không
    """
    try:
        logger.info(f"Đang tải mô hình Ollama {model_name}...")

        # Kiểm tra xem Ollama có được cài đặt không
        try:
            subprocess.check_call(["ollama", "list"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            logger.error("Không tìm thấy lệnh Ollama. Vui lòng cài đặt Ollama trước.")
            return False

        # Tải mô hình
        subprocess.check_call(["ollama", "pull", model_name], stdout=subprocess.PIPE)

        logger.info(f"Đã tải mô hình Ollama {model_name} thành công!")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình Ollama {model_name}: {str(e)}")
        return False


def download_file(url: str, output_path: str) -> bool:
    """
    Tải tệp từ URL

    Args:
        url: URL của tệp
        output_path: Đường dẫn đầu ra

    Returns:
        bool: True nếu tải thành công, False nếu không
    """
    try:
        import requests
        from tqdm import tqdm

        logger.info(f"Đang tải tệp từ {url}...")

        # Thực hiện yêu cầu streaming
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        # Tạo thanh tiến trình
        with open(output_path, 'wb') as f, tqdm(
                desc=os.path.basename(output_path),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)

        logger.info(f"Đã tải tệp thành công: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi tải tệp từ {url}: {str(e)}")
        return False


def check_model_exists(model_info: Dict[str, Any]) -> bool:
    """
    Kiểm tra xem mô hình đã tồn tại chưa

    Args:
        model_info: Thông tin mô hình

    Returns:
        bool: True nếu mô hình đã tồn tại, False nếu không
    """
    try:
        model_name = model_info["name"]
        model_type = model_info["type"]

        if model_type == "transformers":
            # Kiểm tra xem thư mục mô hình có tồn tại không
            model_path = os.path.join(MODEL_DIR, model_name.split('/')[-1])
            return os.path.exists(model_path)
        elif model_type == "spacy":
            # Kiểm tra xem mô hình spaCy đã được tải chưa
            import spacy
            try:
                spacy.load(model_name)
                return True
            except OSError:
                return False
        elif model_type == "pip":
            # Kiểm tra xem gói đã được cài đặt chưa
            try:
                __import__(model_name.split('-')[0])
                return True
            except ImportError:
                return False
        elif model_type == "ollama":
            # Kiểm tra xem mô hình Ollama đã tồn tại chưa
            try:
                result = subprocess.check_output(["ollama", "list"], text=True)
                model_short_name = model_name.split('/')[-1]
                return model_short_name in result
            except Exception:
                return False
        elif model_type == "file":
            # Kiểm tra xem tệp đã tồn tại chưa
            model_path = os.path.join(MODEL_DIR, model_name)
            return os.path.exists(model_path)
        else:
            return False
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra mô hình: {str(e)}")
        return False


def download_model(model_info: Dict[str, Any]) -> bool:
    """
    Tải mô hình dựa trên thông tin

    Args:
        model_info: Thông tin mô hình

    Returns:
        bool: True nếu tải thành công, False nếu không
    """
    model_name = model_info["name"]
    model_type = model_info["type"]

    # Kiểm tra xem mô hình đã tồn tại chưa
    if check_model_exists(model_info):
        logger.info(f"Mô hình {model_name} đã tồn tại!")
        return True

    # Tải mô hình
    if model_type == "transformers":
        return download_transformers_model(model_name)
    elif model_type == "spacy":
        return download_spacy_model(model_name)
    elif model_type == "pip":
        return install_pip_package(model_name)
    elif model_type == "ollama":
        return download_ollama_model(model_name)
    elif model_type == "file":
        model_path = os.path.join(MODEL_DIR, model_name)
        return download_file(model_info["url"], model_path)
    else:
        logger.error(f"Không hỗ trợ loại mô hình: {model_type}")
        return False


def main():
    """Hàm chính"""
    logger.info("Bắt đầu tải các mô hình AI...")

    # Kiểm tra GPU
    has_gpu = check_gpu()

    if has_gpu:
        logger.info("Đã tìm thấy GPU. Tải các mô hình tương thích với GPU...")
    else:
        logger.info("Không tìm thấy GPU. Tải các mô hình tương thích với CPU...")

    # Tải các mô hình
    for model_id, model_info in MODELS.items():
        # Nếu mô hình là bắt buộc hoặc người dùng chọn tải tất cả
        if model_info["required"]:
            logger.info(
                f"Tải mô hình {model_id} ({model_info['name']}) - {model_info['use_case']} ({model_info['size_mb']} MB)")
            success = download_model(model_info)

            if success:
                logger.info(f"✅ Đã tải mô hình {model_id} thành công!")
            else:
                logger.error(f"❌ Không thể tải mô hình {model_id}!")
        else:
            logger.info(
                f"Bỏ qua mô hình tùy chọn {model_id} ({model_info['name']}) - {model_info['use_case']} ({model_info['size_mb']} MB)")

    logger.info("Quá trình tải mô hình đã hoàn tất!")

if __name__ == "__main__":
    main()