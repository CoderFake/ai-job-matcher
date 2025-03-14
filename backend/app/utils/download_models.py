import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import time

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("download_models")

# Thư mục lưu trữ mô hình
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")


# Tạo thư mục nếu chưa tồn tại
def ensure_model_dir():
    """Đảm bảo thư mục mô hình tồn tại"""
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Không thể tạo thư mục mô hình: {str(e)}")
        return False


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


def check_disk_space(required_mb: int) -> bool:
    """
    Kiểm tra không gian đĩa có đủ không

    Args:
        required_mb: Không gian cần thiết (MB)

    Returns:
        bool: True nếu có đủ không gian
    """
    try:
        import shutil

        # Lấy không gian khả dụng trong thư mục mô hình
        free_bytes = shutil.disk_usage(MODEL_DIR).free
        free_mb = free_bytes / (1024 * 1024)

        # Kiểm tra không gian
        if free_mb < required_mb:
            logger.warning(f"Không đủ không gian đĩa. Cần {required_mb}MB, có {free_mb:.2f}MB")
            return False

        return True
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra không gian đĩa: {str(e)}")
        # Giả sử có đủ không gian nếu không kiểm tra được
        return True


def check_gpu() -> bool:
    """
    Kiểm tra GPU có khả dụng không

    Returns:
        bool: True nếu GPU khả dụng
    """
    try:
        # Kiểm tra GPU với torch
        import torch
        has_cuda = torch.cuda.is_available()

        if has_cuda:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if gpu_count > 0 else 0

            logger.info(f"Tìm thấy {gpu_count} GPU:")
            logger.info(f"  - GPU 0: {gpu_name}")
            logger.info(f"  - Memory: {gpu_memory:.2f} GB")

            return True
        else:
            logger.warning("Không tìm thấy GPU với CUDA")
            return False
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra GPU: {str(e)}")
        return False


def download_transformers_model(model_name: str) -> bool:
    """
    Tải mô hình transformers

    Args:
        model_name: Tên mô hình

    Returns:
        bool: True nếu tải thành công
    """
    try:
        from transformers import AutoModel, AutoTokenizer
        import time

        logger.info(f"Đang tải mô hình {model_name} từ Hugging Face...")

        # Số lần thử lại tối đa
        max_retries = 3
        retry_delay = 5  # Giây

        for attempt in range(max_retries):
            try:
                # Tải tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.save_pretrained(os.path.join(MODEL_DIR, model_name.split('/')[-1]))

                # Tải mô hình
                model = AutoModel.from_pretrained(model_name)
                model.save_pretrained(os.path.join(MODEL_DIR, model_name.split('/')[-1]))

                logger.info(f"Đã tải mô hình {model_name} thành công!")
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Lỗi khi tải mô hình, thử lại sau {retry_delay} giây: {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Tăng thời gian chờ theo cấp số nhân
                else:
                    raise e

    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình {model_name}: {str(e)}")
        return False


def download_spacy_model(model_name: str) -> bool:
    """
    Tải mô hình spaCy

    Args:
        model_name: Tên mô hình

    Returns:
        bool: True nếu tải thành công
    """
    try:
        import spacy
        from spacy.cli import download
        import time

        logger.info(f"Đang tải mô hình spaCy {model_name}...")

        # Kiểm tra xem mô hình đã được tải chưa
        try:
            spacy.load(model_name)
            logger.info(f"Mô hình {model_name} đã được tải!")
            return True
        except OSError:
            # Nếu mô hình chưa được tải, tải xuống
            # Thêm cơ chế thử lại
            max_retries = 3
            retry_delay = 5  # Giây

            for attempt in range(max_retries):
                try:
                    download(model_name)
                    logger.info(f"Đã tải mô hình {model_name} thành công!")
                    return True
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Lỗi khi tải mô hình, thử lại sau {retry_delay} giây: {str(e)}")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise e

    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình spaCy {model_name}: {str(e)}")
        return False


def install_pip_package(package_name: str) -> bool:
    """
    Cài đặt gói pip với cơ chế thử lại

    Args:
        package_name: Tên gói

    Returns:
        bool: True nếu cài đặt thành công
    """
    try:
        logger.info(f"Đang cài đặt gói {package_name}...")

        # Thêm cơ chế thử lại
        max_retries = 3
        retry_delay = 5  # Giây

        for attempt in range(max_retries):
            try:
                # Cài đặt gói
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                logger.info(f"Đã cài đặt gói {package_name} thành công!")
                return True
            except subprocess.CalledProcessError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Lỗi khi cài đặt gói, thử lại sau {retry_delay} giây: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise e

    except Exception as e:
        logger.error(f"Lỗi khi cài đặt gói {package_name}: {str(e)}")
        return False


def download_ollama_model(model_name: str) -> bool:
    """
    Tải mô hình Ollama với cơ chế thử lại

    Args:
        model_name: Tên mô hình Ollama

    Returns:
        bool: True nếu tải thành công
    """
    try:
        logger.info(f"Đang tải mô hình Ollama {model_name}...")

        # Kiểm tra xem Ollama có được cài đặt không
        try:
            subprocess.check_call(["ollama", "list"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.error("Không tìm thấy lệnh Ollama. Vui lòng cài đặt Ollama trước.")
            return False

        # Thêm cơ chế thử lại
        max_retries = 3
        retry_delay = 10  # Giây

        for attempt in range(max_retries):
            try:
                # Tải mô hình
                subprocess.check_call(["ollama", "pull", model_name], stdout=subprocess.PIPE)

                logger.info(f"Đã tải mô hình Ollama {model_name} thành công!")
                return True
            except subprocess.CalledProcessError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Lỗi khi tải mô hình, thử lại sau {retry_delay} giây: {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise e

    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình Ollama {model_name}: {str(e)}")
        return False


def download_file(url: str, output_path: str) -> bool:
    """
    Tải tệp từ URL với cơ chế thử lại và hiển thị tiến trình

    Args:
        url: URL của tệp
        output_path: Đường dẫn đầu ra

    Returns:
        bool: True nếu tải thành công
    """
    try:
        import requests
        from tqdm import tqdm

        logger.info(f"Đang tải tệp từ {url}...")

        # Thêm cơ chế thử lại
        max_retries = 5
        retry_delay = 5  # Giây

        for attempt in range(max_retries):
            try:
                # Thực hiện yêu cầu streaming
                with requests.get(url, stream=True, timeout=30) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))

                    # Tải tệp tạm thời trước
                    temp_file = output_path + '.partial'

                    # Tạo thanh tiến trình
                    with open(temp_file, 'wb') as f, tqdm(
                            desc=os.path.basename(output_path),
                            total=total_size,
                            unit='B',
                            unit_scale=True,
                            unit_divisor=1024,
                    ) as bar:
                        for data in response.iter_content(chunk_size=8192):
                            size = f.write(data)
                            bar.update(size)

                    # Nếu tải thành công, đổi tên tệp
                    shutil.move(temp_file, output_path)

                    logger.info(f"Đã tải tệp thành công: {output_path}")
                    return True

            except (requests.exceptions.RequestException, IOError) as e:
                # Xóa tệp tạm nếu có lỗi
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

                if attempt < max_retries - 1:
                    logger.warning(f"Lỗi khi tải tệp, thử lại sau {retry_delay} giây: {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise e

    except Exception as e:
        logger.error(f"Lỗi khi tải tệp từ {url}: {str(e)}")
        return False


def check_model_exists(model_info: Dict[str, Any]) -> bool:
    """
    Kiểm tra xem mô hình đã tồn tại chưa

    Args:
        model_info: Thông tin mô hình

    Returns:
        bool: True nếu mô hình đã tồn tại
    """
    try:
        model_name = model_info["name"]
        model_type = model_info["type"]

        if model_type == "transformers":
            # Kiểm tra xem thư mục mô hình có tồn tại không
            model_path = os.path.join(MODEL_DIR, model_name.split('/')[-1])
            return os.path.exists(model_path) and os.path.isdir(model_path) and len(os.listdir(model_path)) > 0

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
                package_name = model_name.split('-')[0]
                __import__(package_name)
                return True
            except ImportError:
                return False

        elif model_type == "ollama":
            # Kiểm tra xem mô hình Ollama đã tồn tại chưa
            try:
                result = subprocess.check_output(["ollama", "list"], text=True)
                model_short_name = model_name.split('/')[-1]
                return model_short_name in result
            except (subprocess.SubprocessError, FileNotFoundError):
                return False

        elif model_type == "file":
            # Kiểm tra xem tệp đã tồn tại chưa
            model_path = os.path.join(MODEL_DIR, model_name)
            return os.path.exists(model_path) and os.path.getsize(model_path) > 0

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
        bool: True nếu tải thành công
    """
    model_name = model_info["name"]
    model_type = model_info["type"]
    model_size = model_info["size_mb"]

    # Kiểm tra xem mô hình đã tồn tại chưa
    if check_model_exists(model_info):
        logger.info(f"Mô hình {model_name} đã tồn tại!")
        return True

    # Kiểm tra không gian đĩa
    if not check_disk_space(model_size * 1.5):  # Thêm 50% dự phòng
        logger.error(f"Không đủ không gian đĩa để tải mô hình {model_name} ({model_size}MB)")
        return False

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
    """Hàm chính để tải xuống tất cả các mô hình cần thiết"""
    logger.info("Bắt đầu tải các mô hình AI...")

    # Đảm bảo thư mục mô hình tồn tại
    if not ensure_model_dir():
        logger.error("Không thể tạo thư mục mô hình. Thoát.")
        return False

    # Kiểm tra GPU
    has_gpu = check_gpu()

    if has_gpu:
        logger.info("Đã tìm thấy GPU. Tải các mô hình tương thích với GPU...")
    else:
        logger.info("Không tìm thấy GPU. Tải các mô hình tương thích với CPU...")

    # Tính tổng không gian đĩa cần thiết
    total_size_mb = sum(model["size_mb"] for model_id, model in MODELS.items() if model["required"])
    if not check_disk_space(total_size_mb * 1.5):  # Thêm 50% dự phòng
        logger.error(f"Không đủ không gian đĩa. Cần ít nhất {total_size_mb * 1.5:.2f}MB")
        return False

    # Tải các mô hình
    success_count = 0
    required_count = sum(1 for model in MODELS.values() if model["required"])

    for model_id, model_info in MODELS.items():
        if model_info["required"]:
            logger.info(
                f"Tải mô hình {model_id} ({model_info['name']}) - {model_info['use_case']} ({model_info['size_mb']} MB)")
            success = download_model(model_info)

            if success:
                logger.info(f"✅ Đã tải mô hình {model_id} thành công!")
                success_count += 1
            else:
                logger.error(f"❌ Không thể tải mô hình {model_id}!")
        else:
            logger.info(
                f"Bỏ qua mô hình tùy chọn {model_id} ({model_info['name']}) - {model_info['use_case']} ({model_info['size_mb']} MB)")

    if success_count >= required_count:
        logger.info("✅ Tải mô hình hoàn tất thành công!")
        return True
    else:
        logger.warning(f"⚠️ Đã tải {success_count}/{required_count} mô hình bắt buộc.")
        return False


if __name__ == "__main__":
    main()