"""
Script tự động cấu hình GPU cho ứng dụng
"""

import os
import sys
import subprocess
import logging
from typing import Dict, Any, Optional

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("gpu_setup")

def check_gpu() -> bool:
    """
    Kiểm tra xem GPU có khả dụng không

    Returns:
        bool: True nếu GPU khả dụng, False nếu không
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

def check_ram() -> float:
    """
    Kiểm tra dung lượng RAM hệ thống

    Returns:
        float: Dung lượng RAM (GB)
    """
    try:
        import psutil
        total_ram = psutil.virtual_memory().total / (1024 ** 3)
        available_ram = psutil.virtual_memory().available / (1024 ** 3)

        logger.info(f"RAM: {total_ram:.2f} GB (Available: {available_ram:.2f} GB)")

        return total_ram
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra RAM: {str(e)}")
        return 8.0  # Giả định 8GB nếu không kiểm tra được

def configure_gpu_memory_allocation(gpu_memory: float, ram: float) -> Dict[str, Any]:
    """
    Cấu hình phân bổ bộ nhớ GPU

    Args:
        gpu_memory: Dung lượng GPU (GB)
        ram: Dung lượng RAM (GB)

    Returns:
        Dict[str, Any]: Cấu hình bộ nhớ GPU
    """
    # Cấu hình mặc định
    config = {
        "pytorch_device_mem_fraction": 0.8,  # Phần trăm bộ nhớ GPU được sử dụng
        "use_fp16": False,                   # Sử dụng FP16
        "use_int8": False,                   # Sử dụng INT8
        "max_model_size": "small",           # Kích thước mô hình tối đa (small, medium, large)
        "cuda_visible_devices": "0",         # GPU được sử dụng
        "pytorch_cuda_alloc_conf": None      # Cấu hình phân bổ CUDA
    }

    # Dưới 4GB GPU hoặc 8GB RAM: mô hình nhỏ + INT8
    if gpu_memory < 4 or ram < 8:
        config["use_int8"] = True
        config["max_model_size"] = "small"
        config["pytorch_device_mem_fraction"] = 0.7
        config["pytorch_cuda_alloc_conf"] = "max_split_size_mb:128"

    # 4-8GB GPU: mô hình trung bình + FP16
    elif 4 <= gpu_memory < 8:
        config["use_fp16"] = True
        config["max_model_size"] = "medium"
        config["pytorch_device_mem_fraction"] = 0.8
        config["pytorch_cuda_alloc_conf"] = "max_split_size_mb:256"

    # 8GB+ GPU: mô hình lớn + FP16
    else:
        config["use_fp16"] = True
        config["max_model_size"] = "large"
        config["pytorch_device_mem_fraction"] = 0.9

    return config

def apply_gpu_configuration(config: Dict[str, Any]) -> None:
    """
    Áp dụng cấu hình GPU

    Args:
        config: Cấu hình GPU
    """
    # Thiết lập biến môi trường
    os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

    if config["pytorch_cuda_alloc_conf"]:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config["pytorch_cuda_alloc_conf"]

    # Ghi cấu hình vào tệp cấu hình
    config_path = os.path.join(os.path.dirname(__file__), "..", "core", "gpu_config.py")

    with open(config_path, "w") as f:
        f.write("\"\"\"Cấu hình GPU được tạo tự động\"\"\"\n\n")
        f.write("# Cấu hình GPU\n")
        f.write("GPU_CONFIG = {\n")

        for key, value in config.items():
            if isinstance(value, str):
                f.write(f"    \"{key}\": \"{value}\",\n")
            else:
                f.write(f"    \"{key}\": {value},\n")

        f.write("}\n")

    logger.info(f"Đã lưu cấu hình GPU vào {config_path}")

def test_gpu_configuration() -> bool:
    """
    Kiểm tra cấu hình GPU

    Returns:
        bool: True nếu kiểm tra thành công, False nếu không
    """
    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning("Không thể kiểm tra cấu hình GPU vì không có GPU")
            return False

        # Tạo tensor ngẫu nhiên và thực hiện phép tính trên GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)

        logger.info("Kiểm tra GPU thành công!")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra cấu hình GPU: {str(e)}")
        return False

def check_cuda_version() -> Optional[str]:
    """
    Kiểm tra phiên bản CUDA

    Returns:
        Optional[str]: Phiên bản CUDA hoặc None nếu không tìm thấy
    """
    try:
        # Kiểm tra phiên bản CUDA với nvcc
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)

        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line.lower() and "v" in line.lower():
                    version = line.lower().split("v")[1].split(" ")[0]
                    logger.info(f"Phiên bản CUDA: {version}")
                    return version

        # Kiểm tra phiên bản CUDA với nvidia-smi
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)

        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "cuda version" in line.lower():
                    version = line.lower().split("version:")[1].strip().split(" ")[0]
                    logger.info(f"Phiên bản CUDA: {version}")
                    return version

        logger.warning("Không thể xác định phiên bản CUDA")
        return None
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra phiên bản CUDA: {str(e)}")
        return None

def main():
    """Hàm chính"""
    logger.info("Bắt đầu cấu hình GPU...")

    # Kiểm tra GPU
    has_gpu = check_gpu()

    # Kiểm tra RAM
    ram = check_ram()

    # Kiểm tra phiên bản CUDA
    cuda_version = check_cuda_version()

    if has_gpu:
        # Cấu hình GPU
        gpu_memory = 0
        try:
            import torch
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            gpu_memory = 4.0  # Giả định 4GB nếu không kiểm tra được

        config = configure_gpu_memory_allocation(gpu_memory, ram)

        # Áp dụng cấu hình
        apply_gpu_configuration(config)

        # Kiểm tra cấu hình
        test_gpu_configuration()
    else:
        logger.warning("Không tìm thấy GPU. Sử dụng cấu hình CPU...")

        # Cấu hình CPU
        cpu_config = {
            "pytorch_device_mem_fraction": 0.0,
            "use_fp16": False,
            "use_int8": True,
            "max_model_size": "small",
            "cuda_visible_devices": "",
            "pytorch_cuda_alloc_conf": None,
            "cpu_threads": min(os.cpu_count() or 4, 8),  # Giới hạn số luồng CPU
            "cpu_only": True
        }

        # Áp dụng cấu hình CPU
        apply_gpu_configuration(cpu_config)

    logger.info("Hoàn tất cấu hình GPU!")

if __name__ == "__main__":
    main()