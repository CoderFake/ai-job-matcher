import os
import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import psutil
import json

from app.core.settings import settings
from app.core.logging import get_logger

logger = get_logger("config")

def configure_gpu():
    """
    Cấu hình GPU tự động dựa trên phần cứng có sẵn
    Tối ưu hóa cho máy tính 8GB RAM
    """
    if not settings.AUTO_CONFIG_GPU:
        logger.info("Bỏ qua cấu hình GPU tự động (AUTO_CONFIG_GPU=False)")
        return

    try:
        # Kiểm tra GPU có khả dụng không
        if not torch.cuda.is_available():
            logger.info("GPU không khả dụng, sử dụng CPU.")
            settings.USE_GPU = False
            _save_gpu_config({"device": "cpu", "use_gpu": False})
            return

        # Kiểm tra VRAM của GPU
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024  # Chuyển đổi sang GB

        # Kiểm tra RAM hệ thống
        ram_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024  # Chuyển đổi sang GB

        logger.info(f"Phát hiện GPU với {gpu_memory:.2f}GB VRAM")
        logger.info(f"RAM hệ thống: {ram_gb:.2f}GB")

        # Cấu hình dựa trên phần cứng
        settings.USE_GPU = True
        gpu_config = {"device": "cuda", "use_gpu": True}

        # Nếu VRAM ít, sử dụng cấu hình thấp
        if gpu_memory < 4:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            gpu_config["max_split_size_mb"] = 128
            logger.info("Áp dụng cấu hình cho GPU VRAM thấp (<4GB)")

        # Sử dụng half precision nếu GPU hỗ trợ
        if torch.cuda.is_bf16_supported():
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            gpu_config["precision"] = "bf16"
            logger.info("Sử dụng BF16 precision")
        elif torch.cuda.is_fp16_supported():
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            gpu_config["precision"] = "fp16"
            logger.info("Sử dụng FP16 precision")
        else:
            gpu_config["precision"] = "fp32"
            logger.info("Sử dụng FP32 precision")

        # Giới hạn bộ nhớ đệm PyTorch
        if ram_gb <= 8:
            # Giới hạn bộ nhớ đệm cho máy tính 8GB RAM
            torch.cuda.empty_cache()
            # Cài đặt giới hạn bộ nhớ đệm ở mức thấp
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
            gpu_config["max_split_size_mb"] = 64
            logger.info("Đã áp dụng cấu hình cho máy tính RAM thấp (8GB hoặc ít hơn)")

        # Lưu cấu hình GPU
        _save_gpu_config(gpu_config)

    except Exception as e:
        logger.error(f"Lỗi khi cấu hình GPU: {e}")
        settings.USE_GPU = False
        _save_gpu_config({"device": "cpu", "use_gpu": False, "error": str(e)})


def _save_gpu_config(config: Dict[str, Any]):
    """
    Lưu cấu hình GPU vào tệp để tái sử dụng

    Args:
        config: Cấu hình GPU
    """
    try:
        config_path = Path(settings.MODEL_DIR) / "gpu_config.json"
        os.makedirs(settings.MODEL_DIR, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        logger.error(f"Không thể lưu cấu hình GPU: {e}")


def load_gpu_config() -> Dict[str, Any]:
    """
    Đọc cấu hình GPU từ tệp nếu có

    Returns:
        Dict[str, Any]: Cấu hình GPU
    """
    default_config = {"device": "cpu", "use_gpu": False}
    config_path = Path(settings.MODEL_DIR) / "gpu_config.json"

    try:
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        return default_config
    except Exception as e:
        logger.error(f"Không thể đọc cấu hình GPU: {e}")
        return default_config


def get_model_config() -> Dict[str, Any]:
    """
    Trả về cấu hình cho các mô hình dựa trên phần cứng có sẵn

    Returns:
        Dict[str, Any]: Cấu hình cho các mô hình
    """
    # Đọc cấu hình GPU
    gpu_config = load_gpu_config()
    device = gpu_config.get("device", "cpu")

    # Thử chuyển sang CPU nếu GPU không khả dụng
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("GPU được cấu hình nhưng không khả dụng. Chuyển sang CPU.")
        device = "cpu"

    # Cấu hình cơ bản cho tất cả các mô hình
    base_config = {
        "device": device,
        "model_dir": settings.MODEL_DIR,
    }

    # Cấu hình cụ thể cho từng loại mô hình
    model_configs = {
        "sentence_transformer": {
            **base_config,
            "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
            "max_seq_length": 256,
            "fallback_model_name": "paraphrase-multilingual-MiniLM-L6-v2",
        },
        "named_entity_recognition": {
            **base_config,
            "model_name": "dslim/bert-base-NER-uncased",
            "fallback_model_name": "dslim/bert-small-NER-uncased",
        },
        "text_classification": {
            **base_config,
            "model_name": "distilbert-base-uncased",
            "fallback_model_name": "distilbert-base-uncased-finetuned-sst-2-english",
        },
        "spacy": {
            "model_name": "vi_core_news_lg",  # Mô hình tiếng Việt
            "fallback_model_name": "vi_core_news_md",  # Mô hình dự phòng
        }
    }

    # Nếu RAM thấp, sử dụng mô hình nhỏ hơn
    ram_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
    if ram_gb <= 8:
        logger.info("Sử dụng mô hình nhỏ do RAM thấp (≤8GB)")
        for model_type, config in model_configs.items():
            if "fallback_model_name" in config:
                config["model_name"] = config["fallback_model_name"]
                logger.info(f"Sử dụng mô hình {model_type} nhỏ hơn: {config['model_name']}")

    return model_configs


def check_disk_space(min_space_gb: float = 1.0) -> bool:
    """
    Kiểm tra không gian đĩa khả dụng

    Args:
        min_space_gb: Dung lượng tối thiểu cần thiết (GB)

    Returns:
        bool: True nếu đủ không gian
    """
    try:
        # Kiểm tra không gian đĩa trên thư mục mô hình
        disk_usage = psutil.disk_usage(settings.MODEL_DIR)
        free_space_gb = disk_usage.free / (1024 * 1024 * 1024)

        if free_space_gb < min_space_gb:
            logger.warning(f"Không đủ không gian đĩa! Cần ít nhất {min_space_gb}GB, hiện có {free_space_gb:.2f}GB")
            return False

        return True
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra không gian đĩa: {e}")
        # Giả định là không đủ không gian trong trường hợp lỗi
        return False


def get_fallback_model_path(model_type: str) -> Optional[str]:
    """
    Lấy đường dẫn đến mô hình dự phòng trong trường hợp mô hình chính không tải được

    Args:
        model_type: Loại mô hình

    Returns:
        Optional[str]: Đường dẫn đến mô hình dự phòng
    """
    model_configs = get_model_config()
    if model_type not in model_configs:
        return None

    model_dir = Path(settings.MODEL_DIR)
    os.makedirs(model_dir, exist_ok=True)

    # Thử tìm mô hình dự phòng
    if "fallback_model_name" in model_configs[model_type]:
        fallback_name = model_configs[model_type]["fallback_model_name"]
        fallback_path = model_dir / fallback_name.split('/')[-1]
        if fallback_path.exists():
            return str(fallback_path)

    return None


configure_gpu()