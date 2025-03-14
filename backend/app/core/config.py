"""
Cấu hình cho ứng dụng AI Job Matcher
"""

import os
import torch
from typing import Dict, Any
from app.core.settings import settings


def configure_gpu():
    """
    Cấu hình GPU tự động dựa trên phần cứng có sẵn
    Tối ưu hóa cho máy tính 8GB RAM
    """
    if not settings.AUTO_CONFIG_GPU:
        return

    try:
        # Kiểm tra GPU có khả dụng không
        if not torch.cuda.is_available():
            settings.USE_GPU = False
            print("GPU không khả dụng, sử dụng CPU.")
            return

        # Kiểm tra VRAM của GPU
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024  # Chuyển đổi sang GB

        # Kiểm tra RAM hệ thống
        import psutil
        ram_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024  # Chuyển đổi sang GB

        print(f"Phát hiện GPU với {gpu_memory:.2f}GB VRAM")
        print(f"RAM hệ thống: {ram_gb:.2f}GB")

        # Cấu hình dựa trên phần cứng
        settings.USE_GPU = True

        # Nếu VRAM ít, sử dụng cấu hình thấp
        if gpu_memory < 4:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        # Sử dụng half precision nếu GPU hỗ trợ
        if torch.cuda.is_bf16_supported():
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            print("Sử dụng BF16 precision")
        elif torch.cuda.is_fp16_supported():
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            print("Sử dụng FP16 precision")

        # Giới hạn bộ nhớ đệm PyTorch
        if ram_gb <= 8:
            # Giới hạn bộ nhớ đệm cho máy tính 8GB RAM
            torch.cuda.empty_cache()
            # Cài đặt giới hạn bộ nhớ đệm ở mức thấp
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
            print("Đã áp dụng cấu hình cho máy tính RAM thấp (8GB hoặc ít hơn)")

    except Exception as e:
        print(f"Lỗi khi cấu hình GPU: {e}")
        settings.USE_GPU = False


def get_model_config() -> Dict[str, Any]:
    """
    Trả về cấu hình cho các mô hình dựa trên phần cứng có sẵn
    """
    device = "cuda" if settings.USE_GPU else "cpu"

    # Cấu hình cơ bản cho tất cả các mô hình
    base_config = {
        "device": device,
        "model_dir": settings.MODEL_DIR,
    }

    # Cấu hình cụ thể cho từng loại mô hình
    model_configs = {
        "sentence_transformer": {
            **base_config,
            "model_name": "paraphrase-multilingual-MiniLM-L12-v2",  # Mô hình đa ngôn ngữ nhẹ
            "max_seq_length": 256,
        },
        "named_entity_recognition": {
            **base_config,
            "model_name": "dslim/bert-base-NER-uncased",
        },
        "text_classification": {
            **base_config,
            "model_name": "distilbert-base-uncased",
        },
        "spacy": {
            "model_name": "vi_core_news_lg",  # Mô hình tiếng Việt
        }
    }

    # Nếu RAM thấp, sử dụng mô hình nhỏ hơn
    import psutil
    ram_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024

    if ram_gb <= 8:
        model_configs["sentence_transformer"]["model_name"] = "paraphrase-multilingual-MiniLM-L6-v2"
        model_configs["named_entity_recognition"]["model_name"] = "dslim/bert-small-NER-uncased"
        model_configs["text_classification"]["model_name"] = "distilbert-base-uncased-finetuned-sst-2-english"
        model_configs["spacy"]["model_name"] = "vi_core_news_md"  # Mô hình tiếng Việt nhỏ hơn

    return model_configs


# Thực hiện cấu hình GPU khi module được import
configure_gpu()