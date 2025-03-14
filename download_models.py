
import os
import sys
import logging
from pathlib import Path
import requests
import zipfile
import tempfile
import argparse
import subprocess
from tqdm import tqdm

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("download_models")

# Thư mục mô hình
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def download_file(url, output_path):
    """Tải file từ URL với thanh tiến trình"""
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(
        desc=f"Tải {os.path.basename(output_path)}",
        total=total_size_in_bytes,
        unit='iB',
        unit_scale=True
    )

    with open(output_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()


def download_spacy_model():
    """Tải mô hình spaCy tiếng Việt"""
    model_name = "vi_core_news_md"
    logger.info(f"Đang tải mô hình spaCy {model_name}...")

    try:
        # Kiểm tra xem mô hình đã được cài đặt chưa
        import spacy
        try:
            # Nếu tải được mô hình, nghĩa là đã cài đặt
            spacy.load(model_name)
            logger.info(f"Mô hình spaCy {model_name} đã được cài đặt!")
            return True
        except:
            # Cài đặt mô hình
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            logger.info(f"Đã tải mô hình spaCy {model_name} thành công!")
            return True
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình spaCy: {str(e)}")
        return False


def download_sentence_transformer():
    """Tải mô hình sentence-transformers đa ngôn ngữ"""
    model_name = "paraphrase-multilingual-MiniLM-L6-v2"
    model_path = os.path.join(MODEL_DIR, model_name)

    # Kiểm tra xem mô hình đã tồn tại chưa
    if os.path.exists(model_path) and len(os.listdir(model_path)) > 0:
        logger.info(f"Mô hình {model_name} đã tồn tại!")
        return True

    logger.info(f"Đang tải mô hình Sentence Transformer {model_name}...")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        model.save(model_path)
        logger.info(f"Đã tải mô hình {model_name} thành công!")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình Sentence Transformer: {str(e)}")
        return False


def download_tinyllama():
    """Tải mô hình TinyLlama GGUF"""
    model_name = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    model_path = os.path.join(MODEL_DIR, model_name)

    # Kiểm tra xem mô hình đã tồn tại chưa
    if os.path.exists(model_path):
        logger.info(f"Mô hình TinyLlama đã tồn tại!")
        return True

    logger.info("Đang tải mô hình TinyLlama GGUF...")

    try:
        url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        download_file(url, model_path)
        logger.info("Đã tải mô hình TinyLlama thành công!")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình TinyLlama: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Tải các mô hình cho AI Job Matcher')
    parser.add_argument('--all', action='store_true', help='Tải tất cả các mô hình')
    parser.add_argument('--spacy', action='store_true', help='Tải mô hình spaCy tiếng Việt')
    parser.add_argument('--sentence-transformer', action='store_true', help='Tải mô hình Sentence Transformer')
    parser.add_argument('--tinyllama', action='store_true', help='Tải mô hình TinyLlama GGUF')

    args = parser.parse_args()

    # Nếu không có tham số, mặc định tải tất cả
    if not any(vars(args).values()):
        args.all = True

    if args.all or args.spacy:
        download_spacy_model()

    if args.all or args.sentence_transformer:
        download_sentence_transformer()

    if args.all or args.tinyllama:
        download_tinyllama()

    logger.info("Quá trình tải mô hình hoàn tất!")


if __name__ == "__main__":
    main()