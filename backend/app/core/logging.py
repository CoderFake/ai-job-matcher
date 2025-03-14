import sys
import logging
import logging.handlers
import os
from pathlib import Path

# Thư mục logs
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Cấu hình log
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Thiết lập các logger riêng biệt cho các thành phần khác nhau
LOGGERS = {
    "main": {"file": "main.log", "level": logging.INFO},
    "api": {"file": "api.log", "level": logging.INFO},
    "cv_parser": {"file": "cv_parser.log", "level": logging.INFO},
    "job_search": {"file": "job_search.log", "level": logging.INFO},
    "models": {"file": "models.log", "level": logging.INFO},
    "errors": {"file": "errors.log", "level": logging.ERROR},
}


def setup_logging():
    """Thiết lập hệ thống ghi log"""

    # Xóa tất cả các handler có sẵn
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Cấu hình logger gốc
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.handlers.RotatingFileHandler(
                LOG_DIR / "app.log",
                maxBytes=10485760,  # 10MB
                backupCount=5,
                encoding="utf-8",
            ),
        ],
    )

    # Thiết lập các logger cụ thể
    for name, config in LOGGERS.items():
        logger = logging.getLogger(name)
        logger.setLevel(config["level"])

        # Thêm file handler cho logger
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / config["file"],
            maxBytes=5242880,  # 5MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logger.addHandler(file_handler)

        # Đảm bảo logger không lan truyền đến logger cha
        logger.propagate = False

    # Thiết lập logger lỗi đặc biệt
    error_logger = logging.getLogger("errors")
    error_handler = logging.FileHandler(LOG_DIR / "errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    error_logger.addHandler(error_handler)

    # Thiết lập bộ xử lý lỗi không bắt được
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Xử lý các exception không bắt được và ghi log"""
        if issubclass(exc_type, KeyboardInterrupt):
            # Đối với KeyboardInterrupt, sử dụng hành vi mặc định
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        error_logger.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

    # Thiết lập excepthook
    sys.excepthook = handle_exception

    # Ghi log khởi động
    logging.getLogger("main").info("Hệ thống logging đã được khởi tạo")


def get_logger(name):
    """
    Lấy logger cho một thành phần cụ thể

    Args:
        name: Tên của logger (main, api, cv_parser, job_search, models, errors)

    Returns:
        logging.Logger: Logger đã được cấu hình
    """
    if name in LOGGERS:
        return logging.getLogger(name)
    return logging.getLogger("main")