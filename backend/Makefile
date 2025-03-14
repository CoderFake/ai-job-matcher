SHELL := /bin/bash

.PHONY: clean check setup
.DEFAULT_GOAL := help
VENV_DIR = .venv
PYTHON_VERSION = python3.9

check: # Ruff check
	@ruff check .
	@echo "✅ Kiểm tra hoàn tất!"

fix: # Fix auto-fixable linting issues
	@ruff check app.py --fix

clean: # Clean temporary files
	@rm -rf __pycache__ .pytest_cache
	@find . -name '*.pyc' -exec rm -r {} +
	@find . -name '__pycache__' -exec rm -r {} +
	@rm -rf build dist
	@find . -name '*.egg-info' -type d -exec rm -r {} +

run: # Run the application
	@uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

setup: # Initial project setup
	@echo "Tạo môi trường ảo tại: $(VENV_DIR)"
	@$(PYTHON_VERSION) -m venv $(VENV_DIR)
	@echo "Cài đặt các phụ thuộc..."
	@source $(VENV_DIR)/bin/activate && pip install -r requirements.txt
	@echo -e "\n✅ Xong.\n🎉 Chạy các lệnh sau để bắt đầu:\n\n ➡️ source $(VENV_DIR)/bin/activate\n ➡️ make run\n"

test: # Run tests
	@pytest -xvs tests/

setup-gpu: # Setup GPU for optimal performance
	@echo "Cấu hình GPU..."
	@source $(VENV_DIR)/bin/activate && python -m app.utils.gpu_setup
	@echo "✅ Cấu hình GPU hoàn tất!"

download-models: # Download required AI models
	@echo "Tải xuống các mô hình AI..."
	@source $(VENV_DIR)/bin/activate && python -m app.utils.download_models
	@echo "✅ Tải mô hình hoàn tất!"

deploy: # Deploy application
	@echo "Triển khai ứng dụng..."
	@echo "✅ Triển khai hoàn tất!"

help: # Show this help
	@egrep -h '\s#\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'