"""
Module tương tác với mô hình LLM cục bộ
"""

import os
import logging
import tempfile
from typing import Dict, Any, List, Optional
import json

from app.core.logging import get_logger
from app.core.config import get_model_config

logger = get_logger("models")


class LocalLLM:
    """
    Lớp tương tác với mô hình LLM cục bộ
    """

    def __init__(self):
        """
        Khởi tạo mô hình LLM
        """
        self.model_config = get_model_config()
        self.model_type = "auto"  # auto, ollama, llama_cpp
        self._initialize_model()

    def _initialize_model(self):
        """
        Khởi tạo mô hình LLM
        """
        try:
            # Thử sử dụng Ollama nếu có
            if self._is_ollama_available():
                logger.info("Sử dụng Ollama để tương tác với mô hình LLM")
                self.model_type = "ollama"
                return

            # Thử sử dụng llama.cpp nếu có
            if self._is_llama_cpp_available():
                logger.info("Sử dụng llama.cpp để tương tác với mô hình LLM")
                self.model_type = "llama_cpp"
                return

            # Sử dụng transformers nếu có GPU
            if self.model_config.get('device') == 'cuda' and self._is_transformers_available():
                logger.info("Sử dụng transformers để tương tác với mô hình LLM")
                self.model_type = "transformers"
                return

            logger.warning("Không thể khởi tạo bất kỳ model LLM nào. Sẽ sử dụng mô phỏng đơn giản.")
            self.model_type = "mock"

        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo mô hình LLM: {str(e)}")
            self.model_type = "mock"

    def _is_ollama_available(self) -> bool:
        """
        Kiểm tra xem Ollama có khả dụng không

        Returns:
            bool: True nếu có thể sử dụng Ollama
        """
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")

            if response.status_code == 200:
                # Lấy danh sách mô hình có sẵn
                available_models = response.json().get("models", [])

                # Kiểm tra mô hình cần thiết
                llm_models = ["gemma", "llama", "mistral", "phi", "vicuna", "wizard"]

                for model in available_models:
                    model_name = model["name"].lower()
                    for llm_model in llm_models:
                        if llm_model in model_name:
                            self.ollama_model = model["name"]
                            logger.info(f"Tìm thấy mô hình Ollama: {self.ollama_model}")
                            return True

            return False
        except Exception as e:
            logger.warning(f"Không thể kết nối đến Ollama: {str(e)}")
            return False

    def _is_llama_cpp_available(self) -> bool:
        """
        Kiểm tra xem llama.cpp có khả dụng không

        Returns:
            bool: True nếu có thể sử dụng llama.cpp
        """
        try:
            from llama_cpp import Llama

            # Tìm các mô hình GGUF trong thư mục mô hình
            model_dir = self.model_config.get("model_dir", "models")
            gguf_files = [f for f in os.listdir(model_dir) if f.endswith(".gguf")]

            if gguf_files:
                self.llama_cpp_model_path = os.path.join(model_dir, gguf_files[0])
                logger.info(f"Tìm thấy mô hình GGUF: {self.llama_cpp_model_path}")
                return True

            return False
        except ImportError:
            logger.warning("Không thể import llama_cpp")
            return False
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra llama.cpp: {str(e)}")
            return False

    def _is_transformers_available(self) -> bool:
        """
        Kiểm tra xem có thể sử dụng transformers cho LLM không

        Returns:
            bool: True nếu có thể sử dụng transformers
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Kiểm tra GPU
            if not torch.cuda.is_available():
                logger.warning("Transformers yêu cầu GPU")
                return False

            return True
        except ImportError:
            logger.warning("Không thể import transformers hoặc torch")
            return False
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra transformers: {str(e)}")
            return False

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        """
        Tạo văn bản dựa trên prompt

        Args:
            prompt: Lời nhắc
            max_tokens: Số lượng token tối đa
            temperature: Nhiệt độ (độ sáng tạo)

        Returns:
            str: Văn bản được tạo
        """
        try:
            if self.model_type == "ollama":
                return self._generate_with_ollama(prompt, max_tokens, temperature)
            elif self.model_type == "llama_cpp":
                return self._generate_with_llama_cpp(prompt, max_tokens, temperature)
            elif self.model_type == "transformers":
                return self._generate_with_transformers(prompt, max_tokens, temperature)
            else:
                # Sử dụng mô phỏng đơn giản nếu không có mô hình nào khả dụng
                return self._generate_mock(prompt)

        except Exception as e:
            logger.error(f"Lỗi khi tạo văn bản: {str(e)}")
            return f"Lỗi khi tạo văn bản: {str(e)}"

    def _generate_with_ollama(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """
        Tạo văn bản sử dụng Ollama

        Args:
            prompt: Lời nhắc
            max_tokens: Số lượng token tối đa
            temperature: Nhiệt độ (độ sáng tạo)

        Returns:
            str: Văn bản được tạo
        """
        try:
            import requests

            # Chuẩn bị tin nhắn
            data = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }

            # Gửi yêu cầu
            response = requests.post("http://localhost:11434/api/generate", json=data)

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Lỗi khi gọi Ollama API: {response.status_code} - {response.text}")
                return f"Lỗi: {response.status_code} - {response.text}"

        except Exception as e:
            logger.error(f"Lỗi khi tạo văn bản với Ollama: {str(e)}")
            return f"Lỗi khi tạo văn bản với Ollama: {str(e)}"

    def _generate_with_llama_cpp(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """
        Tạo văn bản sử dụng llama.cpp

        Args:
            prompt: Lời nhắc
            max_tokens: Số lượng token tối đa
            temperature: Nhiệt độ (độ sáng tạo)

        Returns:
            str: Văn bản được tạo
        """
        try:
            from llama_cpp import Llama

            # Khởi tạo model nếu chưa có
            if not hasattr(self, "llama_model"):
                self.llama_model = Llama(
                    model_path=self.llama_cpp_model_path,
                    n_ctx=2048,
                    n_threads=os.cpu_count() // 2
                )

            # Tạo văn bản
            result = self.llama_model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "\n\n\n"]
            )

            return result["choices"][0]["text"]

        except Exception as e:
            logger.error(f"Lỗi khi tạo văn bản với llama.cpp: {str(e)}")
            return f"Lỗi khi tạo văn bản với llama.cpp: {str(e)}"

    def _generate_with_transformers(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """
        Tạo văn bản sử dụng transformers

        Args:
            prompt: Lời nhắc
            max_tokens: Số lượng token tối đa
            temperature: Nhiệt độ (độ sáng tạo)

        Returns:
            str: Văn bản được tạo
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            # Khởi tạo pipeline nếu chưa có
            if not hasattr(self, "pipe"):
                model_id = "google/gemma-2b-it"  # Mô hình nhỏ để chạy được trên nhiều GPU

                # Sử dụng BF16 nếu GPU hỗ trợ, không thì dùng FP16
                if torch.cuda.is_bf16_supported():
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float16

                self.pipe = pipeline(
                    "text-generation",
                    model=model_id,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                )

            # Tạo văn bản
            sequences = self.pipe(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=0.95,
                top_k=50,
                pad_token_id=self.pipe.tokenizer.eos_token_id
            )

            return sequences[0]['generated_text'][len(prompt):]

        except Exception as e:
            logger.error(f"Lỗi khi tạo văn bản với transformers: {str(e)}")
            return f"Lỗi khi tạo văn bản với transformers: {str(e)}"

    def _generate_mock(self, prompt: str) -> str:
        """
        Mô phỏng tạo văn bản để testing khi không có mô hình

        Args:
            prompt: Lời nhắc

        Returns:
            str: Văn bản được tạo
        """
        # Kiểm tra nếu prompt yêu cầu trả về JSON
        if "json" in prompt.lower():
            # Tìm các mẫu JSON trong prompt
            import re
            json_pattern = r'\{.*\}'
            json_matches = re.findall(json_pattern, prompt, re.DOTALL)

            if json_matches:
                # Trả về JSON mẫu từ prompt
                return json_matches[0]
            else:
                # Tạo JSON mẫu đơn giản
                sample_json = {
                    "personal_info": {
                        "name": "Nguyễn Văn A",
                        "email": "nguyenvana@example.com",
                        "phone": "0987654321"
                    },
                    "education": [
                        {
                            "institution": "Đại học Bách khoa Hà Nội",
                            "degree": "Cử nhân",
                            "field_of_study": "Công nghệ thông tin"
                        }
                    ],
                    "skills": [
                        {"name": "Python"},
                        {"name": "JavaScript"},
                        {"name": "Machine Learning"}
                    ]
                }
                return json.dumps(sample_json, ensure_ascii=False, indent=2)

        # Trả về phản hồi đơn giản cho trường hợp khác
        return "Đây là câu trả lời mẫu vì không có mô hình LLM nào được tải."

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 2048, temperature: float = 0.7) -> str:
        """
        Tạo văn bản dựa trên cuộc trò chuyện

        Args:
            messages: Danh sách tin nhắn theo định dạng [{"role": "user", "content": "..."}, ...]
            max_tokens: Số lượng token tối đa
            temperature: Nhiệt độ (độ sáng tạo)

        Returns:
            str: Văn bản được tạo
        """
        try:
            if self.model_type == "ollama":
                return self._chat_with_ollama(messages, max_tokens, temperature)
            else:
                # Chuyển đổi messages thành prompt đơn giản cho các model khác
                prompt = self._convert_messages_to_prompt(messages)
                return self.generate(prompt, max_tokens, temperature)

        except Exception as e:
            logger.error(f"Lỗi khi chat: {str(e)}")
            return f"Lỗi khi chat: {str(e)}"

    def _chat_with_ollama(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        """
        Chat sử dụng Ollama

        Args:
            messages: Danh sách tin nhắn
            max_tokens: Số lượng token tối đa
            temperature: Nhiệt độ (độ sáng tạo)

        Returns:
            str: Văn bản được tạo
        """
        try:
            import requests

            # Chuẩn bị tin nhắn
            data = {
                "model": self.ollama_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }

            # Gửi yêu cầu
            response = requests.post("http://localhost:11434/api/chat", json=data)

            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                logger.error(f"Lỗi khi gọi Ollama API: {response.status_code} - {response.text}")
                return f"Lỗi: {response.status_code} - {response.text}"

        except Exception as e:
            logger.error(f"Lỗi khi chat với Ollama: {str(e)}")
            return f"Lỗi khi chat với Ollama: {str(e)}"

    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Chuyển đổi danh sách tin nhắn thành prompt

        Args:
            messages: Danh sách tin nhắn

        Returns:
            str: Prompt
        """
        prompt = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"Human: {content}\n\n"
            elif role == "assistant":
                prompt += f"AI: {content}\n\n"
            else:
                prompt += f"{content}\n\n"

        prompt += "AI: "
        return prompt