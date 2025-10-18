# src/llm/client.py
"""
Module này quản lý việc khởi tạo và truy cập đến Large Language Model (LLM).
Nó hỗ trợ cấu hình thông qua biến môi trường và đảm bảo rằng chỉ một
instance của LLM được tạo ra (singleton pattern) để tiết kiệm tài nguyên.
"""

import os
from dotenv import load_dotenv
from langchain_core.language_models.llms import LLM
from langchain_ollama import OllamaLLM

load_dotenv()

# Biến toàn cục để lưu trữ instance của LLM
_llm_instance = None

def get_llm(temperature: float = 0.0) -> LLM:
    """
    Lấy một instance của LLM đã được cấu hình.
    Sử dụng singleton pattern để tránh khởi tạo lại mô hình.
    Args:
        temperature (float): "Nhiệt độ" của mô hình, kiểm soát sự sáng tạo.
                             0.0 cho câu trả lời nhất quán, >0 cho sự đa dạng.
    Returns:
        Một instance của LLM (ví dụ: OllamaLLM).
    """
    global _llm_instance
    
    # Chỉ khởi tạo nếu chưa có instance nào
    if _llm_instance is None:
        llm_type = os.getenv("LLM_TYPE", "ollama")
        
        if llm_type == "ollama":
            model_name = os.getenv("CHAT_MODEL", "qwen2.5:3b")
            print(f"Đang khởi tạo Ollama LLM với model: {model_name}...")
            _llm_instance = OllamaLLM(
                model=model_name,
                temperature=temperature,
            )
            print("✅ Khởi tạo LLM thành công.")
        else:
            raise ValueError(f"Loại LLM '{llm_type}' không được hỗ trợ.")
            
    # Cập nhật temperature nếu được yêu cầu
    if hasattr(_llm_instance, 'temperature') and _llm_instance.temperature != temperature:
        _llm_instance.temperature = temperature

    return _llm_instance
