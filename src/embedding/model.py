# src/embedding/model.py
"""
Module này định nghĩa class `EmbeddingModel` để xử lý việc tạo vector embeddings
cho văn bản. Nó đóng gói mô hình SentenceTransformer và cung cấp một giao diện
đơn giản để mã hóa văn bản.
"""
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

class EmbeddingModel:
    """
    Wrapper cho mô hình SentenceTransformer để tạo embeddings.
    """
    def __init__(self, model_name: str = None):
        """
        Khởi tạo và tải mô hình embedding.
        Args:
            model_name (str): Tên của mô hình từ Hugging Face.
                              Nếu không được cung cấp, sẽ lấy từ biến môi trường.
        """
        if model_name is None:
            model_name = os.getenv("DENSE_MODEL", "intfloat/multilingual-e5-base")
        
        # Thư mục cache model để tránh tải lại
        cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
        
        print(f"Đang tải embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        print("✅ Tải embedding model thành công.")

    def encode(self, texts: list[str] | str, batch_size: int = 32) -> list[list[float]] | list[float]:
        """
        Mã hóa một hoặc nhiều đoạn văn bản thành vector.
        Args:
            texts (list[str] | str): Văn bản cần mã hóa.
            batch_size (int): Kích thước batch khi xử lý danh sách văn bản.
        Returns:
            Vector embedding hoặc danh sách các vector.
        """
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

    def get_dimension(self) -> int:
        """Trả về số chiều của vector embedding."""
        return self.model.get_sentence_embedding_dimension()

# Singleton pattern để đảm bảo chỉ có một instance của model được tải
_embedding_model_instance = None

def get_embedding_model() -> EmbeddingModel:
    """Trả về một instance duy nhất của EmbeddingModel."""
    global _embedding_model_instance
    if _embedding_model_instance is None:
        _embedding_model_instance = EmbeddingModel()
    return _embedding_model_instance
